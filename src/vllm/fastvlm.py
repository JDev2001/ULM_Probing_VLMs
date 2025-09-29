# Inspired by: https://github.com/jammastergirish/LLMProbe


from typing import Dict, List, Optional, Tuple, Union
import time
import torch
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import AutoTokenizer, AutoModelForCausalLM
from qwen_vl_utils import process_vision_info
from src.utils.experiment_utils import pool_tokens
from src.utils.image_utils import resize_image_aspect_ratio
from src.utils.pre_materialize import _prematerialize_examples_inline
from src.vllm.vllm import VLLM
from tqdm import tqdm


class FastVLM(VLLM):
    """
    FastVLM wrapper that provides:
      - get_hidden_states_batched: batched hidden-state extraction with pooling,
        returning the same shapes as the reference get_hidden_states_batched().

    This class performs vision pre-materialization (downloads, resize, chat templating)
    inline during the first call, without changing the external API.
    """

    def __init__(
        self,
        model_name: str = "apple/FastVLM-0.5B",
        device: Union[str, torch.device] = "cuda",
        *,
        max_length: int = 256,   # conservative cap to limit quadratic attention cost
        use_sdpa: bool = True,   # kept for parity; FastVLM uses standard torch attention
    ):
        self.device = device
        self.max_length = int(max_length)
        self.model_name = model_name


        # Dtype/device setup
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

        # Load FastVLM (trust_remote_code required for vision tower helpers)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()

        # Model stats (if available)
        cfg = getattr(self.model, "config", None)
        self.n_layers = getattr(cfg, "n_layer", None) or getattr(cfg, "num_hidden_layers", None)
        self.d_model  = getattr(cfg, "n_embd", None) or getattr(cfg, "hidden_size", None)

        # Constants from model interface
        self.IMAGE_TOKEN_INDEX = -200

        # Fallback pad token handling
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is not None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            else:
                # Create a pad token if absolutely necessary
                self.tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
                try:
                    self.model.resize_token_embeddings(len(self.tokenizer))
                except Exception:
                    pass


    # --------------------------- Tokenization & batching (FastVLM-specific) ---------------------------

    def _splice_image_token_and_tokenize(
        self,
        rendered_prompt: str,
        has_image: bool
    ) -> torch.Tensor:
        """
        For FastVLM, we must place IMAGE_TOKEN_INDEX exactly where '<image>' appears.
        We tokenize text around it without extra specials (add_special_tokens=False).
        Returns 1D tensor of input_ids.
        """
        if has_image:
            assert "<image>" in rendered_prompt, "Rendered chat must contain <image> once for image inputs."
            pre, post = rendered_prompt.split("<image>", 1)
            pre_ids = self.tokenizer(pre, return_tensors="pt", add_special_tokens=False).input_ids[0]
            post_ids = self.tokenizer(post, return_tensors="pt", add_special_tokens=False).input_ids[0]
            img_tok = torch.tensor([self.IMAGE_TOKEN_INDEX], dtype=pre_ids.dtype)
            input_ids = torch.cat([pre_ids, img_tok, post_ids], dim=0)
        else:
            # Text-only: use standard tokenization (no special tokens to mirror the image path)
            input_ids = self.tokenizer(rendered_prompt, return_tensors="pt", add_special_tokens=False).input_ids[0]
        return input_ids

    def _build_subbatch_inputs(
        self,
        batch_examples: List[Dict],
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Build padded input_ids/attention_mask and optional images tensor for a homogeneous sub-batch
        (either all have an image or none have an image).
        """
        device = self.model.device
        # Tokenize with proper splicing
        ids_list: List[torch.Tensor] = []
        has_image_list: List[bool] = []

        for ex in batch_examples:
            rendered = ex.get("_pre_chat_text", "") or ""
            has_img = bool(ex.get("_pre_images", []))
            ids = self._splice_image_token_and_tokenize(rendered, has_img)
            ids_list.append(ids)
            has_image_list.append(has_img)

        # Left-pad to longest (or right-pad; consistent padding is fine since we build attention_mask)
        max_len = max(t.size(0) for t in ids_list) if ids_list else 1
        pad_id = int(self.tokenizer.pad_token_id)
        input_ids = torch.full((len(ids_list), max_len), pad_id, dtype=ids_list[0].dtype)
        for i, t in enumerate(ids_list):
            input_ids[i, :t.size(0)] = t
        attention_mask = (input_ids != pad_id).to(torch.long)

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        # Build images tensor if any
        if any(has_image_list):
            # FastVLM uses its own image processor from the vision tower
            vision_proc = self.model.get_vision_tower().image_processor
            # Use at most one image per example (FastVLM typical path)
            pil_list: List[Image.Image] = []
            for ex in batch_examples:
                imgs = ex.get("_pre_images", [])
                pil_list.append(imgs[0] if imgs else None)

            blank = Image.new("RGB", (2, 2), (255, 255, 255))
            pil_list = [img if img is not None else blank for img in pil_list]

            px = vision_proc(images=pil_list, return_tensors="pt")["pixel_values"]
            px = px.to(device, dtype=self.model.dtype)
        else:
            px = None

        return input_ids, attention_mask, px



    @torch.inference_mode()
    def get_hidden_states_batched(
        self,
        examples: List[Dict],
        output_layer: str,                           # "CLS" | "mean" | "max" | "token_index_X" | default(decoder last token)
        return_layer: Optional[int] = None,
        batch_size: int = 8,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract hidden states in batches, mirroring the reference function's return format:
          - If return_layer is None:
              returns (all_hidden_states [N, num_layers(+1), H], all_labels [N])
              where the first layer corresponds to embeddings (if provided by model).
          - If return_layer is not None:
              returns (layer_hidden_states [N, H], all_labels [N])

        Supported example formats:
          - {"messages": [...], "label": int}  # multimodal (preferred; expects one <image>)
          - {"text": "...", "label": int}      # text-only convenience
        """
        # Pre-materialize
        examples = _prematerialize_examples_inline(examples, self.tokenizer,"fast_vlm")

        # Iterate in batches; within each batch, split by image presence to satisfy FastVLM's images= tensor
        all_hidden_states: List[torch.Tensor] = []
        all_labels: List[int] = []

        for batch_start in tqdm(range(0, len(examples), batch_size), desc="Batches", leave=False):
            batch_end = min(batch_start + batch_size, len(examples))
            batch = examples[batch_start:batch_end]

            # Partition into two stable sub-batches (with images / without)
            idx_with_img = [i for i, ex in enumerate(batch) if len(ex.get("_pre_images", [])) > 0]
            idx_no_img   = [i for i in range(len(batch)) if i not in idx_with_img]

            sub_results: List[Optional[torch.Tensor]] = [None] * len(batch)
            sub_labels: List[int] = [int(ex["label"]) for ex in batch]

            for index_group in (idx_no_img, idx_with_img):
                if not index_group:
                    continue
                sub_ex = [batch[i] for i in index_group]
                input_ids, attention_mask, images_px = self._build_subbatch_inputs(sub_ex)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    images=images_px if images_px is not None else None,
                    output_hidden_states=True,
                    return_dict=True,
                )
                hidden_states = outputs.hidden_states  # tuple: (layers...) for causal LM

                # adjust mask length to hidden_states length ---
                attn_mask = attention_mask
                S_model = hidden_states[0].shape[1]
                S_mask = attn_mask.shape[1]
                if S_model != S_mask:
                    B = attn_mask.shape[0]
                    # new mask with pad tokens
                    new_mask = torch.ones((B, S_model), dtype=attn_mask.dtype, device=attn_mask.device)
                    # copy old mask values
                    new_mask[:, :min(S_model, S_mask)] = attn_mask[:, :min(S_model, S_mask)]
                    attn_mask = new_mask

                # Some models do not include embeddings layer; handle generically
                pooled_layers: List[torch.Tensor] = []
                for layer_tensor in hidden_states:
                    pooled = pool_tokens(layer_tensor, attn_mask, output_layer)  # [B, H]
                    pooled_layers.append(pooled)
                batch_stack = torch.stack(pooled_layers, dim=1)  # [B, L, H]

                # Scatter back to original sub-positions
                for j, pos in enumerate(index_group):
                    sub_results[pos] = batch_stack[j]

            # Concatenate along batch dimension in original order
            batch_out = torch.stack([sr for sr in sub_results if sr is not None], dim=0)  # [B, L, H]
            all_hidden_states.append(batch_out)
            all_labels.extend(sub_labels)

        all_hidden_states_tensor = torch.cat(all_hidden_states, dim=0)  # [N, L, H]
        all_labels_tensor = torch.tensor(all_labels, device=all_hidden_states_tensor.device)

        if return_layer is not None:
            return all_hidden_states_tensor[:, return_layer, :], all_labels_tensor
        else:
            return all_hidden_states_tensor, all_labels_tensor
