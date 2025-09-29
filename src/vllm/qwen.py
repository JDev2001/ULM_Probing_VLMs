# Inspired by: https://github.com/jammastergirish/LLMProbe


from typing import Dict, List, Optional, Tuple, Union
import math
import time
import torch
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from src.utils.experiment_utils import pool_tokens
from src.utils.image_utils import resize_image_aspect_ratio
from src.utils.pre_materialize import _prematerialize_examples_inline
from src.vllm.vllm import VLLM
from tqdm import tqdm


class QwenVLProbe(VLLM):
    """
    Qwen2-VL wrapper that provides:
      - forward_with_hidden: single-sample generation + hidden states
      - get_hidden_states_batched: batched hidden-state extraction with pooling,
        returning the same shapes as the reference get_hidden_states_batched().

    The class now performs vision pre-materialization (downloads, resize, chat templating)
    inline during the first call, without changing the external API.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-VL-2B-Instruct",
        device: Union[str, torch.device] = "cuda",
        *,
        max_length: int = 256,   # conservative cap to limit quadratic attention cost
        use_sdpa: bool = True,   # use PyTorch SDPA path (stable fast path on 4090)
    ):
        self.device = device
        self.max_length = int(max_length)
        self.model_name = model_name

        # Prefer BF16 on 4090; fall back to FP16 on non-bfloat hardware
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16

        # Enable fast matmul on Ada
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

        attn_impl = "sdpa" if use_sdpa else "eager"

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device,
            attn_implementation=attn_impl,
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model_name, use_fast=True)

        # Basic model stats (for progress/logging)
        self.n_layers = getattr(self.model.config, "num_hidden_layers", None)
        self.d_model  = getattr(self.model.config, "hidden_size", None)

    # --------------------------- Batching & pooling ---------------------------


    def _prepare_batch(
        self,
        batch: List[Dict],
    ) -> Tuple[Dict[str, torch.Tensor], List[int]]:
        """Build processor inputs for a list of examples. Uses pre-materialized fields (no I/O here)."""
        batch_texts: List[str] = []
        batch_images: List[List[Image.Image]] = []
        batch_videos: List[List] = []
        batch_labels: List[int] = []

        for ex in batch:
            batch_labels.append(int(ex["label"]))

            pre_text = ex.get("_pre_chat_text", None)
            pre_imgs = ex.get("_pre_images", None)
            pre_vids = ex.get("_pre_videos", None)

            # Pre-materialized path must be available at this point
            batch_texts.append(pre_text if pre_text is not None else str(ex.get("text", "")))
            batch_images.append(pre_imgs or [])
            batch_videos.append(pre_vids or [])

        use_images = any(len(x) > 0 for x in batch_images)
        use_videos = any(len(x) > 0 for x in batch_videos)

        processor_kwargs = dict(
            text=batch_texts,
            padding="longest",
            truncation=True,
            return_tensors="pt",
            max_length=self.max_length,
        )
        if use_images:
            processor_kwargs["images"] = batch_images
        if use_videos:
            processor_kwargs["videos"] = batch_videos

        inputs = self.processor(**processor_kwargs)
        inputs = {k: v.to(self.device, non_blocking=True) if hasattr(v, "to") else v for k, v in inputs.items()}
        return inputs, batch_labels


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
              where the first layer corresponds to embeddings.
          - If return_layer is not None:
              returns (layer_hidden_states [N, H], all_labels [N])

        Supported example formats:
          - {"messages": [...], "label": int}  # multimodal (preferred)
          - {"text": "...", "label": int}      # text-only convenience
        """

        # Pre-materialize (only once per call; no API change)
        examples = _prematerialize_examples_inline(examples,self.processor)

        all_hidden_states: List[torch.Tensor] = []
        all_labels: List[int] = []

        # Iterate in batches (hot path: GPU-bound only)
        for batch_start in tqdm(range(0, len(examples), batch_size), desc="Batches", leave=False):
            batch_end = min(batch_start + batch_size, len(examples))
            batch = examples[batch_start:batch_end]

            inputs, batch_labels = self._prepare_batch(batch)
            all_labels.extend(batch_labels)

            outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)
            hidden_states = outputs.hidden_states  # tuple: (embeddings, layer1, ..., layerN)

            attn_mask = inputs.get("attention_mask", None)

            pooled_layers: List[torch.Tensor] = []
            for layer_tensor in hidden_states:
                pooled = pool_tokens(layer_tensor, attn_mask, output_layer)  # [B, H]
                pooled_layers.append(pooled)

            batch_stack = torch.stack(pooled_layers, dim=1)  # [B, L, H]
            all_hidden_states.append(batch_stack)

        all_hidden_states_tensor = torch.cat(all_hidden_states, dim=0)  # [N, L, H]
        all_labels_tensor = torch.tensor(all_labels, device=all_hidden_states_tensor.device)

        if return_layer is not None:
            return all_hidden_states_tensor[:, return_layer, :], all_labels_tensor
        else:
            return all_hidden_states_tensor, all_labels_tensor
