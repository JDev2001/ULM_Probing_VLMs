
# -*- coding: utf-8 -*-
# QwenVLProbe with fully in-class pre-materialization (no API change).
# - Automatically pre-downloads/processes vision inputs in parallel on first call.
# - Keeps the same public interface: __init__ and get_hidden_states_batched(...).
# - English identifiers, English comments, UTF-8.

from typing import Dict, List, Optional, Tuple, Union
import math
import time
import torch
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from src.utils.image_utils import resize_image_aspect_ratio
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
        premat_workers: int = 64,
        premat_retries: int = 2,
        premat_retry_sleep: float = 0.3,
        premat_target_img_size: int = 300,
        premat_progress: bool = True
    ):
        self.device = device
        self.max_length = int(max_length)
        self.model_name = model_name

        # Pre-materialization config
        self._premat_workers = int(premat_workers)
        self._premat_retries = int(premat_retries)
        self._premat_retry_sleep = float(premat_retry_sleep)
        self._premat_target_img_size = int(premat_target_img_size)
        self._premat_progress = bool(premat_progress)

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

    # --------------------------- Pre-materialization (inline/private) ---------------------------

    @staticmethod
    def _has_prematerialized_fields(ex: Dict) -> bool:
        """Check if an example already contains pre-materialized fields."""
        return ("_pre_chat_text" in ex) and ("_pre_images" in ex) and ("_pre_videos" in ex)

    def _process_single_example_premat(
        self,
        ex: Dict,
    ) -> Dict:
        """
        Process one example once: download images via process_vision_info, resize, cache chat template.
        Adds fields:
            - _pre_chat_text: str
            - _pre_images: List[PIL.Image.Image]
            - _pre_videos: List[Any]
        Robust against network/image errors.
        """
        out = ex  # in-place augmentation

        # Defaults
        out["_pre_images"] = []
        out["_pre_videos"] = []
        out["_pre_chat_text"] = None

        if "messages" in ex and isinstance(ex["messages"], list):
            # Cache chat template
            out["_pre_chat_text"] = self.processor.apply_chat_template(
                ex["messages"], tokenize=False, add_generation_prompt=False
            )
            # Vision: process with light retries
            imgs, vids = [], []
            last_err = None
            for _ in range(max(1, self._premat_retries)):
                try:
                    imgs, vids = process_vision_info(ex["messages"])
                    last_err = None
                    break
                except Exception as e:
                    last_err = e
                    time.sleep(self._premat_retry_sleep)
            if last_err is not None:
                imgs, vids = [], []

            # Resize once per image
            if imgs:
                try:
                    imgs = [resize_image_aspect_ratio(img, target_size=self._premat_target_img_size) for img in imgs]
                except Exception:
                    imgs = []

            out["_pre_images"] = imgs or []
            out["_pre_videos"] = vids or []
        else:
            # Text-only convenience
            text = str(ex.get("text", ""))
            out["_pre_chat_text"] = text
            out["_pre_images"] = []
            out["_pre_videos"] = []

        return out

    def _prematerialize_examples_inline(self, examples: List[Dict]) -> List[Dict]:
        """
        Run pre-materialization only if required (i.e., if _pre_* keys are missing).
        Uses a thread pool for I/O-bound work. Returns the same list with augmented dicts.
        """
        # Fast path: if all examples are already pre-materialized, return as-is
        if all(self._has_prematerialized_fields(ex) for ex in examples):
            return examples

        # Short lists: do it synchronously
        if len(examples) <= 2:
            return [self._process_single_example_premat(ex) for ex in examples]

        results: List[Optional[Dict]] = [None] * len(examples)
        to_process = [i for i, ex in enumerate(examples) if not self._has_prematerialized_fields(ex)]

        if not to_process:
            return examples

        with ThreadPoolExecutor(max_workers=self._premat_workers) as pool:
            futures = {
                pool.submit(self._process_single_example_premat, examples[i]): i
                for i in to_process
            }
            iterator = as_completed(futures)
            if self._premat_progress:
                iterator = tqdm(iterator, total=len(to_process), desc="Pre-materializing media", leave=False)

            for fut in iterator:
                i = futures[fut]
                try:
                    results[i] = fut.result()
                except Exception:
                    # On failure, fall back to text-only
                    ex = examples[i]
                    try:
                        pre_text = self.processor.apply_chat_template(
                            ex["messages"], tokenize=False, add_generation_prompt=False
                        ) if "messages" in ex and isinstance(ex["messages"], list) else str(ex.get("text", ""))
                    except Exception:
                        pre_text = str(ex.get("text", ""))
                    ex["_pre_chat_text"] = pre_text
                    ex["_pre_images"] = []
                    ex["_pre_videos"] = []
                    results[i] = ex

        # Merge results back; already pre-materialized examples remain unchanged
        for i in range(len(examples)):
            if results[i] is not None:
                examples[i] = results[i]
            elif not self._has_prematerialized_fields(examples[i]):
                # Ensure minimal keys are present
                ex = examples[i]
                try:
                    pre_text = self.processor.apply_chat_template(
                        ex["messages"], tokenize=False, add_generation_prompt=False
                    ) if "messages" in ex and isinstance(ex["messages"], list) else str(ex.get("text", ""))
                except Exception:
                    pre_text = str(ex.get("text", ""))
                ex["_pre_chat_text"] = pre_text
                ex["_pre_images"] = []
                ex["_pre_videos"] = []

        return examples

    # --------------------------- Batching & pooling ---------------------------

    @staticmethod
    def _pool_tokens(
        layer_tensor: torch.Tensor,         # [B, S, H]
        attn_mask: Optional[torch.Tensor],  # [B, S] or None
        mode: str,
    ) -> torch.Tensor:
        """Vectorized pooling across the batch. Returns: [B, H]."""
        B, S, H = layer_tensor.shape
        device = layer_tensor.device
        if attn_mask is None:
            attn_mask = torch.ones((B, S), dtype=torch.long, device=device)

        lengths = attn_mask.sum(dim=1).clamp_min(1)  # [B]

        if mode == "CLS":
            return layer_tensor[:, 0, :]

        if mode == "mean":
            mask = attn_mask.unsqueeze(-1)
            summed = (layer_tensor * mask).sum(dim=1)
            return summed / lengths.unsqueeze(-1)

        if mode == "max":
            very_neg = torch.finfo(layer_tensor.dtype).min
            masked = layer_tensor.clone()
            masked[attn_mask == 0] = very_neg
            return masked.max(dim=1).values

        if mode.startswith("token_index_"):
            try:
                idx = int(mode.split("_")[-1])
            except Exception:
                idx = 0
            clamped = torch.clamp(torch.full_like(lengths, idx), min=0, max=(lengths - 1))
            gather_index = clamped.view(B, 1, 1).expand(B, 1, H)
            return layer_tensor.gather(dim=1, index=gather_index).squeeze(1)

        # Default: last non-padding token
        last_idx = (lengths - 1).view(B, 1, 1).expand(B, 1, H)
        return layer_tensor.gather(dim=1, index=last_idx).squeeze(1)

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

    # --------------------------- Public API (unchanged) ---------------------------

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

        # Step 1: Pre-materialize (only once per call; no API change)
        examples = self._prematerialize_examples_inline(examples)

        all_hidden_states: List[torch.Tensor] = []
        all_labels: List[int] = []

        # Step 2: Iterate in batches (hot path: GPU-bound only)
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
                pooled = self._pool_tokens(layer_tensor, attn_mask, output_layer)  # [B, H]
                pooled_layers.append(pooled)

            batch_stack = torch.stack(pooled_layers, dim=1)  # [B, L, H]
            all_hidden_states.append(batch_stack)

        all_hidden_states_tensor = torch.cat(all_hidden_states, dim=0)  # [N, L, H]
        all_labels_tensor = torch.tensor(all_labels, device=all_hidden_states_tensor.device)

        if return_layer is not None:
            return all_hidden_states_tensor[:, return_layer, :], all_labels_tensor
        else:
            return all_hidden_states_tensor, all_labels_tensor
