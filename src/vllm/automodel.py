# Inspired by: https://github.com/jammastergirish/LLMProbe



# -*- coding: utf-8 -*-
# Optimized drop-in replacement keeping the same public interface.
# Key improvements:
# - In-class pre-materialization: downloads/processes images & chat templates once (parallel), zero I/O in hot path.
# - Uses SDPA attention path, TF32 enabled on Ada, bfloat16 where available.
# - padding="longest" + conservative max_length to reduce quadratic attention cost.
# - Vectorized pooling (batch-wise), no per-token Python loops.
# - Robust Phi-4 handling (single-image expectation) with safe fallback.

from typing import Dict, List, Optional, Tuple, Union
import math
import time
import torch
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import AutoModelForCausalLM, AutoProcessor, AutoConfig
from src.utils.experiment_utils import pool_tokens
from src.utils.image_utils import resize_image_aspect_ratio
from src.utils.pre_materialize import _prematerialize_examples_inline
from src.vllm.vllm import VLLM
from qwen_vl_utils import process_vision_info
from tqdm import tqdm


class AutoModelVLM(VLLM):
    """
    Generic Auto* wrapper for causal LMs + processors (incl. multi-modal):
      - Uses AutoModelForCausalLM and AutoProcessor as per HF docs.
      - get_hidden_states_batched: batched hidden-state extraction with pooling.
      - Works with chat-style inputs (messages) and text-only inputs.

    Notes:
      * Assumes the processor can handle text/images/videos when provided.
      * If the processor does not implement `apply_chat_template`, we fall back
        to a minimal concatenation of role/content pairs.
      * Vision fetching/resizing and chat templating are pre-materialized inline (parallel) on first call.
    """

    def __init__(
        self,
        model_name: str = "microsoft/Phi-4-multimodal-instruct",
        device: Union[str, torch.device] = "cuda",
        device_map: Union[str, Dict[str, int]] = "auto",
        torch_dtype: Union[str, torch.dtype] = "auto",
        max_length: int = 256,   # conservative cap to limit quadratic attention cost
    ):
        # Device bookkeeping
        self.device = device
        self.model_name = model_name

        # Fast math on Ada (RTX 4090)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

        # Load config/model with SDPA attention path
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        config.attn_implementation = "sdpa"

        self._max_length = int(max_length)

        # Prefer BF16 on CUDA; FP16 otherwise
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=dtype,
            trust_remote_code=True,
            attn_implementation="sdpa",
            config=config,
        )
        self.model.eval()

        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=True
        )

        # Basic model stats for logging/shape checks
        self.n_layers = getattr(self.model.config, "num_hidden_layers", None)
        self.d_model = getattr(self.model.config, "hidden_size", None)



    def _adapt_messages_to_phi(self, messages: List[Dict]) -> Tuple[List[Dict], int]:
        """
        Produce Phi-4 compatible messages:
        - Replace image segments with <|image_k|> placeholders in the text stream.
        - Return adapted messages and the expected image count (ordering preserved).
        """
        new_messages: List[Dict] = []
        img_counter = 1
        image_slots = 0

        for msg in messages:
            parts = []
            content = msg.get("content", "")
            if isinstance(content, list):
                for seg in content:
                    if seg.get("type") == "text":
                        parts.append(seg.get("text", ""))
                    elif seg.get("type") == "image":
                        parts.append(f"<|image_{img_counter}|>")
                        img_counter += 1
                        image_slots += 1
            elif isinstance(content, str):
                parts.append(content)
            new_messages.append({"role": msg.get("role", "user"), "content": " ".join(parts).strip()})

        return new_messages, image_slots


    # --------------------------------- Batch prep ---------------------------------

    def _prepare_batch(self, batch: List[Dict]) -> Tuple[Dict[str, torch.Tensor], List[int]]:
        """
        Build processor inputs for a list of examples using pre-materialized fields.
        No network or PIL work here.
        """
        batch_texts: List[str] = []
        batch_images: List[List[Image.Image]] = []
        batch_videos: List[List] = []
        batch_labels: List[int] = []

        for ex in batch:
            batch_labels.append(int(ex["label"]))
            batch_texts.append(ex.get("_pre_chat_text", str(ex.get("text", ""))))
            batch_images.append(ex.get("_pre_images", []) or [])
            batch_videos.append(ex.get("_pre_videos", []) or [])

        any_images = any(len(x) > 0 for x in batch_images)
        any_videos = any(len(x) > 0 for x in batch_videos)

        processor_kwargs = dict(
            text=batch_texts,
            padding="longest",
            truncation=True,
            max_length=self._max_length,
            return_tensors="pt",
        )

        if any_images:
            if "phi-4" in self.model_name.lower():
                # Phi-4: expects a single image tensor per sample; provide first if multiple,
                # and a tiny placeholder for samples without an image to keep batch sizing consistent.
                phi_imgs: List[Image.Image] = []
                for imgs in batch_images:
                    if len(imgs) > 0:
                        phi_imgs.append(imgs[0])
                    else:
                        print("No image found, using placeholder.")
                        # Minimal placeholder to satisfy the batch structure
                        phi_imgs.append(Image.new("RGB", (2, 2), color=0))
                processor_kwargs["images"] = phi_imgs
            else:
                processor_kwargs["images"] = batch_images

        if any_videos:
            processor_kwargs["videos"] = batch_videos

        inputs = self.processor(**processor_kwargs)
        inputs = {k: v.to(self.device, non_blocking=True) if hasattr(v, "to") else v for k, v in inputs.items()}
        return inputs, batch_labels

    # --------------------------------- Public API ---------------------------------

    @torch.inference_mode()
    def get_hidden_states_batched(
        self,
        examples: List[Dict],
        output_layer: str,                 # "CLS" | "mean" | "max" | "token_index_X" | default(last token)
        return_layer: Optional[int] = None,
        batch_size: int = 8,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          - If return_layer is None:
              (all_hidden_states [N, L, H], all_labels [N]) where L = embeddings + num_layers
          - Else:
              (layer_hidden_states [N, H], all_labels [N])

        Supported example formats:
          - {"messages": [...], "label": int}   # preferred (chat/multimodal)
          - {"text": "...", "label": int}       # text-only convenience
        """
        # Step 0: Pre-materialize (downloads/resizes/templating) if needed
        examples = _prematerialize_examples_inline(examples, self.processor)

        all_hidden_states: List[torch.Tensor] = []
        all_labels: List[int] = []

        # Step 1: Iterate in batches (GPU-bound hot path)
        for batch_start in tqdm(range(0, len(examples), batch_size), desc="Batches", leave=False):
            batch_end = min(batch_start + batch_size, len(examples))
            batch = examples[batch_start:batch_end]

            inputs, batch_labels = self._prepare_batch(batch)
            all_labels.extend(batch_labels)

            outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)
            hidden_states = outputs.hidden_states  # tuple: (embeddings, layer1, ..., layerN), each [B, S, H]

            attn_mask = inputs.get("attention_mask", None)
            pooled_layers: List[torch.Tensor] = []
            for layer_tensor in hidden_states:
                pooled = pool_tokens(layer_tensor, attn_mask, output_layer)  # [B, H]
                pooled_layers.append(pooled)

            batch_stack = torch.stack(pooled_layers, dim=1)  # [B, L, H]
            all_hidden_states.append(batch_stack)

        # Step 2: Concatenate & format return
        all_hidden_states_tensor = torch.cat(all_hidden_states, dim=0)  # [N, L, H]
        all_labels_tensor = torch.tensor(all_labels, device=all_hidden_states_tensor.device)

        if return_layer is not None:
            return all_hidden_states_tensor[:, return_layer, :], all_labels_tensor
        else:
            return all_hidden_states_tensor, all_labels_tensor
