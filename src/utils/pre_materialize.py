# -*- coding: utf-8 -*-
# FastVLMProbe with fully in-class pre-materialization (no external API change).
# - Mirrors the Qwen2-VL wrapper's public API and behavior.
# - Adapts tokenization & vision path for apple/FastVLM-0.5B (<image> splice + IMAGE_TOKEN_INDEX).
# - English identifiers, English comments, UTF-8.

from typing import Dict, List, Optional, Tuple, Union
import time
import torch
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import AutoTokenizer, AutoModelForCausalLM
from qwen_vl_utils import process_vision_info
from src.utils.image_utils import resize_image_aspect_ratio
from src.vllm.vllm import VLLM
from tqdm import tqdm


_premat_workers = 64
_premat_retries = 2
_premat_retry_sleep = 0.3
_premat_target_img_size = 300
_premat_progress = True

def _adapt_messages_to_fast_vlm(messages: List[Dict]) -> Tuple[List[Dict], int]:
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
                        parts.append(f"<image>")
                        img_counter += 1
                        image_slots += 1
            elif isinstance(content, str):
                parts.append(content)
            new_messages.append({"role": msg.get("role", "user"), "content": " ".join(parts).strip()})

        return new_messages, image_slots


def _adapt_messages_to_phi(messages: List[Dict]) -> Tuple[List[Dict], int]:
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

def _has_prematerialized_fields(ex: Dict) -> bool:
    """Check if an example already contains pre-materialized fields."""
    return ("_pre_chat_text" in ex) and ("_pre_images" in ex) and ("_pre_videos" in ex)

def apply_chat_template(messages: List[Dict], *, add_generation_prompt: bool = False, tokenizer) -> str:
    """
    Render chat to a string via tokenizer.apply_chat_template.
    The rendered string must contain exactly one '<image>' if images are present.
    """
    rendered = tokenizer.apply_chat_template(
        messages, add_generation_prompt=add_generation_prompt, tokenize=False
    )
    return rendered

def _process_single_example_premat(ex: Dict, tokenizer, prompt_adaption="default") -> Dict:
    """
    Process one example once: download images via process_vision_info, resize, cache chat template.
    Adds fields:
        - _pre_chat_text: str
        - _pre_images: List[PIL.Image.Image]  (FastVLM effectively uses at most one image)
        - _pre_videos: List[Any]              (unused; kept for parity)
    Robust against network/image errors.
    """
    out = ex  # in-place augmentation
    out["_pre_images"] = []
    out["_pre_videos"] = []
    out["_pre_chat_text"] = None

    if "messages" in ex and isinstance(ex["messages"], list):
        # Cache chat template (no generation prompt needed for hidden-states encoding)
        try:
            if prompt_adaption == "fast_vlm":
                #adapted, _ = _adapt_messages_to_phi(ex["messages"])
                adapted, _ = _adapt_messages_to_fast_vlm(ex["messages"])
                rendered = apply_chat_template(adapted, add_generation_prompt=False, tokenizer=tokenizer)
            elif prompt_adaption == "phi":
                adapted, _ = _adapt_messages_to_phi(ex["messages"])
                rendered = apply_chat_template(adapted, add_generation_prompt=False, tokenizer=tokenizer)
            else:
                rendered = apply_chat_template(ex["messages"], add_generation_prompt=False, tokenizer=tokenizer)
        except Exception:
            # Fallback to simple join if templating fails
            rendered = "\n".join([str(m.get("content", "")) for m in ex["messages"]])
        out["_pre_chat_text"] = rendered

        # Vision: process with light retries
        imgs, vids = [], []
        last_err = None
        for _ in range(max(1, _premat_retries)):
            try:
                imgs, vids = process_vision_info(ex["messages"])
                last_err = None
                break
            except Exception as e:
                last_err = e
                time.sleep(_premat_retry_sleep)
        if last_err is not None:
            imgs, vids = [], []

        # Resize once per image (small target to reduce memory, FastVLM will reprocess)
        if imgs:
            try:
                imgs = [resize_image_aspect_ratio(img, target_size=_premat_target_img_size) for img in imgs]
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

def _prematerialize_examples_inline(examples: List[Dict],tokenizer, prompt_adaption="default") -> List[Dict]:
    """
    Run pre-materialization only if required (i.e., if _pre_* keys are missing).
    Uses a thread pool for I/O-bound work. Returns the same list with augmented dicts.
    Args:
        examples: List of examples to process.
        tokenizer: Tokenizer with apply_chat_template method.
        prompt_adaption: "default" | "phi" | "fast_vlm" -
    """
    if all(_has_prematerialized_fields(ex) for ex in examples):
        return examples

    if len(examples) <= 2:
        return [_process_single_example_premat(ex, tokenizer, prompt_adaption) for ex in examples]

    results: List[Optional[Dict]] = [None] * len(examples)
    to_process = [i for i, ex in enumerate(examples) if not _has_prematerialized_fields(ex)]
    if not to_process:
        return examples

    with ThreadPoolExecutor(max_workers=_premat_workers) as pool:
        futures = {
            pool.submit(_process_single_example_premat, examples[i], tokenizer, prompt_adaption): i
            for i in to_process
        }
        iterator = as_completed(futures)
        if _premat_progress:
            iterator = tqdm(iterator, total=len(to_process), desc="Pre-materializing media", leave=False)

        for fut in iterator:
            i = futures[fut]
            try:
                results[i] = fut.result()
            except Exception:
                # On failure, fall back to text-only
                ex = examples[i]
                try:
                    pre_text = apply_chat_template(ex["messages"], add_generation_prompt=False) \
                        if "messages" in ex and isinstance(ex["messages"], list) else str(ex.get("text", ""))
                except Exception:
                    pre_text = str(ex.get("text", ""))
                ex["_pre_chat_text"] = pre_text
                ex["_pre_images"] = []
                ex["_pre_videos"] = []
                results[i] = ex

    for i in range(len(examples)):
        if results[i] is not None:
            examples[i] = results[i]
        elif not _has_prematerialized_fields(examples[i]):
            ex = examples[i]
            try:
                pre_text = apply_chat_template(ex["messages"], add_generation_prompt=False) \
                    if "messages" in ex and isinstance(ex["messages"], list) else str(ex.get("text", ""))
            except Exception:
                pre_text = str(ex.get("text", ""))
            ex["_pre_chat_text"] = pre_text
            ex["_pre_images"] = []
            ex["_pre_videos"] = []

    return examples
