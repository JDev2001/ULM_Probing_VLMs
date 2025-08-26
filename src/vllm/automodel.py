# Inspired by: https://github.com/jammastergirish/LLMProbe

# -*- coding: utf-8 -*-
from PIL import Image
from typing import Dict, List, Optional, Tuple, Union
import time
import math
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from src.utils.image_utils import resize_image_aspect_ratio
from src.vllm.vllm import VLLM
from qwen_vl_utils import process_vision_info
from transformers import AutoConfig, AutoModelForCausalLM
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
    """

    def __init__(
        self,
        model_name: str = "microsoft/Phi-4-multimodal-instruct",
        device: Union[str, torch.device] = "cuda",
        device_map: Union[str, Dict[str, int]] = "auto",
        torch_dtype: Union[str, torch.dtype] = "auto"
    ):
        # Device bookkeeping
        self.device = device

        self.model_name = model_name

        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        config.attn_implementation = "sdpa" # disable flash attention

        # Load model and processor via Auto* to stay generic
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            attn_implementation="sdpa", #disable flash attention
            config=config,
            offload_folder="offload",  # legt eine Auslagerung auf Disk an
            offload_state_dict=True,
            #load_in_8bit=True

        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        # Basic model stats for logging/shape checks
        self.n_layers = getattr(self.model.config, "num_hidden_layers", None)
        self.d_model = getattr(self.model.config, "hidden_size", None)

    def _apply_chat_template_safe(self, messages: List[Dict]) -> str:
        """Apply chat template if available; otherwise fall back to a simple join.

        Messages format example:
          [{"role":"user","content":"..."} , {"role":"assistant","content":"..."}]
        """
        # Prefer processor.apply_chat_template if it exists
        if hasattr(self.processor, "apply_chat_template"):
            try:
                return self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
            except Exception as e:
                print("Error applying chat template:", e)
                pass

        # Fallback: minimal, neutral concatenation
        parts = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            parts.append(f"{role}: {content}")
        return "\n".join(parts)

    def adapt_messages_to_phi(self,messages):
        new_messages = []
        images = []
        img_counter = 1

        for msg in messages:
            parts = []
            if isinstance(msg["content"], list):
                for seg in msg["content"]:
                    if seg["type"] == "text":
                        parts.append(seg["text"])
                    elif seg["type"] == "image":
                        parts.append(f"<|image_{img_counter}|>")
                        images.append(seg["image"])
                        img_counter += 1
            elif isinstance(msg["content"], str):
                parts.append(msg["content"])

            new_messages.append({
                "role": msg["role"],
                "content": " ".join(parts).strip()
            })

        return new_messages, images


    @torch.no_grad()
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
        dev = self.device

        n_layers = self.n_layers or 0
        d_model = self.d_model or 0

        num_batches = math.ceil(len(examples) / batch_size)
        all_hidden_states: List[torch.Tensor] = []
        all_labels: List[int] = []

        for batch_start in tqdm(range(0, len(examples), batch_size)):
            batch_end = min(batch_start + batch_size, len(examples))
            batch = examples[batch_start:batch_end]

            # Collect labels
            batch_labels = [int(ex["label"]) for ex in batch]

            # Build batched inputs for the processor
            batch_texts: List[str] = []
            batch_images: List[list] = []
            batch_videos: List[list] = []

            for ex in batch:
                if "messages" in ex and isinstance(ex["messages"], list):
                    msgs = ex["messages"]
                    if self.model_name.lower().__contains__("phi-4"):
                        adapted, img = self.adapt_messages_to_phi(msgs)
                        chat_text = self._apply_chat_template_safe(adapted)
                    else:
                        chat_text = self._apply_chat_template_safe(msgs)
                    imgs, vids = process_vision_info(msgs)
                    imgs = [resize_image_aspect_ratio(img, target_size=300) for img in imgs]
                    batch_texts.append(chat_text)
                    batch_images.append(imgs if imgs is not None else [])
                    batch_videos.append(vids if vids is not None else [])
                else:
                    text = str(ex.get("text", ""))
                    batch_texts.append(text)
                    batch_images.append([])
                    batch_videos.append([])

            # Only pass multimodal fields if any sample actually contains them
            any_images = any(len(x) > 0 for x in batch_images)
            any_videos = any(len(x) > 0 for x in batch_videos)

            processor_kwargs = dict(
                text=batch_texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            if any_images:
                processor_kwargs["images"] = batch_images
                if self.model_name.lower().__contains__("phi-4"):
                    processor_kwargs["images"] = [item[0] for item in batch_images]  # phi-4 expects a batch of images (no list of list)
            if any_videos:
                processor_kwargs["videos"] = batch_videos

            inputs = self.processor(**processor_kwargs).to(dev)

            # Forward with hidden states
            with torch.no_grad():
                outputs = self.model(
                    **inputs,
                    output_hidden_states=True,
                    return_dict=True,
                )
            hidden_states = outputs.hidden_states  # tuple: (embeddings, layer1, ..., layerN)

            B, S, H = hidden_states[0].shape

            # Infer sequence lengths from attention mask if present
            if "attention_mask" in inputs:
                seq_lens = inputs["attention_mask"].sum(dim=1)  # [B]
            else:
                seq_lens = torch.full((B,), S, dtype=torch.long, device=hidden_states[0].device)

            # Pooling per example per layer
            batch_hidden_pooled: List[torch.Tensor] = []
            for b in range(B):
                example_layers: List[torch.Tensor] = []
                seqlen_b = int(seq_lens[b].item())
                for layer_tensor in hidden_states:
                    tokens_b = layer_tensor[b]                 # [S, H]
                    seqlen_eff = max(1, min(seqlen_b, tokens_b.shape[0]))

                    if output_layer == "CLS":
                        token_repr = tokens_b[0, :]
                    elif output_layer == "mean":
                        token_repr = tokens_b[:seqlen_eff, :].mean(dim=0)
                    elif output_layer == "max":
                        token_repr = tokens_b[:seqlen_eff, :].max(dim=0).values
                    elif isinstance(output_layer, str) and output_layer.startswith("token_index_"):
                        idx = int(output_layer.split("_")[-1])
                        safe_idx = min(max(0, idx), seqlen_eff - 1)
                        token_repr = tokens_b[safe_idx, :]
                    else:
                        token_repr = tokens_b[seqlen_eff - 1, :]

                    example_layers.append(token_repr)

                example_stack = torch.stack(example_layers, dim=0)  # [L, H]
                batch_hidden_pooled.append(example_stack)

            all_hidden_states.extend(batch_hidden_pooled)
            all_labels.extend(batch_labels)

            # Small sleep to emulate incremental UI feedback
            time.sleep(0.01)

        all_hidden_states_tensor = torch.stack(all_hidden_states, dim=0)  # [N, L, H]
        all_labels_tensor = torch.tensor(all_labels, device=all_hidden_states_tensor.device)

        if return_layer is not None:
            return all_hidden_states_tensor[:, return_layer, :], all_labels_tensor
        return all_hidden_states_tensor, all_labels_tensor
