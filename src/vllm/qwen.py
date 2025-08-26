# Inspired by: https://github.com/jammastergirish/LLMProbe
from PIL import Image
from typing import Dict, List, Optional, Tuple, Union
import time
import math
import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from src.vllm.vllm import VLLM
from qwen_vl_utils import process_vision_info
from tqdm import tqdm


class QwenVLProbe(VLLM):
    """
    Qwen2-VL wrapper that provides:
      - forward_with_hidden: single-sample generation + hidden states
      - get_hidden_states_batched: batched hidden-state extraction with pooling,
        returning the same shapes as the reference get_hidden_states_batched().
    """

    def __init__(self, model_name: str = "Qwen/Qwen2-VL-2B-Instruct", device: Union[str, torch.device] = "cuda"):
        self.device = device
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map=device
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model_name)

        # Basic model stats (for progress/logging)
        self.n_layers = getattr(self.model.config, "num_hidden_layers", None)
        self.d_model  = getattr(self.model.config, "hidden_size", None)

    @torch.no_grad()
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
        # Use provided device for tensors
        dev = self.device

        # Basic model stats for consistency with the reference
        n_layers = self.n_layers if self.n_layers is not None else 0
        d_model  = self.d_model  if self.d_model  is not None else 0

        # Batching meta
        num_batches = math.ceil(len(examples) / batch_size)

        all_hidden_states: List[torch.Tensor] = []
        all_labels: List[int] = []

        # Iterate over batches
        for batch_start in tqdm(range(0, len(examples), batch_size)):
            batch_end = min(batch_start + batch_size, len(examples))
            batch = examples[batch_start:batch_end]
            batch_idx = batch_start // batch_size + 1

            # Collect labels
            batch_labels = [int(ex["label"]) for ex in batch]

            # Build processor inputs for the whole batch
            batch_texts: List[str] = []
            batch_images: List[list] = []
            batch_videos: List[list] = []

            for ex in batch:
                if "messages" in ex and isinstance(ex["messages"], list):
                    msgs = ex["messages"]
                    chat_text = self.processor.apply_chat_template(
                        msgs, tokenize=False, add_generation_prompt=False
                    )
                    imgs, vids = process_vision_info(msgs)
                    #imgs = [img.resize((img.width//4,img.height//4),Image.Resampling.LANCZOS) for img in imgs]  
                    # Ensure per-sample container lists; processor expects lists aligned with `text`
                    batch_texts.append(chat_text)
                    batch_images.append(imgs if imgs is not None else [])
                    batch_videos.append(vids if vids is not None else [])
                else:
                    # Text-only fallback
                    text = ex.get("text", "")
                    batch_texts.append(str(text))
                    batch_images.append([])
                    batch_videos.append([])

            # Prepare model inputs
            inputs = self.processor(
                text=batch_texts,
                images=batch_images if any(len(x) > 0 for x in batch_images) else None,
                videos=batch_videos if any(len(x) > 0 for x in batch_videos) else None,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(dev)

            # Forward pass with hidden states
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)
            # Tuple of tensors: (embeddings, layer1, ..., layerN)
            hidden_states = outputs.hidden_states

            # For each example, compute a pooled representation per layer
            # Shapes: hidden_states[k] -> [B, S, H]
            B = hidden_states[0].shape[0]
            S = hidden_states[0].shape[1]
            H = hidden_states[0].shape[2]

            # Use attention mask to find sequence lengths (non-padding)
            if "attention_mask" in inputs:
                seq_lens = inputs["attention_mask"].sum(dim=1)  # [B]
            else:
                # Fallback: assume full length (no padding)
                seq_lens = torch.full((B,), S, dtype=torch.long, device=dev)

            # Pooling per example per layer
            batch_hidden_pooled: List[torch.Tensor] = []
            for b in range(B):
                example_layers: List[torch.Tensor] = []

                for layer_idx, layer_tensor in enumerate(hidden_states):
                    # layer_tensor[b]: [S, H]
                    tokens_b = layer_tensor[b]  # [S, H]
                    seqlen_b = int(seq_lens[b].item())
                    seqlen_b = max(1, min(seqlen_b, tokens_b.shape[0]))

                    if output_layer == "CLS":
                        # First token (CLS-like)
                        token_repr = tokens_b[0, :]
                    elif output_layer == "mean":
                        token_repr = tokens_b[:seqlen_b, :].mean(dim=0)
                    elif output_layer == "max":
                        token_repr = tokens_b[:seqlen_b, :].max(dim=0).values
                    elif isinstance(output_layer, str) and output_layer.startswith("token_index_"):
                        idx = int(output_layer.split("_")[-1])
                        safe_idx = min(max(0, idx), seqlen_b - 1)
                        token_repr = tokens_b[safe_idx, :]
                    else:
                        # Default (decoder-only style): last non-padding token
                        token_repr = tokens_b[seqlen_b - 1, :]

                    example_layers.append(token_repr)

                # Stack to [num_layers(+1), H]
                example_stack = torch.stack(example_layers, dim=0)  # [L, H]
                batch_hidden_pooled.append(example_stack)

            # Collect batch
            all_hidden_states.extend(batch_hidden_pooled)
            all_labels.extend(batch_labels)

            # Tiny sleep to simulate UI updates like in the reference
            time.sleep(0.01)

        # Stack across all examples
        all_hidden_states_tensor = torch.stack(all_hidden_states, dim=0).to(dev)  # [N, L, H]
        all_labels_tensor = torch.tensor(all_labels, device=dev)

        # Return according to API
        if return_layer is not None:
            # Select specific layer: [N, H]
            return all_hidden_states_tensor[:, return_layer, :], all_labels_tensor
        else:
            # Full stack: [N, L, H]
            return all_hidden_states_tensor, all_labels_tensor
