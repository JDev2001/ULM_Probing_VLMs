# Inspired by: https://github.com/jammastergirish/LLMProbe
from typing import Dict, List, Optional, Tuple, Union
import time
import math
import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

"""
Abstract base class for VLLMs
"""
class VLLM():


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
        raise NotImplementedError("This method should be implemented by subclasses")
