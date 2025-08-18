"""
Interface for VLLMs
"""
from typing import Dict, List, Tuple
import torch

"""
Abstract base class for VLLMs
"""
class VLLM():
    def load_model(self, model_name):
        """
        Load a model by its name.

        Args:
            model_name (str): The name of the model to load.

        Returns:
            bool: True if the model was loaded successfully, False otherwise.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def forward_with_hidden(
        self,
        messages: List[dict],
        max_new_tokens: int = 64,
    ) ->  Tuple[str, Dict[str, List[torch.Tensor]]]:
        """
        Run the model and collect hidden states via the standard HF output.

        Args:
            messages: Chat template messages (with "image"/"text" entries).
            max_new_tokens: For generation mode.
            use_generate: If True, also run model.generate() to produce text (hidden states are taken from the forward pass).
            capture_attention_and_mlp: Unused (kept for compatibility).

        Returns:
            Tuple[str, Dict[str, List[torch.Tensor]]]
            - output_text: The generated text output from the model.
            - hidden_states: {"decoder.hidden_states": [T0, T1, ..., TL]} where T0 is embeddings and TL is the last layer output.
        """
