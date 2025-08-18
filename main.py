from ast import List
from src.vllm.qwen import QwenVLProbe

    # create simple hf da
from datasets import Dataset

from src.vllm.automodel import AutoModelVLM


def main():

    #probe = QwenVLProbe(model_name="Qwen/Qwen2-VL-2B-Instruct", device="cpu")
    # probe = AutoModelVLM(model_name="google/gemma-3-4b-it",device="cpu")
    # probe = AutoModelVLM(model_name="microsoft/Phi-4-mini-instruct", device="cpu")

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "https://users.wfu.edu/matthews/misc/graphics/ResVsComp/25percent4bcx2px2.jpg",
                },
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]
    messages = [
        {"label": 0, "messages": messages}
    ]

    hidden_out, label_out = probe.get_hidden_states_batched(
        examples=messages,
        output_layer="CLS",
        return_layer=None,
        batch_size=8
    )
    print(hidden_out.shape)  # should be (batch_size, n_layers, d_model)
    # output_text[:, 8, :] select layer 8

if __name__ == "__main__":
    main()
