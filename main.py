from ast import List
from src.vllm.qwen import QwenVLProbe
from src.vllm.load_model import get_hidden_states_batched, load_model_and_tokenizer
    # create simple hf da
from datasets import Dataset
def main():

    probe = QwenVLProbe(model_name="Qwen/Qwen2-VL-2B-Instruct", device="cpu")

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
        {"label": 0, "text":"Hallo"}
    ]

    hidden_out, label_out = probe.get_hidden_states_batched(
        examples=messages,
        output_layer="CLS",
        dataset_type="test",
        return_layer=None,
        progress_callback=None,
        batch_size=8,
        device="cpu"
    )
    # output_text[:, 8, :] select layer 8

if __name__ == "__main__":
    main()
