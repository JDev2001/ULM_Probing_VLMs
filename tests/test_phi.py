from src.vllm.automodel import AutoModelVLM


def test_phi():

    probe = AutoModelVLM(model_name="microsoft/Phi-4-multimodal-instruct", device="cpu")

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
    assert hidden_out.shape[0]==1
