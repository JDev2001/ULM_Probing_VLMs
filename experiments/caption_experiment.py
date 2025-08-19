import os, sys
from src.vllm.qwen import QwenVLProbe
from src.vllm.automodel import AutoModelVLM
import torch
from tqdm import tqdm
from src.utils.experiment_utils import  get_repr, get_repr_for_layer, train_probe, load_captions_prompt, load_caption_ds


ds = load_caption_ds()


device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "Qwen/Qwen2.5-3B-Instruct"

models = {
    "exp1/Qwen_Qwen2-VL-2B-Instruct": QwenVLProbe(model_name="Qwen/Qwen2-VL-2B-Instruct", device=device),
    "exp2/google_gemma-3-4b-it": AutoModelVLM(model_name="google/gemma-3-4b-it",device=device),
    "exp3/microsoft_Phi-4-multimodal-instruct": AutoModelVLM(model_name="microsoft/Phi-4-multimodal-instruct", device=device)
}

for model_name, model in models.items():

    print(f"Processing {model_name}")

    reprs = []
    labels = []

    ds = ds.shuffle().select(range(10000)) # cause of computational limitations, we just consider the first 10.000 entries

    #num_dataset_items = len(ds)
    num_dataset_items = 1 #test mode

    print("Computing representations for positive examples")
    for i in tqdm(range(num_dataset_items)):
        prompt = load_captions_prompt().format(caption=ds[i]['caption_pos'])
        hidden_out, _ = get_repr(model,prompt, ds[i]['url'])
        reprs.append(hidden_out)
        labels.append(1)

    print("Computing representations for negative examples")
    for i in tqdm(range(num_dataset_items)):
        prompt = load_captions_prompt().format(caption=ds[i]['caption_neg'])
        hidden_out, _ = get_repr(model,prompt, ds[i]['url'])
        reprs.append(hidden_out)
        labels.append(0)

    print("Training probes for each layer")
    for layer in tqdm(range(reprs[0].shape[1])):
        layer_repr = [get_repr_for_layer(h,layer)for h in reprs] #e.g Layer 3
        run_name = f"{model_name}_Test_layer{layer}"
        train_probe(layer_repr, labels, run_name)
