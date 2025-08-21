import os, sys
from src.vllm.qwen import QwenVLProbe
from src.vllm.automodel import AutoModelVLM
import torch
from tqdm import tqdm
from src.utils.experiment_utils import  get_repr, get_repr_for_layer, train_probe, load_captions_prompt, load_caption_ds


ds = load_caption_ds()
ds = ds.train_test_split(0.1)
ds_train = ds['train']
ds_eval = ds['test']

print(len(ds_train), len(ds_eval))


device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "Qwen/Qwen2.5-3B-Instruct"

models = {
    "test": QwenVLProbe(model_name="Qwen/Qwen2-VL-2B-Instruct", device=device),
    # "exp1/Qwen_Qwen2-VL-2B-Instruct": QwenVLProbe(model_name="Qwen/Qwen2-VL-2B-Instruct", device=device),
    # "exp2/google_gemma-3-4b-it": AutoModelVLM(model_name="google/gemma-3-4b-it",device=device),
    # "exp3/microsoft_Phi-4-multimodal-instruct": AutoModelVLM(model_name="microsoft/Phi-4-multimodal-instruct", device=device)
}

for model_name, model in models.items():

    print(f"Processing {model_name}")

    reprs = []
    labels = []

    reprs_eval = []
    labels_eval = []

    ds_train = ds_train.shuffle().select(range(10000)) # cause of computational limitations, we just consider random 10.000 entries
    ds_eval = ds_eval.shuffle().select(range(10000)) # cause of computational limitations, we just consider random 10000 entries

    num_dataset_train = len(ds_train)
    num_dataset_eval = len(ds_eval)

    num_dataset_train = 1 #test mode
    num_dataset_eval = 1 #test mode

    print("Computing representations for train examples")
    for i in tqdm(range(num_dataset_train)):
        prompt = load_captions_prompt().format(caption=ds_train[i]['caption_pos'])
        hidden_out, _ = get_repr(model,prompt, ds_train[i]['url'])
        reprs.append(hidden_out)
        labels.append(1)

        prompt = load_captions_prompt().format(caption=ds_train[i]['caption_neg'])
        hidden_out, _ = get_repr(model,prompt, ds_train[i]['url'])
        reprs.append(hidden_out)
        labels.append(0)


    print("Computing representations for eval examples")
    for i in tqdm(range(num_dataset_eval)):

        prompt = load_captions_prompt().format(caption=ds_eval[i]['caption_pos'])
        hidden_out, _ = get_repr(model,prompt, ds_eval[i]['url'])
        reprs_eval.append(hidden_out)
        labels_eval.append(1)


        prompt = load_captions_prompt().format(caption=ds_eval[i]['caption_neg'])
        hidden_out, _ = get_repr(model,prompt, ds_eval[i]['url'])
        reprs_eval.append(hidden_out)
        labels_eval.append(0)



    print("Training probes for each layer")
    for layer in tqdm(range(reprs[0].shape[1])):
        layer_repr = [get_repr_for_layer(h,layer)for h in reprs] #e.g Layer 3
        layer_repr_eval = [get_repr_for_layer(h,layer)for h in reprs_eval]
        run_name = f"{model_name}_Test_layer{layer}"
        train_probe(layer_repr, labels, layer_repr_eval, labels_eval, run_name)
