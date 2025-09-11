import os, sys
import torch
from tqdm import tqdm
import gc # Import the garbage collection module

from src.utils.store_representations import save_repr, load_representations
from src.vllm.qwen import QwenVLProbe
from src.vllm.automodel import AutoModelVLM
from src.vllm.fastvlm import FastVLM
from src.utils.experiment_utils import  get_repr_batch, get_repr_for_layer, train_probe, load_captions_prompt, load_caption_ds


USE_OFFLINE_REPR = True # Set to True to use precomputed representations, False to compute on-the-fly (much slower and costly)

num_dataset_train = 20000
num_dataset_eval = 2000

device = "cuda" if torch.cuda.is_available() else "cpu"


model_configs = {
    "exp1_1/Qwen_Qwen2-VL-2B-Instruct": ("Qwen/Qwen2-VL-2B-Instruct", QwenVLProbe),
    "exp1_2/google_gemma-3-4b-it": ("google/gemma-3-4b-it", AutoModelVLM),
    "exp1_3/apple_fast_vlm": ("apple/FastVLM-0.5B", FastVLM)
}


for experiment_name, (model_hf_name, model_class) in model_configs.items():

    print("----------------------------------------------------")
    print(f"Processing {experiment_name}")
    print("----------------------------------------------------")

    if USE_OFFLINE_REPR:
        reprs_train = load_representations("artifacts/repr", experiment_name, "train")
        reprs_eval = load_representations("artifacts/repr", experiment_name, "eval")

        labels_train = []
        labels_eval = []

        for i in tqdm(range(num_dataset_train)):
            labels_train.append(1)
            labels_train.append(0)
        for i in tqdm(range(num_dataset_eval)):
            labels_eval.append(1)
            labels_eval.append(0)

    else:
        ds = load_caption_ds()
        ds = ds.train_test_split(0.1)
        ds_train = ds['train']
        ds_eval = ds['test']

        print(len(ds_train), len(ds_eval))

        print(f"Loading model: {model_hf_name}...")
        model = model_class(model_name=model_hf_name, device=device)
        print("Model loaded.")

        reprs_train = []
        labels_train = []
        prompts_train = []
        imgs_train = []

        reprs_eval = []
        labels_eval = []
        prompts_eval = []
        imgs_eval = []

        ds_train_sample = ds_train.shuffle().select(range(num_dataset_train))
        ds_eval_sample = ds_eval.shuffle().select(range(num_dataset_eval))


        if True: # Testmode
            num_dataset_train = 1
            num_dataset_eval = 1

        print("Preparing train data...")
        for i in tqdm(range(num_dataset_train)):
            prompt_pos = load_captions_prompt().format(caption=ds_train_sample[i]['caption_pos'])
            prompts_train.append(prompt_pos)
            imgs_train.append(ds_train_sample[i]['url'])
            labels_train.append(1)

            prompt_neg = load_captions_prompt().format(caption=ds_train_sample[i]['caption_neg'])
            prompts_train.append(prompt_neg)
            imgs_train.append(ds_train_sample[i]['url'])
            labels_train.append(0)

        print("Preparing eval data...")
        for i in tqdm(range(num_dataset_eval)):
            prompt_pos = load_captions_prompt().format(caption=ds_eval_sample[i]['caption_pos'])
            prompts_eval.append(prompt_pos)
            imgs_eval.append(ds_eval_sample[i]['url'])
            labels_eval.append(1)

            prompt_neg = load_captions_prompt().format(caption=ds_eval_sample[i]['caption_neg'])
            prompts_eval.append(prompt_neg)
            imgs_eval.append(ds_eval_sample[i]['url'])
            labels_eval.append(0)

        print("Computing representations for train examples")
        repr_train_batched, _ = get_repr_batch(model, prompts_train, imgs_train)

        print("Computing representations for eval examples")
        reprs_eval_batched, _ = get_repr_batch(model, prompts_eval, imgs_eval)


        for i in range(len(repr_train_batched)):
            reprs_train.append(repr_train_batched[i])

        for i in range(len(reprs_eval_batched)):
            reprs_eval.append(reprs_eval_batched[i])

        save_repr(
            representations=reprs_train,
            base_path="artifacts/repr",
            experiment_name=experiment_name,
            split_name="train"
        )
        save_repr(
            representations=reprs_eval,
            base_path="artifacts/repr",
            experiment_name=experiment_name,
            split_name="eval"
        )


    print("Training probes for each layer")
    for layer in tqdm(range(reprs_train[0].shape[0]), desc=f"Training Probes for {experiment_name}"):
        layer_repr = [get_repr_for_layer(h, layer) for h in reprs_train]
        layer_repr_eval = [get_repr_for_layer(h, layer) for h in reprs_eval]
        run_name = f"{experiment_name}_Test_layer{layer}"
        train_probe(layer_repr, labels_train, layer_repr_eval, labels_eval, run_name)


    print(f"Finished processing {experiment_name}. Releasing resources.")
    if not USE_OFFLINE_REPR:
        del model
    del reprs_train
    del reprs_eval


    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache() # Release cached memory on the GPU
    print("Resources released. Moving to next model.\n")
