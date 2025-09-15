import os, sys
import torch
from tqdm import tqdm
import gc # Import the garbage collection module

from src.utils.store_representations import save_repr, save_labels, load_representations, load_labels
from src.vllm.fastvlm import FastVLM
from src.vllm.qwen import QwenVLProbe
from src.vllm.automodel import AutoModelVLM
from src.utils.experiment_utils import  get_repr_batch, get_repr_for_layer, train_probe_local, load_category_ds, load_categories_prompt

# Added for sample negative objects
import random

USE_OFFLINE_REPR = True

num_dataset_train = 20000
num_dataset_eval = 2000

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset, categories = load_category_ds()

model_configs = {
    "exp2_1/Qwen_Qwen2-VL-2B-Instruct": ("Qwen/Qwen2-VL-2B-Instruct", QwenVLProbe),
    "exp2_2/google_gemma-3-4b-it": ("google/gemma-3-4b-it", AutoModelVLM),
    "exp2_3/apple_fast_vlm": ("apple/FastVLM-0.5B", FastVLM)
}


# Added to sample local categories to make a list of 5 categories for each image
def sample_categories(pos_categories, neg_categories, sample_size=5):
    num_pos_categories = len(pos_categories)
    num_neg_to_sample = sample_size - num_pos_categories

    selected_neg_categories = random.sample(neg_categories, num_neg_to_sample)
    selected_categories = pos_categories + selected_neg_categories
    random.shuffle(selected_categories)

    true_labels = [1 if category in set(pos_categories).intersection(selected_categories) else 0 for category in categories]
    mask_sampled_candidates = [1 if category in set(selected_categories) else 0 for category in categories]

    return list(selected_categories), true_labels, mask_sampled_candidates


for experiment_name, (model_hf_name, model_class) in model_configs.items():

    print("----------------------------------------------------")
    print(f"Processing {experiment_name}")
    print("----------------------------------------------------")

    if USE_OFFLINE_REPR:
        repr_train = load_representations("artifacts/repr", experiment_name, "train")
        reprs_eval = load_representations("artifacts/repr", experiment_name, "eval")

        labels_train = []
        labels_eval = []

        for i in tqdm(range(num_dataset_train)):
            labels_train.append(1)

        for i in tqdm(range(num_dataset_eval)):
            labels_eval.append(1)

    else:
        ds = dataset
        ds = ds.train_test_split(0.1)
        ds_train = ds['train']
        ds_eval = ds['test']

        print(f"Loading model: {model_hf_name}...")
        model = model_class(model_name=model_hf_name, device=device)
        print("Model loaded.")

        repr_train = []
        labels_train = []
        prompts_train = []
        imgs_train = []

        reprs_eval = []
        labels_eval = []
        prompts_eval = []
        imgs_eval = []

        # Added to shuffle the dataset of local semantics
        ds_train_sample = ds_train.shuffle().select(range(20000))
        ds_eval_sample = ds_eval.shuffle().select(range(2000))

        # Added to calculate the length of sampled dataset of local semantics
        num_dataset_train = len(ds_train_sample)
        num_dataset_eval = len(ds_eval_sample)

        print("Preparing train data...")

        for i in tqdm(range(num_dataset_train)):
            sampled_categories, label_vector, mask_vector = sample_categories(ds_train_sample[i]['pos_categories'], ds_train_sample[i]['neg_categories'])
            prompt = load_categories_prompt().format(category=', '.join(sampled_categories))
            prompts_train.append(prompt)

            imgs_train.append(ds_train_sample[i]['url'])

            labels_train.append((label_vector, mask_vector))

        print("Preparing eval data...")
        for i in tqdm(range(num_dataset_eval)):
            sampled_categories, label_vector, mask_vector = sample_categories(ds_eval_sample[i]['pos_categories'], ds_eval_sample[i]['neg_categories'])
            prompt = load_categories_prompt().format(category=', '.join(sampled_categories))
            prompts_eval.append(prompt)

            imgs_eval.append(ds_eval_sample[i]['url'])

            labels_eval.append((label_vector, mask_vector))


        print("Computing representations for train examples")
        repr_train_batched, _ = get_repr_batch(model, prompts_train, imgs_train)

        print("Computing representations for eval examples")
        reprs_eval_batched, _ = get_repr_batch(model, prompts_eval, imgs_eval)

        for i in range(len(repr_train_batched)):
            repr_train.append(repr_train_batched[i])

        for i in range(len(reprs_eval_batched)):
            reprs_eval.append(reprs_eval_batched[i])


        save_labels(
            labels=labels_train,
            base_path="artifacts/labels",
            experiment_name=experiment_name,
            split_name="train"
        )
        save_labels(
            labels=labels_eval,
            base_path="artifacts/labels",
            experiment_name=experiment_name,
            split_name="eval"
        )

        save_repr(
            representations=repr_train,
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
    for layer in tqdm(range(repr_train[0].shape[0]), desc=f"Training Probes for {experiment_name}"):
        layer_repr = [get_repr_for_layer(h, layer) for h in repr_train]
        layer_repr_eval = [get_repr_for_layer(h, layer) for h in reprs_eval]
        run_name = f"{experiment_name}_Test_layer{layer}"
        train_probe_local(layer_repr, labels_train, layer_repr_eval, labels_eval, run_name)

    print(f"Finished processing {experiment_name}. Releasing resources.")
    del model
    del repr_train
    del reprs_eval


    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache() # Release cached memory on the GPU
    print("Resources released. Moving to next model.\n")
