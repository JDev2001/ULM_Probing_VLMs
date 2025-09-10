import os
from typing import List

import torch
from tqdm import tqdm
import glob


from src.utils.experiment_utils import get_repr_for_layer


def save_repr(representations: List[torch.Tensor], base_path: str, experiment_name: str, split_name: str):

    if not representations:
        print(f"Warning: No representations to save for '{split_name}'. Skipping save.")
        return


    save_path = os.path.join(base_path, experiment_name, split_name)
    os.makedirs(save_path, exist_ok=True)
    print(f"Storing representations in '{save_path}'...")

    num_layers = representations[0].shape[0]


    for layer in tqdm(range(num_layers), desc=f"Storing layer for {split_name}"):

        layer_data = [get_repr_for_layer(h, layer) for h in representations]


        stacked_tensor = torch.stack(layer_data, dim=0)


        file_path = os.path.join(save_path, f"layer_{layer}.pt")
        torch.save(stacked_tensor, file_path)



def load_representations(base_path: str, experiment_name: str, split_name: str, device: str = "cpu") -> List[torch.Tensor]:

    load_path = os.path.join(base_path, experiment_name, split_name)


    if not os.path.isdir(load_path):
        print(f"Info: Path '{load_path}' does not exist. Returning empty list.")
        return []

    file_paths = glob.glob(os.path.join(load_path, "layer_*.pt"))
    if not file_paths:
        print(f"Info: No layer files found in '{load_path}'.")
        return []


    file_paths.sort(key=lambda f: int(os.path.basename(f).split('_')[1].split('.')[0]))

    print(f"Loading {len(file_paths)} layer representations from '{load_path}'...")

    layers_data = [torch.load(p, map_location=device) for p in file_paths]


    stacked_by_layer = torch.stack(layers_data, dim=0)

    permuted = stacked_by_layer.permute(1, 0, 2, 3)

    reconstructed_list = [example_tensor for example_tensor in permuted]

    return reconstructed_list
