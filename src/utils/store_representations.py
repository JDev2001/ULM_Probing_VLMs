import os
from typing import List

import torch
from tqdm import tqdm
import glob


from src.utils.experiment_utils import get_repr_for_layer

import os
from typing import List, Any
import torch


def save_labels(labels: List[Any], base_path: str, experiment_name: str, split_name: str):
    """
    Saves a list of labels to a file using torch.save.

    Args:
        labels (List[Any]): A list containing the labels for a dataset split.
        base_path (str): The base directory for storing experiments.
        experiment_name (str): The name of the specific experiment.
        split_name (str): The name of the data split (e.g., 'train', 'test').
    """
    if not labels:
        print(f"Warning: No labels to save for '{split_name}'. Skipping save.")
        return

    # Construct the full path and create directories if they don't exist
    save_path = os.path.join(base_path, experiment_name, split_name)
    os.makedirs(save_path, exist_ok=True)
    print(f"Storing labels in '{save_path}'...")

    # Define the file path for the labels
    file_path = os.path.join(save_path, "labels.pt")

    # Save the list of labels
    torch.save(labels, file_path)
    print(f"Successfully saved {len(labels)} labels to '{file_path}'.")


def load_labels(base_path: str, experiment_name: str, split_name: str, device: str = "cpu") -> List[Any]:
    """
    Loads a list of labels from a file.

    Args:
        base_path (str): The base directory where experiments are stored.
        experiment_name (str): The name of the specific experiment.
        split_name (str): The name of the data split (e.g., 'train', 'test').
        device (str): The device to map the loaded data to (e.g., 'cpu', 'cuda').
                      This is included for consistency but labels are typically device-agnostic.

    Returns:
        List[Any]: The loaded list of labels. Returns an empty list if the file is not found.
    """
    file_path = os.path.join(base_path, experiment_name, split_name, "labels.pt")

    # Check if the labels file exists
    if not os.path.isfile(file_path):
        print(f"Info: Labels file '{file_path}' does not exist. Returning empty list.")
        return []

    print(f"Loading labels from '{file_path}'...")

    # Load the labels file
    loaded_labels = torch.load(file_path, map_location=device)

    return loaded_labels


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
    """
    Loads layer representations from disk, reconstructs them, and returns them as a list of tensors.
    """
    load_path = os.path.join(base_path, experiment_name, split_name)

    if not os.path.isdir(load_path):
        print(f"Info: Path '{load_path}' does not exist. Returning empty list.")
        return []

    # Find all layer files and sort them numerically
    file_paths = glob.glob(os.path.join(load_path, "layer_*.pt"))
    if not file_paths:
        print(f"Info: No layer files found in '{load_path}'. Returning empty list.")
        return []

    file_paths.sort(key=lambda f: int(os.path.basename(f).split('_')[1].split('.')[0]))

    print(f"Loading {len(file_paths)} layer representations from '{load_path}'...")

    # Load all tensors from their respective files
    layers_data = [torch.load(p, map_location=device) for p in file_paths]

    # Stack the list of tensors along a new dimension (dim=0)
    # This creates a tensor of shape [num_layers, num_examples, ...]
    stacked_by_layer = torch.stack(layers_data, dim=0)


    # Check the number of dimensions and apply the correct permutation
    if stacked_by_layer.dim() == 4:
        # For 4D tensors: [layers, examples, seq_len, hidden_dim]
        # We permute to: [examples, layers, seq_len, hidden_dim]
        permuted = stacked_by_layer.permute(1, 0, 2, 3)
    elif stacked_by_layer.dim() == 3:
        # For 3D tensors (the case causing the error): [layers, examples, hidden_dim]
        # We permute to: [examples, layers, hidden_dim]
        permuted = stacked_by_layer.permute(1, 0, 2)
    else:
        # Handle unexpected tensor shapes
        raise ValueError(f"Unexpected tensor dimension after stacking: {stacked_by_layer.dim()}. Expected 3 or 4.")

    # Reconstruct the original list format, where each item is one example's
    # complete representation across all layers.
    reconstructed_list = [example_tensor for example_tensor in permuted]

    return reconstructed_list
