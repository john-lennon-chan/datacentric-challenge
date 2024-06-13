import numpy as np
import nibabel as nib
import torch
from matplotlib import pyplot as plt
from random import shuffle, seed
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor

from autopet3.fixed.utils import plot_ct_pet_label, plot_results
from autopet3.fixed.evaluation import AutoPETMetricAggregator

import os
"""
def detect_false_positives(prediction, ground_truth):
    connected_components = cc3d.connected_components(prediction.astype(int), connectivity=18)

    for idx in range(1, connected_components.max() + 1):
        component_mask = np.isin(connected_components, idx)
        if (component_mask * ground_truth).sum() == 0:
            pass
"""

def plot_graph(
    ct,
    pet,
    prediction,
    label,
    organ_mask,
    visualization,
    axis: int = 1,
) -> None:
    """Plot the sum of the label, CT, and PET images along the second dimension.
    Args:
        ct (Union[np.ndarray, torch.Tensor]): The CT image.
        pet (Union[np.ndarray, torch.Tensor]): The PET image.
        label (Union[np.ndarray, torch.Tensor]): The label image.
        axis (int): The axis along which the sum will be computed.

    """
    ct = ct.detach().cpu().numpy() if isinstance(ct, torch.Tensor) else ct
    pet = pet.detach().cpu().numpy() if isinstance(pet, torch.Tensor) else pet
    pred_array = prediction.detach().cpu().numpy() if isinstance(prediction, torch.Tensor) else prediction
    label_array = label.detach().cpu().numpy() if isinstance(label, torch.Tensor) else label
    organ_mask_array = organ_mask.detach().cpu().numpy() if isinstance(organ_mask, torch.Tensor) else organ_mask

    data = {"ct": ct, "pet": pet}
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))

    # Plot the CT image
    for i in range(2):
        for j in range(3):
            im_ct = axes[i, j].imshow(np.rot90(data["ct"].squeeze().sum(axis)), cmap="gray")
    axes[0, 0].set_title("Sum of PET/CT Image")

    # Overlay the PET image on top of the CT image
    im_pet = axes[0, 0].imshow(np.rot90(np.amax(data["pet"].squeeze(), axis)), cmap="jet",
                            alpha=0.5)  # Change alpha to suit your needs

    # Add a colorbar for each image
    #plt.colorbar(im_pet, ax=axes[0, 0], fraction=0.046, pad=0.04)

    def plot_organ_mask(organ_mask_array, axis):
        colour_list = ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
                      'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
                      'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn'] * 100

        def process_organ(organ_number, colour, organ_mask_array, axis, axes):
            number_mask = np.isin(organ_mask_array.squeeze(), organ_number)
            organ_copy = deepcopy(organ_mask_array.squeeze())
            organ_copy[number_mask] = 1
            organ_copy[~number_mask] = 0
            axes[1, 0].imshow(np.rot90(np.sum(organ_copy, axis)), cmap=colour)

        with ThreadPoolExecutor() as executor:
            for organ_number, colour in zip(range(1, int(np.max(organ_mask_array.squeeze()))), colour_list):
                executor.submit(process_organ, organ_number, colour, organ_mask_array, axis, axes)
        axes[1, 0].set_title("Organ Mask")

    plot_organ_mask(organ_mask_array, axis)
    #axes[1, 0].imshow(np.rot90(np.sum(organ_mask_array.squeeze(), axis)), cmap='tab20c')
    #axes[1, 0].set_title("Organ Mask")

    # Plot the organ mask
    axes[1, 0].imshow(np.rot90(np.sum(organ_mask_array.squeeze(), axis)), cmap='tab20c')
    axes[1, 0].set_title("Organ Mask")

    # Plot the predicted Label
    axes[0, 1].imshow(np.rot90(np.sum(pred_array.squeeze(), axis)), cmap="jet", alpha=0.5)
    axes[0, 1].set_title("Predicted Label")

    # Plot the GT Label
    axes[0, 2].imshow(np.rot90(np.sum(label_array.squeeze(), axis)), cmap="jet", alpha=0.5)
    axes[0, 2].set_title("GT Label")

    # Plot FP error
    axes[1, 1].imshow(np.rot90((np.maximum(0, pred_array.squeeze() - label_array.squeeze())).mean(1)), cmap="Reds", alpha=0.5)
    axes[1, 1].set_title("False Positive error")

    # Plot FN error
    axes[1, 2].imshow(np.rot90((np.maximum(0, label_array.squeeze() - pred_array.squeeze())).mean(1)), cmap="Reds",
                      alpha=0.5)
    axes[1, 2].set_title("False Negative error")
    plt.savefig(visualization, dpi=300, bbox_inches='tight')

def plot(train_val, data_dir, results_dir, test_file):
    ct_path = os.path.join(data_dir, "imagesTr", test_file + "_0000.nii.gz")
    pet_path = os.path.join(data_dir, "imagesTr", test_file + "_0001.nii.gz")
    prediction_path = os.path.join(results_dir, train_val, "predicted_images", test_file + "_0000_pred.nii.gz")
    label_path = os.path.join(data_dir, "labelsTr", test_file + ".nii.gz")
    organ_mask_path = os.path.join(results_dir, "SegResNet", test_file + "_0000", test_file + "_0000_trans.nii.gz")
    visualization_path = os.path.join(results_dir, train_val, "visualizations_organ_mask", test_file + "_visualization.png")

    ct = nib.load(ct_path).get_fdata()
    pet = nib.load(pet_path).get_fdata()
    prediction = nib.load(prediction_path).get_fdata()
    label = nib.load(label_path).get_fdata()
    organ_mask = nib.load(organ_mask_path).get_fdata()
    #plot_ct_pet_label(ct, pet, label)
    #plot_results(prediction, label)
    plot_graph(ct, pet, prediction, label, organ_mask, visualization_path)




if __name__ == "__main__":
    import json
    from tqdm import tqdm

    # iterate through the filenames in read_split
    from Codes.autopet3.datacentric.utils import read_split
    data_dir = r"D:\testing_AI_environment\Autopet"
    results_dir = r"D:\testing_AI_environment\results"
    split_file = os.path.join(data_dir, "splits_final.json")
    split = read_split(split_file, 0)
    processed_files = [] # json.load(open(os.path.join(results_dir, "val", "visualizations_organ_mask", "processed_list.json"), "r"))


    for train_val in ["train", "val"]:
        iterable = list(filter(lambda f: f not in processed_files, split[train_val]))
        shuffle(iterable)
        for file in tqdm(iterable, desc=f"Processing {train_val} data"):
            plot(train_val, data_dir, results_dir, file)
            processed_files.append(file)
            json.dump(processed_files, open(os.path.join(results_dir, train_val, "visualizations_organ_mask", "processed_list.json"), "w"))
            break
    #test_file = "..\\test_original\\psma_95b833d46f153cd2_2017-11-18"
    #plot(test_file)

