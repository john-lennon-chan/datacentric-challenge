import numpy as np
import nibabel as nib
import torch
from matplotlib import pyplot as plt

from autopet3.fixed.utils import plot_ct_pet_label, plot_results
from autopet3.fixed.evaluation import AutoPETMetricAggregator

import os

def detect_false_positives(prediction, ground_truth):
    connected_components = cc3d.connected_components(prediction.astype(int), connectivity=18)

    for idx in range(1, connected_components.max() + 1):
        component_mask = np.isin(connected_components, idx)
        if (component_mask * ground_truth).sum() == 0:
            pass


def plot_graph(
    ct,
    pet,
    prediction,
    label,
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
    plt.show()

def plot(test_file):
    ct_path = os.path.join("../AutoPET/imagesTr", test_file + "_0000.nii.gz")
    pet_path = os.path.join("../AutoPET/imagesTr", test_file + "_0001.nii.gz")
    prediction_path = os.path.join("../results/val", test_file + "_0000_pred.nii.gz")
    label_path = os.path.join("../AutoPET/labelsTr", test_file + ".nii.gz")

    ct = nib.load(ct_path).get_fdata()
    pet = nib.load(pet_path).get_fdata()
    prediction = nib.load(prediction_path).get_fdata()
    label = nib.load(label_path).get_fdata()
    #plot_ct_pet_label(ct, pet, label)
    #plot_results(prediction, label)
    plot_graph(ct, pet, prediction, label)




if __name__ == "__main__":
    # iterate through the filenames of ../results/

    for file in os.listdir("..\\results\\val"):
        if file.endswith(".nii.gz"):
            plot(file[:-17])
    #test_file = "..\\test_original\\psma_95b833d46f153cd2_2017-11-18"
    #plot(test_file)