import numpy as np
import nibabel as nib
import torch

from autopet3.fixed.utils import plot_ct_pet_label, plot_results

def plot(ct_path, pet_path, prediction_path, label_path):
    ct = nib.load(ct_path).get_fdata()
    pet = nib.load(pet_path).get_fdata()
    prediction = nib.load(prediction_path).get_fdata()
    label = nib.load(label_path).get_fdata()
    plot_ct_pet_label(ct, pet, label)
    plot_results(prediction, label)




if __name__ == "__main__":
    ct_path = "test\\psma_95b833d46f153cd2_2017-11-18_0000.nii.gz"
    pet_path = "test\\psma_95b833d46f153cd2_2017-11-18_0001.nii.gz"
    prediction_path = "test\\psma_95b833d46f153cd2_2017-11-18_0000_pred.nii.gz"
    label_path = "test\\psma_95b833d46f153cd2_2017-11-18.nii.gz"
    plot(ct_path, pet_path, prediction_path, label_path)