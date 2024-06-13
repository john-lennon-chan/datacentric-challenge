import nibabel as nib
#from tqdm import tqdm
#from Codes.autopet3.datacentric.utils import get_file_dict_nn, read_split
import os
import shutil
#import json
#import torch
from moosez import moose
from Codes.autopet3.datacentric.utils import read_split

def rename_files_in_directory(directory):
    for filename in os.listdir(directory):
        if not filename.startswith("CT_"):
            new_filename = "CT_" + filename
            source = os.path.join(directory, filename)
            destination = os.path.join(directory, new_filename)
            os.rename(source, destination)

#rename_files_in_directory(r"D:\testing_AI_environment\Autopet\imagesTr - Copy\S1")


def move_files_to_new_folder(source_dir, target_dir):
    # Check if the target directory exists. If not, create it.
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Iterate over all files in the source directory
    for filename in os.listdir(source_dir):
        # If a file ends with 0000.nii.gz, move it to the target directory
        if filename.endswith('_0000.nii.gz'):
            shutil.move(os.path.join(source_dir, filename), target_dir)

def decompress_nii_gz(input_dir, output_dir):


    # Check if the output directory exists. If not, create it.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate over all files in the input directory
    for filename in os.listdir(input_dir):
        # If a file ends with .nii.gz, decompress it and save it to the output directory
        if filename.endswith('.nii.gz'):
            img = nib.load(os.path.join(input_dir, filename))
            new_filename = filename[:-3]  # Remove .gz from the filename
            nib.save(img, os.path.join(output_dir, new_filename))

def refill_files(input_dir: str, output_dir: str, decompress_dir: str):
    input_list = list(filter(lambda fn: fn.endswith("_0000.nii.gz"), os.listdir(input_dir)))
    output_list = os.listdir(output_dir)
    for file in input_list:
        if file not in list(map(lambda fn: fn[3:], output_list)):
            shutil.copy2(os.path.join(input_dir, file), os.path.join(output_dir, "CT_" + file))
            img = nib.load(os.path.join(input_dir, file))
            new_filename = "CT_" + file[:-3]  # Remove .gz from the filename
            nib.save(img, os.path.join(decompress_dir, new_filename))

def get_the_nii_into_individual_folders(dir):
    for file in os.listdir(dir):
        os.makedirs(os.path.join(dir, file[3:-4]), exist_ok=True)
        os.rename(os.path.join(dir, file), os.path.join(dir, file[3:-4], file))

def get_the_folders_into_individual_nii(dir):
    for folder in os.listdir(dir):
        file = os.listdir(os.path.join(dir, folder))[0]
        os.rename(os.path.join(dir, folder, file), os.path.join(dir, file))

#get_the_folders_into_individual_nii(r"D:\testing_AI_environment\Autopet\imagesTr - Copy\S1_decompressed_python")

#get_the_nii_into_individual_folders(r"D:\testing_AI_environment\Autopet\imagesTr - Copy\S1_decompressed")



if __name__ == "__main__":
    try:
        model_name = 'clin_ct_organs'
        input_dir = r"D:\testing_AI_environment\Autopet\imagesTr - Copy\S1_decompressed_python"
        output_dir = r"D:\testing_AI_environment\results\Moose_segmentations"
        accelerator = 'cuda'
        moose(model_name, input_dir, output_dir, accelerator)
    except KeyboardInterrupt:
        pass
        #segmented_files = map(lambda fn: os.listdir(output_dir))
