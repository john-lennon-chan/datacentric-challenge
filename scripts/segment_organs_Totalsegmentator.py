import nibabel as nib
from tqdm import tqdm
from totalsegmentator.python_api import totalsegmentator
from Codes.autopet3.datacentric.utils import get_file_dict_nn, read_split
import os, json
import torch

if __name__ == "__main__":
    data_dir = r"D:\testing_AI_environment\Autopet\imagesTr"
    output_dir = r"D:\testing_AI_environment\results"
    splits_file = r"D:\testing_AI_environment\Autopet\splits_final.json"
    split = read_split(splits_file, 0)
    transform = None
    processed_list = json.load(open(os.path.join(output_dir, "train", "organ_segmentations", "processed_list.json"), "r")) + json.load(open(os.path.join(output_dir, "val", "organ_segmentations", "processed_list.json"), "r"))

    for train_or_val in ["train", "val"]:
        for file in tqdm([file for file in split[train_or_val] if file not in processed_list], desc=f"Processing {train_or_val} data"):
            input_path = os.path.join(data_dir, file + "_0000.nii.gz")
            output_path = os.path.join(output_dir, train_or_val, "organ_segmentations", file)
            totalsegmentator(input_path, output_path)
            processed_list.append(file)
            json.dump(processed_list, open(os.path.join(output_dir, train_or_val, "organ_segmentations", "processed_list.json"), "w"))
            #torch.cuda.empty_cache()

            continue

            #os.system(f"set nnUNet_raw={datadir}")
            #os.system("set nnUNet_preprocessed=C:/Users/fabian/nnUNet_preprocessed")
            #os.system("set nnUNet_results=C:/Users/fabian/fabian/nnUNet_results")