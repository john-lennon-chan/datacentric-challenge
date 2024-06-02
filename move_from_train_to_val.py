import json
import os
import shutil

def read_split(splits_file: str, fold: int = 0) -> dict:
    """Read a specific fold from a JSON file containing splits.
    Args:
        splits_file (str): The path to the JSON file containing the splits.
        fold (int): The fold number to read from the splits file. Defaults to 0.
    Returns:
        dict: The dictionary representing the split for the specified fold.

    """
    with open(splits_file) as json_file:
        splits_dict = json.load(json_file)[fold]
    return splits_dict

splits_dict = read_split("../Autopet/splits_final.json", 0)


train_dir = "../preprocessed_1_random_sample/train"
val_dir = "../preprocessed_1_random_sample/val"
list_train_dir_copy = os.listdir(train_dir)

# Iterate over all files in the train directory
for filename in list_train_dir_copy:
    # Check if the file is in splits_dict["val"]
    if filename.split('_000')[0] in splits_dict["val"]:
        # Construct full file paths
        source = os.path.join(train_dir, filename)
        destination = os.path.join(val_dir, filename)
        # Move the file to the val directory
        shutil.move(source, destination)