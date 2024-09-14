import os
from multiprocessing import Lock

import monai
import monai.transforms as mt
import numpy as np
import torch
from autopet3.datacentric.transforms import get_transforms
from autopet3.datacentric.utils import get_file_dict_nn, read_split, get_file_dict_nn_synthesized, \
    get_file_dict_nn_synthesized_all
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class ResampleDataset(Dataset):
    def __init__(
            self,
            data_dir: str,
            save_path: str,
            transform: mt.Compose,
            samples_per_file: int = 15,
            seed: int = 42,
            resume: bool = False,
    ) -> None:
        """Initialize the class with the provided parameters.
        Args:
            data_dir (str): Path to the directory containing the data.
            save_path (str): Path to save the processed data.
            transform (monai composable): Transformation function to apply to the data.
            samples_per_file (int): Number of samples per file.
            seed (int): Seed for reproducibility.
            resume (bool): Flag indicating whether to resume preprocessing.

        """
        monai.utils.set_determinism(seed=seed)
        np.random.seed(seed)

        split_data = read_split(os.path.join(data_dir, "splits_final_all.json"), 0)
        train_val_data = split_data["val"]  # + split_data["val"]

        self.files = get_file_dict_nn_synthesized_all(data_dir, train_val_data, suffix=".nii.gz")
        self.transform = transform
        self.destination = save_path
        self.root = data_dir
        self.samples_per_file = samples_per_file

        print(f"self.files length is {len(self.files)}")
        print(f"train_val_data length is {len(train_val_data)}")

        if resume:
            valid_files = self.resume_preprocessing()
            train_val_data = list(set(train_val_data) - set(valid_files))

            print(f"valid files length is {len(valid_files)}")
            print(f"train_val_data length after is {len(train_val_data)}")

        self.files = get_file_dict_nn_synthesized_all(data_dir, train_val_data, suffix=".nii.gz")
        # print("self files are")
        # print(self.files)
        print(f"self.files length after is {len(self.files)}")
        self.lock = Lock()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        for i in range(self.samples_per_file):
            image, label = self.transform(file_path)
            label_name = str(file_path["label"]).replace(".nii.gz", "").split("\\")[
                -1]  # / vs \ makes the whole difference
            output_path = os.path.join(self.destination, f"{file_path['element']}_synthesized_{i:03d}.npz")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with self.lock:
                np.savez_compressed(output_path, input=image.numpy(), label=label.numpy())
        return image, label

    def resume_preprocessing(self):
        unique_files, counts = np.unique(
            ["_".join(i.split("_")[:-2]) for i in os.listdir(self.destination)], return_counts=True
        )
        valid_files = list(unique_files[counts == self.samples_per_file])
        invalid_files = list(unique_files[counts != self.samples_per_file])
        # just for testin
        # input()
        # valid_files_to_return = []
        for j, i in tqdm(enumerate(valid_files), desc=f"Resuming preprocessing. Validate {len(valid_files)} files"):

            try:
                data = np.load(test_file)

                image = torch.from_numpy(data["input"])
                label = torch.from_numpy(data["label"])
                # valid_files_to_return.append(test_file)
            except Exception:
                valid_files.pop(j)

        print(f"Found {len(valid_files)} valid files!")
        valid_files = ['_'.join(valid_file.split('_')[:-1]) for valid_file in valid_files]
        # print(valid_files)
        invalid_files = ['_'.join(invalid_file.split('_')[:-1]) for invalid_file in invalid_files]
        valid_files = [valid_file for valid_file in valid_files if valid_file not in invalid_files]
        # valid_files.remove("fdg_d40a16781a_09-13-2003-NA-PET-CT Ganzkoerper  primaer mit KM-42002")
        # valid_files.remove("fdg_5060603ba4_09-19-2003-NA-PET-CT Ganzkoerper nativ-36416")
        return valid_files


def test_integrity(dir_path):
    for filename in tqdm(os.listdir(dir_path)):
        file_path = os.path.join(dir_path, filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File '{filename}' does not exist in directory.")

        # Loa
        try:
            data = np.load(file_path)

            image = torch.from_numpy(data["input"])
            label = torch.from_numpy(data["label"])
        except Exception as e:
            print("Error occurred:", e)
            print(filename)


if __name__ == "__main__":
    root = "DiffTumor_data/Autopet/"
    dest = "DiffTumor_data/Autopet/preprocessed_all_synthesized_30/val_but_train"
    worker = 24
    samples_per_file = 30
    seed = 42

    transform = get_transforms("train", target_shape=(128, 160, 112), resample=True)
    ds = ResampleDataset(root, dest, transform, samples_per_file=samples_per_file, seed=seed, resume=True)

    dataloader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=worker)
    for _ in tqdm(dataloader, total=len(dataloader)):
        pass
    test_integrity(dest)