import os
from multiprocessing import Lock

import sys
sys.path.append(r'/data/lychan/24_Summer_Research/datacentric-challenge/scripts/DiffTumor/STEP3_SegmentationModel')

import monai
import monai.transforms as mt
import numpy as np
import torch
#from autopet3.datacentric.transforms import get_transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from DiffTumor.STEP3_SegmentationModel.main import _get_synthesize_transform
from DiffTumor.STEP3_SegmentationModel.TumorGeneration.utils_AutoPET import synthesize_early_tumor, synthesize_medium_tumor, synthesize_large_tumor, synt_model_prepare


class AugmentedDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        save_path: str,
        files,
        #transform: mt.Compose,
        synthesize_transform: mt.Compose = None,
        samples_per_file: int = 15,
        seed: int = 42,
        resume: bool = False,
        args: object = None
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

        #split_data = read_split(os.path.join(data_dir, "splits_final.json"), 0)
        #train_val_data = split_data["train"] + split_data["val"]

        self.files = files
        #self.transform = transform
        self.synthesize_transform = synthesize_transform
        self.destination = save_path
        self.root = data_dir
        self.samples_per_file = samples_per_file
        self.args = args

        print(f"self.files length is {len(self.files)}")
        #print(f"train_val_data length is {len(train_val_data)}")

        #if resume:
        #    valid_files = self.resume_preprocessing()
        #    train_val_data = list(set(train_val_data) - set(valid_files))

        #    print(f"valid files length is {len(valid_files)}")
        #    print(f"train_val_data length after is {len(train_val_data)}")

        self.files = files
        print(f"self.files length after is {len(self.files)}")
        self.lock = Lock()

    def __len__(self):
        return len(self.files)

    def synthesize_tumour_preparation(args):
        """This function is mostly copied from what is inside train_epoch of monai_trainer.py in the DiffTumor folder."""

        if args.organ_type == 'liver':
            sample_thresh = 0.5
        elif args.organ_type == 'pancreas':
            sample_thresh = 0.5
        elif args.organ_type == 'kidney':
            sample_thresh = 0.5
        # model prepare
        vqgan, early_sampler, noearly_sampler = synt_model_prepare(device=torch.device("cuda", args.rank),
                                                                   fold=args.fold, organ=args.organ_model)

        return vqgan, early_sampler, noearly_sampler, sample_thresh

    def synthesize_tumour(batch_data):
        data, target = data.cuda(args.rank), target.cuda(args.rank)


    def __getitem__(self, idx):
        args = self.args

        file_dict = self.files[idx]
        # the dictionary of the paths of the data

        transformed_data = self.synthesize_transform(file_dict)

        vqgan, early_sampler, noearly_sampler, sample_thresh = self.synthesize_tumour_preparation(args)
        for i in range(self.samples_per_file):
            #image, label = self.transform(file_path)
            #label_name = str(file_path["label"]).replace(".nii.gz", "").split("\\")[-1] # / vs \ makes the whole difference
            output_path = os.path.join(self.destination, f"{label_name}_{i:03d}.npz")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with self.lock:
                np.savez_compressed(output_path, input=image.numpy(), label=label.numpy())
        return image, label

    def resume_preprocessing(self):
        unique_files, counts = np.unique(
            ["_".join(i.split("_")[:-1]) for i in os.listdir(self.destination)], return_counts=True
        )
        valid_files = list(unique_files[counts == self.samples_per_file])
        #valid_files_to_return = []
        for j, i in tqdm(enumerate(valid_files), desc=f"Resuming preprocessing. Validate {len(valid_files)} files"):
            test_file = os.path.join(self.destination, f"{i}_{self.samples_per_file - 1:03d}.npz")
            # Load and process data
            data = np.load(test_file)
            try:
                image = torch.from_numpy(data["input"])
                label = torch.from_numpy(data["label"])
                #valid_files_to_return.append(test_file)
            except Exception:
                valid_files.pop(j)

        print(f"Found {len(valid_files)} valid files!")
        return valid_files


def test_integrity(dir_path):
    for filename in tqdm(os.listdir(dir_path)):
        file_path = os.path.join(dir_path, filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File '{filename}' does not exist in directory.")

        # Load data
        data = np.load(file_path)
        try:
            image = torch.from_numpy(data["input"])
            label = torch.from_numpy(data["label"])
        except Exception as e:
            print("Error occurred:", e)
            print(filename)

def creating_data_dicts(data_root, txt):
    cts = []
    pets = []
    lbls = []
    names = []
    for line in open(txt):
        names.append(line[240:].strip().split('.')[0])
        cts.append(data_root + line[:120].strip())
        pets.append(data_root + line[120:240].strip())
        lbls.append(data_root + line[240:].strip())
    data_dicts = [{'ct': ct, 'pet': pet, 'label': label, 'name': name}
                  for ct, pet, label, name in zip(cts, pets, lbls, names)]
    return data_dicts

if __name__ == "__main__":
    root = "DiffTumor_data/Autopet/"
    dest = "DiffTumor_data/imagesTr_Step_3_preprocessed/"
    worker = 20
    samples_per_file = 15
    seed = 42

    # this is for the arguments of the procedure of transforms
    class Args:
        def __init__(self):
            self.organ_type = "liver"
            self.organ_number = 1
            self.rank = 0
            self.fold = 0
            self.organ_model = "liver"
            self.fg_thresh = 0.5

            self.ct_a_min = -832.062744140625 # -175
            self.ct_a_max = 1127.758544921875 # 250
            self.pet_a_min = 1.0433332920074463
            self.pet_a_max = 51.211158752441406
            self.b_min = -1.0
            self.b_max = 1.0




    args = Args()

    files = creating_data_dicts(root, "DiffTumor_data/Autopet/Step_3_synthesize_datalist.txt")
    #transform = get_transforms("train", target_shape=(128, 160, 112), resample=True)
    organ_type_dicts = {5: "liver",
                        2: "kidney",
                        3: "kidney",
                        1: "spleen"}

    _, synthesize_transform = _get_synthesize_transform(args)
    ds = AugmentedDataset(root, dest, files, synthesize_transform=synthesize_transform, samples_per_file=samples_per_file, seed=seed, resume=True, args=args)

    dataloader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=worker)
    for _ in tqdm(dataloader, total=len(dataloader)):
        pass
    test_integrity(dest)
