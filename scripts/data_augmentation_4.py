import os
from multiprocessing import Lock

import sys
sys.path.append(r'/data/lychan/24_Summer_Research/datacentric-challenge/scripts/DiffTumor/STEP3_SegmentationModel')

import monai
from monai import transforms
import monai.transforms as mt
import numpy as np
import torch
import random
#from autopet3.datacentric.transforms import get_transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from DiffTumor.STEP3_SegmentationModel.main import _get_synthesize_transform, _get_post_synthesize_transform
from DiffTumor.STEP3_SegmentationModel.TumorGeneration.utils_AutoPET import synthesize_early_tumor, synthesize_medium_tumor, synthesize_large_tumor, synt_model_prepare

from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

import threading
import time

class AugmentedDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        save_path: str,
        files,
        #transform: mt.Compose,
        synthesize_transform: mt.Compose = None,
        post_synthesize_transform: mt.Compose = None,
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
        self.post_synthesize_transform = post_synthesize_transform
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

    @staticmethod
    def sliding_window(synthesize_stage_tumour):
        def sliding_window_version(healthy_data, healthy_target, *args, **kwargs):
            def new_synthesize_stage_tumour(concatenated_image, *args, **kwargs):
                synt_data, synt_target = synthesize_stage_tumour(concatenated_image[:, 0:2, :, :, :], concatenated_image[:, 2:3, :, :, :], *args, **kwargs)
                return torch.cat((synt_data, synt_target), 1)
            inputs=torch.cat((healthy_data, healthy_target), 1)
            inferer = monai.inferers.SlidingWindowInferer((96, )*3, 10, 0.5)
            synt_concatenated = inferer(inputs, new_synthesize_stage_tumour,*args, **kwargs)
            return synt_concatenated[:, 0:2, :, :, :], synt_concatenated[:, 2:3, :, :, :]
        return sliding_window_version


    def synthesize_tumour_preparation(self, args):
        """This function is mostly copied from what is inside train_epoch of monai_trainer.py in the DiffTumor folder."""
        # model prepare
        vqgan, early_sampler, noearly_sampler = synt_model_prepare(device=torch.device("cuda", args.rank),
                                                                   vqgan_ckpt = args.vqgan_ckpt,
                                                                   diffusion_ckpt = args.diffusion_ckpt,
                                                                   fold=args.fold, organ=args.organ_model, args=args)

        return vqgan, early_sampler, noearly_sampler

    def synthesize_tumour(self, batch_data, vqgan, early_sampler, noearly_sampler, args):
        data, target, original_label = batch_data['image'], batch_data['label'], batch_data['original_label']
        data, target, original_label = data.cuda(args.rank), target.cuda(args.rank), original_label.cuda(args.rank)

        # synthesize tumour
        healthy_data = data#[None, ...]
        healthy_target = target#[None, ...]
        healthy_original_label = original_label#[None, ...]
        tumor_types = ['early', 'medium', 'large']
        tumor_probs = np.array([0.8, 0.1, 0.1]) #np.array([0.8, 0.1, 0.1])
        synthetic_tumor_type = np.random.choice(tumor_types, p=tumor_probs.ravel())
        if synthetic_tumor_type == 'early':
            synt_data, synt_target = synthesize_early_tumor(healthy_data, healthy_target, healthy_original_label, args.organ_type, vqgan,
                                                            early_sampler, args)
        elif synthetic_tumor_type == 'medium':

            synt_data_list, synt_target_list = [], []
            for i in range(healthy_data.shape[0]):
                synt_data, synt_target = synthesize_medium_tumor(healthy_data[i:i + 1],
                                                                 healthy_target[i:i + 1],
                                                                 healthy_original_label[i:i + 1], args.organ_type,
                                                                 vqgan,
                                                                 noearly_sampler, ddim_ts=args.ddim_ts, args=args)
                synt_data_list.append(synt_data[0])
                synt_target_list.append(synt_target[0])
            synt_data, synt_target = torch.stack(synt_data_list), torch.stack(synt_target_list)
            # synt_data, synt_target = synthesize_medium_tumor(healthy_data, healthy_target, healthy_original_label, args.organ_type, vqgan,
            #                                                 noearly_sampler, ddim_ts=args.ddim_ts, args=args)
        elif synthetic_tumor_type == 'large':

            synt_data_list, synt_target_list = [], []
            for i in range(healthy_data.shape[0]):
                synt_data, synt_target = synthesize_large_tumor(healthy_data[i:i + 1],
                                                                healthy_target[i:i + 1],
                                                                healthy_original_label[i:i + 1], args.organ_type,
                                                                vqgan,
                                                                noearly_sampler, ddim_ts=args.ddim_ts, args=args)
                synt_data_list.append(synt_data[0])
                synt_target_list.append(synt_target[0])
            synt_data, synt_target = torch.stack(synt_data_list), torch.stack(synt_target_list)
            # synt_data, synt_target = synthesize_large_tumor(healthy_data, healthy_target, healthy_original_label, args.organ_type, vqgan,
            #                                                noearly_sampler, ddim_ts=args.ddim_ts, args=args)

        data = synt_data#[0]
        target = synt_target#[0]
        return data, target

    class SynthesizeTransform(transforms.Transform):
        def __init__(self, synthesize_tumour, *args, **kwargs):
            self.synthesize_tumour = synthesize_tumour
            self.args = args
            self.kwargs = kwargs

        def __call__(self, data):

            if isinstance(data, dict):
                data["image"], data["label"] = self.synthesize_tumour(data, *self.args, **self.kwargs)
                return data
            elif isinstance(data, list):
                #return [self.__call__(d) for d in data]
                data_stack = {k: torch.stack(tuple([d[k] for d in data])) for k in ["image", "label", "original_label"]}
                data_stack["image"], data_stack["label"] = self.synthesize_tumour(data_stack, *self.args, **self.kwargs)
                for i, d in enumerate(data):
                    d["image"] = data_stack["image"][i]
                    d["label"] = data_stack["label"][i]
                return data

    #def synthesize_tumour_transform(self, data, i, *args, **kwargs):
    #    data["image"], data["label"] = self.synthesize_tumour(data, i, *args, **kwargs)
    #    return data


    def __getitem__(self, idx):
        random.seed(2)

        args = self.args

        file_dict = self.files[idx]
        # the dictionary of the paths of the data

        transformed_data = self.synthesize_transform(file_dict)
        #input("after data transform")

        vqgan, early_sampler, noearly_sampler = self.synthesize_tumour_preparation(args)
        #for i in range(self.samples_per_file):

        data = transformed_data
        data = self.SynthesizeTransform(self.synthesize_tumour, vqgan, early_sampler, noearly_sampler, args)(data)
        transformed_synthesized_data = self.post_synthesize_transform(data)
        return transformed_synthesized_data
        #except Exception as e:
        #    print(f"error is {e}")
        #    print(f"Error occurred in {file_dict['name']}")
        #    continue
        #output_path = os.path.join(self.destination, f"{label_name}_{i:03d}.npz")
        #os.makedirs(os.path.dirname(output_path), exist_ok=True)
        #with self.lock:
        #    np.savez_compressed(output_path, input=image.numpy(), label=label.numpy())

        #image, label = self.transform(file_path)
        #label_name = str(file_path["label"]).replace(".nii.gz", "").split("\\")[-1] # / vs \ makes the whole difference
        #output_path = os.path.join(self.destination, f"{label_name}_{i:03d}.npz")
        #os.makedirs(os.path.dirname(output_path), exist_ok=True)
        #with self.lock:
        #    np.savez_compressed(output_path, input=image.numpy(), label=label.numpy())


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
    original_lbls = []
    for line in open(txt):
        names.append(line[360:].strip().split('.')[0])
        cts.append(data_root + line[:120].strip())
        pets.append(data_root + line[120:240].strip())
        lbls.append(data_root + line[240:360].strip())
        original_lbls.append(data_root + line[360:].strip())
    data_dicts = [{'ct': ct, 'pet': pet, 'label': label, 'original_label': original_label, 'name': name}
                  for ct, pet, label, original_label, name in zip(cts, pets, lbls, original_lbls, names)]
    return data_dicts

#if __name__ == "__main__":
#def main():
root = "/data/lychan/24_Summer_Research/datacentric-challenge/scripts/DiffTumor_data/Autopet/"
dest = "/data/lychan/24_Summer_Research/datacentric-challenge/scripts/DiffTumor_data/Autopet/imagesTr_Step_3_synthesized/"
worker = 0

seed = 42

# this is for the arguments of the procedure of transforms
class Args:
    def __init__(self):
        self.organ_type = "liver"
        self.organ_number = 7
        self.rank = 0
        self.fold = 0
        self.organ_model = "liver"
        self.fg_thresh = 10
        self.spatial_size = (96, ) * 3

        self.ct_a_min = -832.062744140625 # -175
        self.ct_a_max = 1127.758544921875 # 250
        self.pet_a_min = 1.0433332920074463
        self.pet_a_max = 51.211158752441406
        self.b_min = 0.0
        self.b_max = 1.0
        self.samples_per_file = 3

        self.vqgan_ckpt = "/data/lychan/24_Summer_Research/datacentric-challenge/scripts/DiffTumor/STEP1_AutoencoderModel/checkpoints/vq_gan/synt/AutoPET_with_val/lightning_logs/version_1/checkpoints/latest_checkpoint-v1.ckpt"
        self.diffusion_ckpt = "/data/lychan/24_Summer_Research/datacentric-challenge/scripts/DiffTumor/STEP3_SegmentationModel/TumorGeneration/model_weight/"

        self.ddim_ts=50
        self.output_dir = "/data/lychan/24_Summer_Research/datacentric-challenge/scripts/DiffTumor_data/Autopet/imagesTr_Step_3_synthesized/"

        self.diffusion_img_size=(24, ) * 3
        self.sw_batch_size=1
        self.devices=[0]
        self.distributed=True



args = Args()

files = creating_data_dicts(root, "/data/lychan/24_Summer_Research/datacentric-challenge/scripts/DiffTumor_data/Autopet/Step_3_synthesize_datalist_all.txt")

def divide_list(lst, n):
    # Calculate the size of each part
    size = len(lst) // n

    parts = []

    for i in range(n):
        # Use list slicing to create the parts
        parts.append(lst[i * size:(i + 1) * size])

    return parts

if __name__ == "__main__":
    #torch.multiprocessing.set_start_method('spawn')

    files = divide_list(files, n=4)[3]
    for i in reversed(range(len(files))):
        file_entry = files[i]
        if file_entry["ct"].split('/')[-1].split(".nii")[0] + f"_{args.samples_per_file-1:02}_synthesized.nii.gz" in list(os.listdir(dest)):
            files.remove(file_entry)



    #transform = get_transforms("train", target_shape=(128, 160, 112), resample=True)




    synthesize_transform, _ = _get_synthesize_transform(args)
    post_synthesize_transform, _ = _get_post_synthesize_transform(args)
    ds = AugmentedDataset(root, dest, files, synthesize_transform=synthesize_transform, post_synthesize_transform=post_synthesize_transform, samples_per_file=args.samples_per_file, seed=seed, resume=True, args=args)

    dataloader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=worker)

    # Function to monitor the loop
    def monitor_loop(iterator, threshold):
        def watchdog():
            nonlocal stuck
            while not stop_event.is_set():
                time.sleep(threshold)
                if not stop_event.is_set():
                    stuck = True
                    print("Loop is stuck, restarting...")
                    break

        stop_event = threading.Event()
        stuck = False

        while True:
            for item in tqdm(iterator):
                stuck = False
                watchdog_thread = threading.Thread(target=watchdog)
                watchdog_thread.start()

                stop_event.set()
                watchdog_thread.join()

                if stuck:
                    break
            if not stuck:
                break

    iterator = dataloader
    while True:
        try:
            monitor_loop(iterator, threshold=600)
            break
        except ValueError:
            continue

    #while True:
    #    try:
    #        for _ in tqdm(dataloader, total=len(dataloader)):
    #            pass
    #        break
    #    except (ValueError, MemoryError, SystemExit, SystemError):
    #        continue

    test_integrity(dest)

