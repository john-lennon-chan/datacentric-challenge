from autopet3.fixed.evaluation import AutoPETMetricAggregator
import numpy as np
from copy import deepcopy
from autopet3.datacentric.utils import read_split
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import cc3d

class AutoPETParallelMetricAggregator(AutoPETMetricAggregator):
    @staticmethod
    def count_false_positives(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
        """Count the number of false positive pixel, which do not overlap with the ground truth, based on the prediction
        and ground truth arrays.
        Returns zero if the prediction array is empty.
        Args:
            prediction (np.ndarray): The predicted array.
            ground_truth (np.ndarray): The ground truth array.
        returns:
            float: The number of false positive pixel which do not overlap with the ground truth.

        """
        if prediction.sum() == 0:
            return 0

        if ground_truth.sum() == 0:
            # a little bit faster than calculating connected components
            return prediction.sum()

        connected_components = cc3d.connected_components(prediction.astype(int), connectivity=18)
        """
        def count_component_false_positives(idx):
            component_mask = np.isin(connected_components, idx)
            if (component_mask * ground_truth).sum() == 0:
                return component_mask.sum()
            else:
                return 0

        false_positives = 0
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(count_component_false_positives, idx) for idx in
                       range(1, connected_components.max() + 1)}
            for future in as_completed(futures):
                false_positives += future.result()
        """
        false_positives = 0

        for idx in range(1, connected_components.max() + 1):
            component_mask = np.isin(connected_components, idx)
            if (component_mask * ground_truth).sum() == 0:
                false_positives += component_mask.sum()


        return false_positives

class AutoPETPerOrganMetricAggregator:
    def __init__(self, organ_list):
        self.organ_list = organ_list
        self.metric_aggregators = {organ: AutoPETParallelMetricAggregator() for organ in organ_list}

    from concurrent.futures import ThreadPoolExecutor


    def update(self, prediction, label, organ_segmentation):
        def only_organ_part(prediction, label, organ_segmentation, organ_number):
            mask = np.isin(organ_segmentation, organ_number)
            prediction_copy = deepcopy(prediction)
            label_copy = deepcopy(label)
            prediction_copy[~mask] = 0
            label_copy[~mask] = 0
            return prediction_copy, label_copy

        def update_aggregator(organ_number, organ_aggregator):
            return organ_aggregator.update(*only_organ_part(prediction, label, organ_segmentation, organ_number))

        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_organ = {executor.submit(update_aggregator, organ_number, organ_aggregator): organ for
                               organ_number, (organ, organ_aggregator) in enumerate(self.metric_aggregators.items(), 1)}
            return {future_to_organ[future]: future.result() for future in
                    as_completed(future_to_organ)}
    """
    def update(self, prediction, label, organ_segmentation):
        def only_organ_part(prediction, label, organ_segmentation, organ_number):
            mask = np.isin(organ_segmentation, organ_number)
            prediction_copy = deepcopy(prediction)
            label_copy = deepcopy(label)
            prediction_copy[~mask] = 0
            label_copy[~mask] = 0
            return prediction_copy, label_copy
        return {organ: aggregator.update(*only_organ_part(prediction, label, organ_segmentation, organ_number)) for organ_number, (organ, aggregator) in enumerate(self.metric_aggregators.items(), 1)}
    """

    def print(self):
        for organ, aggregator in self.metric_aggregators.items():
            print(f"Organ: {organ}")
            print(f"Dice Score: {aggregator.dice_scores}")
            print(f"False Positives: {aggregator.false_positives}")
            print(f"False Negatives: {aggregator.false_negatives}")
            print("\n")

    def reset(self):
        for aggregator in self.metric_aggregators.values():
            aggregator.reset()

    def compute(self):
        return {organ: aggregator.compute() for organ, aggregator in self.metric_aggregators.items()}

if __name__ == "__main__":
    import nibabel as nib
    import json
    split = read_split(r"C:\Users\lenno\Documents\splits_final.json", 0)
    prediction_folder = r"C:\Users\lenno\Documents\predicted_images"
    ground_truth_folder = r"C:\Users\lenno\Documents\labelsTr"
    organ_segmentation_folder = r"C:\Users\lenno\Documents\SegResNet"
    metrics_folder = r"C:\Users\lenno\Documents\metrics_per_organ"

    organ_list_path = r"scripts\totalsegmentator_organ_list.json"
    organ_list = json.load(open(organ_list_path, "r"))

    aggregator = {"train": AutoPETPerOrganMetricAggregator(organ_list),
                  "val": AutoPETPerOrganMetricAggregator(organ_list)}

    processed_files = json.load(open(os.path.join(metrics_folder, "processed_list.json"), "r"))
    for train_or_val in ["val"]:
        for file in tqdm([file for file in split[train_or_val] if file not in processed_files], desc=f"Processing {train_or_val} data"):
            prediction_path = os.path.join(prediction_folder, file + "_0000_pred.nii.gz")
            ground_truth_path = os.path.join(ground_truth_folder, file + ".nii.gz")
            organ_segmentation_path = os.path.join(organ_segmentation_folder, file + "_0000", file + "_0000_trans.nii.gz")
            gt_array = nib.load(ground_truth_path).get_fdata() #np.zeros((1, 1, 10, 10, 10))
            pred_array = nib.load(prediction_path).get_fdata() #np.zeros((1, 1, 10, 10, 10))
            organ_segmentation_array = nib.load(organ_segmentation_path).get_fdata()


            def default(o):
                if isinstance(o, np.int32):
                    return int(o)
                raise TypeError

            metrics = aggregator[train_or_val].update(pred_array, gt_array, organ_segmentation_array)
            json.dump(metrics, open(os.path.join(metrics_folder, file + "_metrics.json"), "w"), default=default)
            processed_files.append(file)
            json.dump(processed_files, open(os.path.join(metrics_folder, "processed_list.json"), "w"))

        results = aggregator[train_or_val].compute()
        aggregator[train_or_val].reset()

        # check fn
        print(results)