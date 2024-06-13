import json
import pandas as pd
import os

def load_json_files_and_compile_as_csv(metrics_folder, analysis_folder):
    # Load all json files in the metrics folder
    json_files = [f for f in os.listdir(metrics_folder) if f.endswith(".json")]
    data = {}
    for file in json_files:
        with open(os.path.join(metrics_folder, file), "r") as f:
            data[file.split("_metrics")[0]] = json.load(f)

    dataframes = {metric: pd.DataFrame.from_dict(
        {patient: [values[metric] for values in organs.values()] for patient, organs in
         data.items()}, orient='index', columns=list(list(data.values())[0].keys())) for metric in ["false_positives", "false_negatives", "dice_score"]}

    # Save each DataFrame as a CSV file
    for metric, df in dataframes.items():
        try:
            os.remove(os.path.join(analysis_folder, f'{metric}.csv'))
        except Exception as e:
            pass
        print(df)
        df.to_csv(os.path.join(analysis_folder, f'{metric}.csv'), )

def do_analysis_on_csv_files(analysis_folder):
    # load the dataframes
    metrics = ["false_positives", "false_negatives", "dice_score"]
    dataframes = {metric: pd.read_csv(os.path.join(analysis_folder, f'{metric}.csv'), index_col=0) for metric in metrics}


    print(dataframes["false_positives"])

    # for each dataframe, for each column, calculate its mean, median and S.D.
    for metric, df in dataframes.items():
        outliers = [] #["fdg_adc49adb3d_09-22-2002-NA-PET-CT Ganzkoerper  primaer mit KM-59119", "psma_18eba3b35ee1ddac_2019-06-08"]
        # add row of means, medians and S.D.
        # replace all nan with 0
        df_new = df.fillna(0).drop(outliers)
        df.loc["mean"] = df_new.mean(skipna=False)
        df.loc["median"] = df_new.median(skipna=False)
        df.loc["std"] = df_new.std(skipna=False)
        # sort the means the medians and the stds
        df = df.sort_values(by="mean", axis=1)
        try:
            os.remove(os.path.join(analysis_folder, f'{metric}_organ_analysis.csv'))
        except Exception as e:
            print(e)
            pass
        df.to_csv(os.path.join(analysis_folder, f'{metric}_organ_analysis.csv'))






if __name__ == "__main__":
    metrics_folder = r"D:\testing_AI_environment\results\metrics_per_organ"
    analysis_folder = r"D:\testing_AI_environment\results\metrics_analysis"
    #load_json_files_and_compile_as_csv(metrics_folder, analysis_folder)
    do_analysis_on_csv_files(analysis_folder)