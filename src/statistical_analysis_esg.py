import os
import glob
import numpy as np
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_and_save_statistics(folder_path):
    """
    Calculates statistics and saves results to CSVs and heatmaps, handling
    variable column lengths in input CSVs.
    """
    all_data = []
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    if not csv_files:
        raise ValueError(f"No CSV files found in {folder_path}")
    csv_files = [
        file for file in csv_files if "esg_scores_with_confidence" in file.lower()
    ]

    for file_path in csv_files:
        df = pd.read_csv(file_path)
        all_data.append(df)

    data_arrays = []
    for df in all_data:
        numeric_cols = [col for col in df.columns if col not in ["company", "year"]]
        numeric_df = df[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
        data_arrays.append(numeric_df.to_numpy())

    data_array = np.stack(data_arrays)

    mean_data = np.nanmean(data_array, axis=0)
    median_data = np.nanmedian(data_array, axis=0)
    mode_data = scipy.stats.mode(np.round(data_array, 4), axis=0, keepdims=True)[0][0]
    stddev_data = np.nanstd(data_array, axis=0)
    iqr_data = scipy.stats.iqr(data_array, axis=0, nan_policy="omit")

    statistics = {
        "mean.csv": mean_data,
        "median.csv": median_data,
        "mode.csv": mode_data,
        "stddev.csv": stddev_data,
        "iqr.csv": iqr_data,
    }

    first_df = pd.read_csv(csv_files[0], nrows=0)
    column_names = list(first_df.columns)

    num_q_cols = sum("q" in col and "conf" not in col for col in column_names)
    num_conf_q_cols = sum("conf_q" in col for col in column_names)

    if num_q_cols > mean_data.shape[1]:
        num_q_cols = mean_data.shape[1]
    if num_conf_q_cols > mean_data.shape[1] - num_q_cols:
        num_conf_q_cols = mean_data.shape[1] - num_q_cols

    updated_column_names = [f"q{i}" for i in range(1, num_q_cols + 1)]
    updated_column_names += [f"conf_q{i}" for i in range(1, num_conf_q_cols + 1)]
    if len(updated_column_names) < mean_data.shape[1]:
        updated_column_names.append("esg_score")

    for filename, data in statistics.items():
        try:
            company_year_df = all_data[0][["company", "year"]]
        except KeyError:
            print("Warning: 'company' or 'year' columns not found in CSV files.")
            continue  # skip this file.
        output_df = pd.concat(
            [company_year_df, pd.DataFrame(data, columns=updated_column_names)], axis=1
        )
        output_df.to_csv(os.path.join(folder_path, filename), index=False)

    for filename, data in statistics.items():
        heatmap_df = pd.DataFrame(data, columns=updated_column_names)
        if "esg_score" in heatmap_df.columns:
            heatmap_df = heatmap_df.drop("esg_score", axis=1)

        plt.figure(figsize=(12, 8))
        sns.heatmap(heatmap_df, annot=False, cmap="viridis")
        plt.title(f"Heatmap of {filename.split('.')[0].capitalize()}")
        plt.savefig(os.path.join(folder_path, f"{filename.split('.')[0]}_heatmap.png"))
        plt.close()

    print("Statistical analysis complete. CSVs and heatmaps saved.")


FOLDER_PATH = "./csv"  # REPLACE
try:
    calculate_and_save_statistics(FOLDER_PATH)
except ValueError as e:
    print(e)
