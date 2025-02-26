import os
import glob
import pandas as pd

# Define the directory where the CSV files are stored
directory = "csv/"

# Use glob to find all files containing "esg_scores_with_confidence"
file_pattern = os.path.join(directory, "*esg_scores_with_confidence*.csv")
csv_files = glob.glob(file_pattern)

# Check if any files were found
if not csv_files:
    raise ValueError(f"No files found matching the pattern '{file_pattern}'")

# Initialize an empty list to store DataFrames (no longer strictly needed, but good practice)
dfs = []

# Loop through each CSV file and read it into a pandas DataFrame
for file in csv_files:
    df = pd.read_csv(file)
    dfs.append(df)

# Concatenate all DataFrames into a single DataFrame
combined_df = pd.concat(dfs, ignore_index=True)

# --- Calculate Average Confidence and Binary Values Per Group (Efficiently) ---

confidence_cols = [f"conf_q{i}" for i in range(1, 118)]

# Group by 'company' and 'year' and calculate the mean of confidence columns
grouped_df = (
    combined_df.groupby(["company", "year"])[confidence_cols].mean().reset_index()
)

# Create binary confidence columns using a single operation (MUCH faster)
binary_confidence_df = (grouped_df[confidence_cols] > 0.5).astype(int)
binary_confidence_df.columns = [f"binary_{col}" for col in confidence_cols]

# Calculate the aggregated ESG score (efficiently using the binary_confidence_df)
grouped_df["aggregated_esg_score"] = binary_confidence_df.sum(axis=1)

# Concatenate all parts together:  original grouped data + binary data
output_df = pd.concat([grouped_df, binary_confidence_df], axis=1)

# Select columns for final output (no longer strictly needed, but makes output explicit)
output_columns = (
    ["company", "year", "aggregated_esg_score"]
    + confidence_cols
    + list(binary_confidence_df.columns)
)
output_df = output_df[output_columns]

# --- Create and save the new DataFrame ---
output_file = os.path.join(directory, "aggregated_esg_scores_with_confidence.csv")
output_df.to_csv(output_file, index=False)
print(f"Aggregated ESG scores with confidence saved to '{output_file}'")
