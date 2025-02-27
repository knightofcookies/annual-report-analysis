""" Combines ESG CSV files into a single CSV file. """
import os
import glob
import pandas as pd


def combine_esg_scores(directory, output_file):
    """
    Combines ESG scores from all CSV files in a directory into a single CSV,
    retaining company and year columns.

    Args:
        directory (str): The path to the directory containing the CSV files.
        output_file (str): The path to the output CSV file.
    """

    csv_files = glob.glob(os.path.join(directory, "*.csv"))

    csv_files = [
        file for file in csv_files if "esg_scores_with_confidence" in file.lower()
    ]

    if len(csv_files) != 5:
        print(
            f"Warning: Found {len(csv_files)} CSV files in the directory. Expected 5."
        )

    data = []
    company_year_data = {}

    for csv_index, csv_file in enumerate(csv_files):
        try:
            df = pd.read_csv(csv_file)

            if "company" not in df.columns or "year" not in df.columns:
                print(f"Error: 'company' or 'year' column not found in {csv_file}")
                return

            for _, row in df.iterrows():
                company = row["company"]
                year = row["year"]
                esg_score = row["esg_score"]

                key = (company, year)

                if key not in company_year_data:
                    company_year_data[key] = {
                        "company": company,
                        "year": year,
                        "esg_scores": [None] * len(csv_files),
                    }

                company_year_data[key]["esg_scores"][csv_index] = esg_score

        except FileNotFoundError:
            print(f"Error: File not found: {csv_file}")
            return
        except KeyError:
            print(f"Error: 'esg_score' column not found in {csv_file}")
            return
        except Exception as e:
            print(f"An unexpected error occurred while processing {csv_file}: {e}")
            return

    for key, value in company_year_data.items():
        row_data = {"company": value["company"], "year": value["year"]}
        for i, esg_score in enumerate(value["esg_scores"]):
            row_data[f"csv_{i+1}"] = esg_score
        data.append(row_data)

    output_df = pd.DataFrame(data)
    output_df["average_esg"] = output_df[
        [f"csv_{i+1}" for i in range(len(csv_files))]
    ].mean(axis=1)

    output_df.to_csv(output_file, index=False)


DIRECTORY = "./csv/2014_2018/"
OUTPUT_FILE = "./csv/2014_2018_combined_esg_scores.csv"

combine_esg_scores(DIRECTORY, OUTPUT_FILE)

print(f"Combined ESG scores saved to {OUTPUT_FILE}")
