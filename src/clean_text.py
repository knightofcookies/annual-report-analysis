""" This script cleans the text of the parsed reports by removing newlines and tokenizing the text into sentences. """

import os
import nltk

nltk.download("punkt")
nltk.download("punkt_tab")

PARSED_REPORTS_DIR = "../parsed/2014-2018/"
CLEANED_REPORTS_DIR = "../cleaned/2014-2018/"


def main():
    """Main function to clean the text of the parsed reports."""
    if not os.path.exists(CLEANED_REPORTS_DIR):
        os.makedirs(CLEANED_REPORTS_DIR)
    for directory in os.listdir(PARSED_REPORTS_DIR):
        if directory == ".DS_Store":
            continue
        if not os.path.exists(CLEANED_REPORTS_DIR + directory):
            os.makedirs(CLEANED_REPORTS_DIR + directory)
        for file in os.listdir(PARSED_REPORTS_DIR + directory):
            with open(
                PARSED_REPORTS_DIR + directory + "/" + file, "r", encoding="utf-8"
            ) as f:
                text = f.read()
                text = text.replace("\n", " ").strip()
                tokens = nltk.sent_tokenize(text)
                with open(
                    CLEANED_REPORTS_DIR + directory + "/" + file, "w", encoding="utf-8"
                ) as f:
                    f.write("\n".join(tokens))


if __name__ == "__main__":
    main()
