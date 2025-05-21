"""
Download raw datasets and required NLTK corpora for sentiment analysis.

- Downloads datasets from Google Drive.
- Downloads necessary NLTK corpora (e.g., wordnet).
"""

import argparse
import os

import gdown
import nltk


def download_from_drive(file_id, output_path):
    """
    Download a file from Google Drive given its file ID.

    Args:
        file_id (str): The unique file ID on Google Drive.
        output_path (str): Path to save the downloaded file.
    """
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output_path, quiet=False)


def download_nltk_resources():
    """
    Ensure required NLTK corpora are downloaded.
    """
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet")


def main():
    """
    Main function to handle dataset and nltk corpora download.
    """
    parser = argparse.ArgumentParser(
        description="Download datasets and NLTK resources."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="datasets",
        help="Directory to save downloaded files.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Dataset 1
    file_id_1 = "1_SHjQJVxZdr_LW2aIHAiOSBPWWGWd7Bs"
    output_path_1 = os.path.join(
        args.output_dir, "a1_RestaurantReviews_HistoricDump.tsv"
    )
    download_from_drive(file_id_1, output_path_1)

    # Dataset 2
    file_id_2 = "1-8lz8Kf6XjQdeOZ1Hew1ysxpGz6ZOEg6"
    output_path_2 = os.path.join(args.output_dir, "a2_RestaurantReviews_FreshDump.tsv")
    download_from_drive(file_id_2, output_path_2)

    # NLTK corpora
    download_nltk_resources()


if __name__ == "__main__":
    main()
