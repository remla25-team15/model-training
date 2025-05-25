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
    return output_path


def download_nltk_resources():
    """
    Ensure required NLTK corpora are downloaded.
    """
    try:
        nltk.data.find("corpora/wordnet")
        return False  # Already present
    except LookupError:
        nltk.download("wordnet")
        return True


def run_get_data(output_dir="datasets", nltk_dir=None, file_ids=None, output_names=None):
    os.makedirs(output_dir, exist_ok=True)
    # Defaults for DVC pipeline
    if file_ids is None:
        file_ids = [
            "1_SHjQJVxZdr_LW2aIHAiOSBPWWGWd7Bs",
            "1-8lz8Kf6XjQdeOZ1Hew1ysxpGz6ZOEg6"
        ]
    if output_names is None:
        output_names = [
            "a1_RestaurantReviews_HistoricDump.tsv",
            "a2_RestaurantReviews_FreshDump.tsv"
        ]
    downloaded = []
    for file_id, name in zip(file_ids, output_names):
        out_path = os.path.join(output_dir, name)
        download_from_drive(file_id, out_path)
        downloaded.append(out_path)
    nltk_downloaded = download_nltk_resources()
    return {"downloaded": downloaded, "nltk_downloaded": nltk_downloaded}


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
    parser.add_argument(
        "--nltk_dir",
        type=str,
        default="nltk_data",
        help="Directory to save NLTK data.",
    )
    args = parser.parse_args()
    summary = run_get_data(output_dir=args.output_dir, nltk_dir=args.nltk_dir)
    print("Downloaded:", summary["downloaded"])
    print("NLTK wordnet downloaded:", summary["nltk_downloaded"])


if __name__ == "__main__":
    main()
