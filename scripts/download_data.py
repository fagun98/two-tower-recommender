"""Download MovieLens dataset."""

import argparse
import zipfile
from pathlib import Path
from urllib.request import urlretrieve


def download_movielens(dataset: str = "ml-latest-small", data_dir: str = "data"):
    """Download and extract MovieLens dataset.

    Args:
        dataset: Dataset name ('ml-latest-small' or 'ml-100k')
        data_dir: Directory to save data
    """
    data_path = Path(data_dir)
    data_path.mkdir(exist_ok=True)

    if dataset == "ml-latest-small":
        url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
    elif dataset == "ml-100k":
        url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    zip_path = data_path / f"{dataset}.zip"
    extract_path = data_path / dataset

    if extract_path.exists():
        print(f"Dataset already exists at {extract_path}")
        return

    print(f"Downloading {dataset} from {url}...")
    urlretrieve(url, zip_path)

    print(f"Extracting to {data_path}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(data_path)

    zip_path.unlink()
    print(f"Dataset ready at {extract_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download MovieLens dataset")
    parser.add_argument(
        "--dataset",
        default="ml-latest-small",
        choices=["ml-latest-small", "ml-100k"],
        help="Dataset to download",
    )
    parser.add_argument("--data-dir", default="data", help="Data directory")

    args = parser.parse_args()
    download_movielens(args.dataset, args.data_dir)
