from urllib.error import HTTPError, URLError
from urllib.request import urlretrieve

from config import DATA_DIR, DATASET_PATH, DATASET_URL


def download_dataset() -> None:
    """
    Downloads the Davidson Hate Speech and Offensive Language Dataset
    from the official GitHub repository.
    """

    DATA_DIR.mkdir(exist_ok=True)

    if DATASET_PATH.exists():
        print(f"Dataset already exists at: {DATASET_PATH}")
        print("Skipping download.")
        return

    print("Downloading dataset...")
    print(f"Source: {DATASET_URL}")

    try:
        urlretrieve(DATASET_URL, DATASET_PATH)
    except HTTPError as error:
        raise RuntimeError(
            f"HTTP error occurred while downloading the dataset: {error}"
        ) from error
    except URLError as error:
        raise RuntimeError(
            f"URL error occurred while downloading the dataset: {error}"
        ) from error

    print(f"Dataset downloaded successfully to: {DATASET_PATH}")


def main() -> None:
    download_dataset()


if __name__ == "__main__":
    main()