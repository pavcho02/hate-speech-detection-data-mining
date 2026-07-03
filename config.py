from pathlib import Path


# Main project folders
DATA_DIR = Path("data")
RESULTS_DIR = Path("results")


# Original dataset file
DATASET_FILENAME = "labeled_data.csv"
DATASET_PATH = DATA_DIR / DATASET_FILENAME


# Processed dataset file
PROCESSED_DATASET_FILENAME = "processed_labeled_data.csv"
PROCESSED_DATASET_PATH = DATA_DIR / PROCESSED_DATASET_FILENAME


# Experiment results file
EXPERIMENT_RESULTS_FILENAME = "experiment_results.csv"
EXPERIMENT_RESULTS_PATH = RESULTS_DIR / EXPERIMENT_RESULTS_FILENAME


# Report output files
EXPERIMENT_REPORT_CSV_PATH = RESULTS_DIR / "experiment_report.csv"
EXPERIMENT_REPORT_MARKDOWN_PATH = RESULTS_DIR / "experiment_report.md"


# Davidson Hate Speech and Offensive Language Dataset
DATASET_URL = (
    "https://raw.githubusercontent.com/"
    "t-davidson/hate-speech-and-offensive-language/"
    "master/data/labeled_data.csv"
)


# Class labels from the dataset
CLASS_LABELS = {
    0: "hate_speech",
    1: "offensive_language",
    2: "neither"
}


# Experimental settings
RANDOM_STATE = 42

K_FOLD_VALUES = [3, 5, 10]

TEXT_COLUMN = "clean_text"
TARGET_COLUMN = "class"


# Vectorizer settings
MAX_FEATURES = 5000
NGRAM_RANGE = (1, 2)
MIN_DF = 2