import pandas as pd

from config import (
    CLASS_LABELS,
    DATASET_PATH,
    EXPERIMENT_RESULTS_PATH,
    PROCESSED_DATASET_PATH,
    RESULTS_DIR,
    TARGET_COLUMN,
    TEXT_COLUMN
)
from experiments import run_all_experiments
from models import get_models, get_vectorizers
from preprocessing import preprocess_dataframe


def load_dataset() -> pd.DataFrame:
    """
    Loads the Davidson Hate Speech and Offensive Language Dataset.
    """

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset file was not found at: {DATASET_PATH}\n"
            "Run the following command first:\n"
            "python download_data.py"
        )

    df = pd.read_csv(DATASET_PATH)

    return df


def validate_dataset(df: pd.DataFrame) -> None:
    """
    Checks whether the dataset contains the required columns.
    """

    required_columns = {"tweet", "class"}

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"The dataset is missing required columns: {missing_columns}\n"
            f"Available columns are: {list(df.columns)}"
        )


def print_basic_info(df: pd.DataFrame) -> None:
    """
    Prints basic information about the dataset.
    """

    print("\n==============================")
    print("BASIC DATASET INFORMATION")
    print("==============================")

    print(f"\nNumber of rows: {df.shape[0]}")
    print(f"Number of columns: {df.shape[1]}")

    print("\nColumns:")
    print(list(df.columns))

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDuplicated tweets:")
    duplicated_count = df["tweet"].duplicated().sum()
    print(duplicated_count)


def analyze_class_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyzes the distribution of the target classes.
    """

    class_distribution = (
        df["class"]
        .value_counts()
        .sort_index()
        .reset_index()
    )

    class_distribution.columns = ["class", "count"]
    class_distribution["class_name"] = class_distribution["class"].map(CLASS_LABELS)
    class_distribution["percentage"] = (
        class_distribution["count"] / class_distribution["count"].sum() * 100
    ).round(2)

    print("\n==============================")
    print("CLASS DISTRIBUTION")
    print("==============================")

    print(class_distribution)

    return class_distribution


def print_sample_tweets_by_class(
    df: pd.DataFrame,
    samples_per_class: int = 3
) -> None:
    """
    Prints a few example tweets from each class.
    """

    print("\n==============================")
    print("SAMPLE TWEETS BY CLASS")
    print("==============================")

    for class_id, class_name in CLASS_LABELS.items():
        print(f"\nClass {class_id}: {class_name}")
        print("-" * 50)

        class_samples = (
            df[df["class"] == class_id]["tweet"]
            .dropna()
            .head(samples_per_class)
        )

        for index, tweet in enumerate(class_samples, start=1):
            print(f"{index}. {tweet}")


def print_preprocessing_examples(
    df: pd.DataFrame,
    samples: int = 5
) -> None:
    """
    Prints examples of original and cleaned text.
    """

    print("\n==============================")
    print("PREPROCESSING EXAMPLES")
    print("==============================")

    examples = df[["tweet", "clean_text"]].head(samples)

    for index, row in examples.iterrows():
        print(f"\nExample {index + 1}")
        print("Original:")
        print(row["tweet"])
        print("Cleaned:")
        print(row["clean_text"])


def analyze_clean_text(df: pd.DataFrame) -> None:
    """
    Prints basic information about the cleaned text column.
    """

    print("\n==============================")
    print("CLEAN TEXT ANALYSIS")
    print("==============================")

    empty_clean_text_count = (df["clean_text"].str.len() == 0).sum()

    df["original_text_length"] = df["tweet"].astype(str).str.len()
    df["clean_text_length"] = df["clean_text"].astype(str).str.len()

    print(f"\nEmpty cleaned texts: {empty_clean_text_count}")

    print("\nAverage original text length:")
    print(round(df["original_text_length"].mean(), 2))

    print("\nAverage cleaned text length:")
    print(round(df["clean_text_length"].mean(), 2))


def save_class_distribution(class_distribution: pd.DataFrame) -> None:
    """
    Saves class distribution results to the results folder.
    """

    RESULTS_DIR.mkdir(exist_ok=True)

    output_path = RESULTS_DIR / "class_distribution.csv"
    class_distribution.to_csv(output_path, index=False)

    print("\nClass distribution saved to:")
    print(output_path)


def save_processed_dataset(df: pd.DataFrame) -> None:
    """
    Saves the dataset with the cleaned text column.
    """

    PROCESSED_DATASET_PATH.parent.mkdir(exist_ok=True)

    df.to_csv(PROCESSED_DATASET_PATH, index=False)

    print("\nProcessed dataset saved to:")
    print(PROCESSED_DATASET_PATH)


def prepare_experiment_data(df: pd.DataFrame):
    """
    Prepares input features and target labels for experiments.
    """

    if TEXT_COLUMN not in df.columns:
        raise ValueError(f"Text column '{TEXT_COLUMN}' was not found.")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' was not found.")

    experiment_df = df[[TEXT_COLUMN, TARGET_COLUMN]].copy()

    experiment_df[TEXT_COLUMN] = experiment_df[TEXT_COLUMN].fillna("")
    experiment_df = experiment_df[experiment_df[TEXT_COLUMN].str.len() > 0]

    x = experiment_df[TEXT_COLUMN]
    y = experiment_df[TARGET_COLUMN]

    print("\n==============================")
    print("EXPERIMENT DATA")
    print("==============================")

    print(f"\nNumber of usable rows: {len(experiment_df)}")
    print(f"Text column: {TEXT_COLUMN}")
    print(f"Target column: {TARGET_COLUMN}")

    return x, y


def save_experiment_results(results_df: pd.DataFrame) -> None:
    """
    Saves experiment results to the results folder.
    """

    RESULTS_DIR.mkdir(exist_ok=True)

    results_df.to_csv(EXPERIMENT_RESULTS_PATH, index=False)

    print("\nExperiment results saved to:")
    print(EXPERIMENT_RESULTS_PATH)


def print_best_results(results_df: pd.DataFrame) -> None:
    """
    Prints the best experiment configurations sorted by validation macro F1-score.
    """

    print("\n==============================")
    print("BEST RESULTS BY VALIDATION MACRO F1")
    print("==============================")

    columns_to_show = [
        "vectorizer",
        "model",
        "k",
        "validation_accuracy_mean",
        "validation_precision_macro_mean",
        "validation_recall_macro_mean",
        "validation_f1_macro_mean",
        "validation_f1_macro_std",
        "train_f1_macro_mean"
    ]

    best_results = (
        results_df[columns_to_show]
        .sort_values(by="validation_f1_macro_mean", ascending=False)
        .head(10)
    )

    print(best_results.to_string(index=False))


def main() -> None:
    """
    Main execution function for Stage 3.
    """

    print("Loading dataset...")

    df = load_dataset()

    validate_dataset(df)

    print_basic_info(df)

    class_distribution = analyze_class_distribution(df)

    print_sample_tweets_by_class(df)

    print("\nApplying text preprocessing...")

    processed_df = preprocess_dataframe(df)

    print_preprocessing_examples(processed_df)

    analyze_clean_text(processed_df)

    save_class_distribution(class_distribution)

    save_processed_dataset(processed_df)

    x, y = prepare_experiment_data(processed_df)

    vectorizers = get_vectorizers()
    models = get_models()

    print("\n==============================")
    print("RUNNING MACHINE LEARNING EXPERIMENTS")
    print("==============================")

    results_df = run_all_experiments(
        x=x,
        y=y,
        vectorizers=vectorizers,
        models=models
    )

    save_experiment_results(results_df)

    print_best_results(results_df)

if __name__ == "__main__":
    main()