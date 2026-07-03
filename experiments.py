import pandas as pd

from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from config import K_FOLD_VALUES, RANDOM_STATE


SCORING = {
    "accuracy": "accuracy",

    "precision_macro": "precision_macro",
    "recall_macro": "recall_macro",
    "f1_macro": "f1_macro",
    "f1_weighted": "f1_weighted",

    # Class 0: hate_speech
    "precision_hate_speech": make_scorer(
        precision_score,
        labels=[0],
        average="macro",
        zero_division=0
    ),
    "recall_hate_speech": make_scorer(
        recall_score,
        labels=[0],
        average="macro",
        zero_division=0
    ),
    "f1_hate_speech": make_scorer(
        f1_score,
        labels=[0],
        average="macro",
        zero_division=0
    ),

    # Class 1: offensive_language
    "precision_offensive_language": make_scorer(
        precision_score,
        labels=[1],
        average="macro",
        zero_division=0
    ),
    "recall_offensive_language": make_scorer(
        recall_score,
        labels=[1],
        average="macro",
        zero_division=0
    ),
    "f1_offensive_language": make_scorer(
        f1_score,
        labels=[1],
        average="macro",
        zero_division=0
    ),

    # Class 2: neither
    "precision_neither": make_scorer(
        precision_score,
        labels=[2],
        average="macro",
        zero_division=0
    ),
    "recall_neither": make_scorer(
        recall_score,
        labels=[2],
        average="macro",
        zero_division=0
    ),
    "f1_neither": make_scorer(
        f1_score,
        labels=[2],
        average="macro",
        zero_division=0
    ),
}


def create_pipeline(vectorizer, model) -> Pipeline:
    """
    Creates a machine learning pipeline.

    The pipeline contains two steps:
    1. Text vectorization
    2. Classification
    """

    return Pipeline([
        ("vectorizer", vectorizer),
        ("classifier", model)
    ])


def run_single_experiment(
    x,
    y,
    vectorizer_name: str,
    vectorizer,
    model_name: str,
    model,
    k: int
) -> dict:
    """
    Runs one experiment for a specific combination of:
    - vectorizer
    - model
    - k-fold value

    Returns mean and standard deviation for all evaluation metrics.
    """

    pipeline = create_pipeline(vectorizer, model)

    cross_validator = StratifiedKFold(
        n_splits=k,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    scores = cross_validate(
        estimator=pipeline,
        X=x,
        y=y,
        cv=cross_validator,
        scoring=SCORING,
        return_train_score=True,
        n_jobs=-1
    )

    result = {
        "vectorizer": vectorizer_name,
        "model": model_name,
        "k": k,
    }

    for metric_name in SCORING.keys():
        result[f"train_{metric_name}_mean"] = scores[f"train_{metric_name}"].mean()
        result[f"train_{metric_name}_std"] = scores[f"train_{metric_name}"].std()

        result[f"validation_{metric_name}_mean"] = scores[f"test_{metric_name}"].mean()
        result[f"validation_{metric_name}_std"] = scores[f"test_{metric_name}"].std()

    result["fit_time_mean"] = scores["fit_time"].mean()
    result["score_time_mean"] = scores["score_time"].mean()

    return result


def run_all_experiments(
    x,
    y,
    vectorizers: dict,
    models: dict
) -> pd.DataFrame:
    """
    Runs all combinations of:
    - vectorizers
    - models
    - k values

    Returns a dataframe with all experiment results.
    """

    results = []

    total_experiments = len(vectorizers) * len(models) * len(K_FOLD_VALUES)
    current_experiment = 1

    for vectorizer_name, vectorizer in vectorizers.items():
        for model_name, model in models.items():
            for k in K_FOLD_VALUES:
                print(
                    f"\nRunning experiment {current_experiment}/{total_experiments}: "
                    f"{vectorizer_name} + {model_name}, k={k}"
                )

                result = run_single_experiment(
                    x=x,
                    y=y,
                    vectorizer_name=vectorizer_name,
                    vectorizer=vectorizer,
                    model_name=model_name,
                    model=model,
                    k=k
                )

                results.append(result)

                print(
                    "Validation Macro F1: "
                    f"{result['validation_f1_macro_mean']:.4f} "
                    f"± {result['validation_f1_macro_std']:.4f}"
                )

                print(
                    "Class Recall "
                    f"| Hate: {result['validation_recall_hate_speech_mean']:.4f} "
                    f"| Offensive: {result['validation_recall_offensive_language_mean']:.4f} "
                    f"| Neither: {result['validation_recall_neither_mean']:.4f}"
                )

                current_experiment += 1

    results_df = pd.DataFrame(results)

    return results_df