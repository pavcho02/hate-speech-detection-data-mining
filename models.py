from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB

from config import MAX_FEATURES, MIN_DF, NGRAM_RANGE, RANDOM_STATE


def get_vectorizers() -> dict:
    """
    Returns the text vectorizers used in the experiments.

    BoW is implemented with CountVectorizer.
    TF-IDF is implemented with TfidfVectorizer.
    """

    return {
        "BoW": CountVectorizer(
            max_features=MAX_FEATURES,
            ngram_range=NGRAM_RANGE,
            min_df=MIN_DF
        ),
        "TF-IDF": TfidfVectorizer(
            max_features=MAX_FEATURES,
            ngram_range=NGRAM_RANGE,
            min_df=MIN_DF
        )
    }


def get_models() -> dict:
    """
    Returns the machine learning models used in the experiments.

    Complement Naive Bayes is used as the Naive Bayes approach because it is
    better suited for imbalanced text classification problems than standard
    Multinomial Naive Bayes.

    Logistic Regression and Random Forest use class_weight='balanced'
    because they directly support this parameter.
    """

    return {
        "Complement Naive Bayes": ComplementNB(
            alpha=1.0
        ),

        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    }