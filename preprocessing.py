import html
import re

import pandas as pd


URL_PATTERN = re.compile(r"http\S+|www\S+|https\S+")
MENTION_PATTERN = re.compile(r"@\w+")
HASHTAG_PATTERN = re.compile(r"#(\w+)")
RETWEET_PATTERN = re.compile(r"\brt\b")
NON_LETTER_PATTERN = re.compile(r"[^a-zA-Z\s]")
MULTIPLE_SPACES_PATTERN = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """
    Cleans a single tweet.

    The function applies moderate preprocessing:
    - converts text to lowercase
    - unescapes HTML entities
    - removes URLs
    - removes user mentions
    - keeps hashtag words but removes the # symbol
    - removes retweet marker 'rt'
    - removes punctuation, digits and special symbols
    - removes extra spaces
    """

    if not isinstance(text, str):
        return ""

    text = html.unescape(text)
    text = text.lower()

    text = URL_PATTERN.sub(" ", text)
    text = MENTION_PATTERN.sub(" ", text)
    text = HASHTAG_PATTERN.sub(r"\1", text)
    text = RETWEET_PATTERN.sub(" ", text)

    text = NON_LETTER_PATTERN.sub(" ", text)
    text = MULTIPLE_SPACES_PATTERN.sub(" ", text)

    return text.strip()


def preprocess_dataframe(
    df: pd.DataFrame,
    text_column: str = "tweet",
    output_column: str = "clean_text"
) -> pd.DataFrame:
    """
    Applies text preprocessing to a dataframe.

    A new column is created with the cleaned version of the original text.
    """

    if text_column not in df.columns:
        raise ValueError(
            f"Column '{text_column}' was not found in the dataframe. "
            f"Available columns are: {list(df.columns)}"
        )

    processed_df = df.copy()
    processed_df[output_column] = processed_df[text_column].apply(clean_text)

    return processed_df