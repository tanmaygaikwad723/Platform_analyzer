import nltk
import re
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)

def remove_punctuation_numbers(text: pd.DataFrame) -> pd.DataFrame:
    """Remove punctuation and numbers from the text dataframe."""
    cleaned_text = text.apply(lambda x: re.sub(r'[^\w\s]', '', x))
    return cleaned_text


def lowercase_text(text: pd.DataFrame) -> pd.DataFrame:
    """ Converts text to lowercase. """
    lowered_text = text.apply(lambda x: x.lower())
    return lowered_text


def remove_stopwords(text: pd.DataFrame) -> pd.DataFrame:
    """ Remove stopwwords from the text dataframe."""
    stop_words = set(stopwords.words("english"))
    filtered_text = text.apply(lambda x: ''.join([word for word in x.split() if word not in stop_words]))
    return filtered_text


def lemmatize_text(text: pd.DataFrame) -> pd.DataFrame:
    """ Lemmatizes the text in the dataframe. """
    lemmatizer = WordNetLemmatizer()
    lemmatized_text = text.apply(lambda x: ''.join([lemmatizer.lemmatize(word) for word in x]))
    return lemmatized_text


