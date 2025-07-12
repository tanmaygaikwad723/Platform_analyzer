import nltk
import re
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)

def remove_punctuation_numbers(text: pd.DataFrame) -> pd.DataFrame:
    """Remove punctuation and numbers from the text dataframe."""
    cleaned_text = text.map(lambda x: re.sub(r'[^A-Za-z0-9 ]+', "", x))
    return cleaned_text


def lowercase_text(text: pd.DataFrame) -> pd.DataFrame:
    """ Converts text to lowercase. """
    lowered_text = text.map(lambda x: x.lower())
    return lowered_text


def remove_stopwords(text: pd.DataFrame) -> pd.DataFrame:
    """ Remove stopwwords from the text dataframe."""
    stop_words = set(stopwords.words("english"))
    filtered_text = text.map(lambda x: "".join([word for word in x.split(" ") if word not in stop_words]))
    return filtered_text


def lemmatize_text(text: pd.DataFrame) -> pd.DataFrame:
    """ Lemmatizes the text in the dataframe. """
    lemmatizer = WordNetLemmatizer()
    lemmatized_text = text.map(lemmatizer.lemmatize)
    return lemmatized_text


def remove_url(text: pd.DataFrame) -> pd.DataFrame:
    """ Remove URL's from the text dataframe."""
    cleaned_text = text.map(lambda x: re.sub(r'http\S+', "", x))
    return cleaned_text


def remove_html_tags(text: pd.DataFrame) -> pd.DataFrame:
    """ Remove HTML tags from the text dataframe. """
    cleaned_text = text.map(lambda x: re.sub(r'<.*?>', "", x))
    return cleaned_text


def preprocess_text(text: pd.DataFrame) -> pd.DataFrame:
    """ Preprocess the text in the dataframe. """
    text = remove_punctuation_numbers(text)
    print(text.head(1))
    text = remove_url(text)
    print(text.head(1))
    text = remove_html_tags(text)
    print(text.head(1))
    text = lowercase_text(text)
    print(text.head(1))
    return text