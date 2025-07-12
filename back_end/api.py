from transformers import BertTokenizer, TFBertForSequenceClassification, BertConfig
from preprocess import preprocess_text
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import tensorflow as tf
import pandas as pd
import numpy as np
import nltk
import praw
import os


config_file = BertConfig.from_json_file("/content/config.json")
model = TFBertForSequenceClassification.from_pretrained("/content/tf_model.h5", config=config_file)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")



load_dotenv()
reddit = praw.Reddit("bot1")


def fetch_reddit_comments(url:str, reddit_instance: praw.Reddit = reddit) -> list:
    """Fetches comments of a reddit post from a given url using praw"""
    submission = reddit_instance.submission(url=url)
    submission.comments.replace_more(limit=None)
    comments = []
    for comment in submission.comments.list():
        comments.append(comment.body)
    return comments



def tokenize_comments(comments: list) -> pd.DataFrame:
    cleaned_comments = preprocess_text(pd.DataFrame(comments))
    tokenized_comments = cleaned_comments.map(lambda x:tokenizer.encode_plus(text=x, add_special_tokens=True, padding="max_length", max_length=512, return_attention_mask=True, truncation=True, return_tensors="tf"))
    return tokenized_comments, cleaned_comments


def prepare_dataset(tokenized_comments:pd.DataFrame, cleaned_comments:pd.DataFrame) -> tf.data.Dataset:
    input_ids = [x["input_ids"] for x in tokenized_comments[0]]
    attn_mask = [x["attention_mask"] for x in tokenized_comments[0]]
    input_ids_tensor = tf.convert_to_tensor(input_ids)
    attn_mask_tensor = tf.convert_to_tensor(attn_mask)
    model_input = tf.data.Dataset.from_tensor_slices({"input_ids": input_ids_tensor, "attention_mask": attn_mask_tensor})
    return model_input, cleaned_comments


def classify_comments(model_input:tf.data.Dataset, cleaned_comments:pd.DataFrame) -> pd.DataFrame:
    model_ouput = model.predict(model_input)
    probs = pd.DataFrame(tf.nn.softmax(model_ouput[0]), columns=["negative", "neutral", "positive"])
    newcol = np.argmax(probs, axis=1).reshape(probs.shape[0], 1)
    reverse_label = {2:1, 1:0, 0:-1}
    cleaned_comments["labels"] = newcol
    cleaned_comments["labels"] = cleaned_comments["labels"].map(reverse_label)
    cleaned_comments["negative"] = probs["negative"]
    cleaned_comments["neutral"] = probs["neutral"]
    cleaned_comments["positive"] = probs["positive"]
    return cleaned_comments


def sort_comments(cleaned_comments:pd.DataFrame) -> pd.DataFrame:
    negatives_sorted = cleaned_comments[cleaned_comments["negative"]>0.5].sort_values("negative", ascending=False)
    neutral_sorted = cleaned_comments[cleaned_comments["neutral"]>0.5].sort_values("neutral", ascending=False)
    positive_sorted = cleaned_comments[cleaned_comments["positive"]>0.5].sort_values("positive", ascending=False)
    return negatives_sorted, neutral_sorted, positive_sorted


def generate_wordclouds(negatives_sorted:pd.DataFrame, neutral_sorted:pd.DataFrame, positive_sorted:pd.DataFrame) -> None:
    negative_wrdcld = WordCloud().generate(text=negatives_sorted[0].to_string())
    positive_wrdcld = WordCloud().generate(text=positive_sorted[0].to_string())
    neutral_wrdcld = WordCloud().generate(text=neutral_sorted[0].to_string())
    negative_wrdcld.to_file("negative_cloud.png")
    positive_wrdcld.to_file("positive_cloud.png")
    neutral_wrdcld.to_file("neutral_cloud.png")


def generate_output(url:str) -> pd.DataFrame:
    comments = fetch_reddit_comments(url)
    tokenized_comments, cleaned_comments = tokenize_comments(comments)
    model_input, cleaned_comments = prepare_dataset(tokenized_comments, cleaned_comments)
    cleaned_comments = classify_comments(model_input, cleaned_comments)
    negatives_sorted, neutral_sorted, positive_sorted = sort_comments(cleaned_comments)
    generate_wordclouds(negatives_sorted, neutral_sorted, positive_sorted)
    return negatives_sorted, neutral_sorted, positive_sorted

