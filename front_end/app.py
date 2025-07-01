import streamlit as st
from back_end.helpers import fetch_reddit_comments

st.title("Platform Analyzer")

link = st.text_input("Enter the URL of the reddit post : ")

def check_url(url:str) -> bool:
    if url.startswith("https://www.reddit.com/r/") or url.startswith("https://reddit.com/r/"):
      print(fetch_reddit_comments(url))
      return True
    else:
      return False

comments = st.button("Analyze", on_click=check_url, args=(link,))

print(comments)