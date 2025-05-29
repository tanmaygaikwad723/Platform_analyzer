import praw
import os 
from dotenv import load_dotenv

load_dotenv()

client_id = os.getenv("reddit_client")
client_secret = os.getenv("reddit_client_secret")
refresh_token = os.getenv("reddit_refresh_token")

reddit = praw.Reddit("bot1")

def fetch_reddit_comments(url:str, reddit_instance: praw.Reddit = reddit) -> list:
    """Fetches comments of a reddit post from a given url using praw"""
    submission = reddit_instance.submission(url=url)
    submission.comments.replace_more(limit=None)
    comments = []
    for comment in submission.comments.list():
        comments.append(comment.body)
    return comments

if __name__ == "__main__":
    reddit_url = str(input("Enter the url  of reddit post: "))
    print(fetch_reddit_comments(reddit_url))

