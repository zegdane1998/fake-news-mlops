import os
import re
import pandas as pd
import tweepy
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Political query — filters retweets, replies, and non-English tweets
QUERY = (
    "(congress OR election OR senate OR \"white house\" OR trump OR biden "
    "OR republican OR democrat OR \"supreme court\" OR legislation OR ballot) "
    "lang:en -is:retweet -is:reply"
)


def _clean_tweet(text):
    """Remove URLs, mentions, and extra whitespace from tweet text."""
    text = re.sub(r'http\S+|www\S+', '', text)   # remove URLs
    text = re.sub(r'@\w+', '', text)              # remove @mentions
    text = re.sub(r'\s+', ' ', text).strip()      # normalize whitespace
    return text


def scrape_us_politics(max_results=25):  # usage-based: 25/day × 30 = 750 tweets/month × $0.005 = ~$3.75/mo
    bearer_token = os.getenv("X_BEARER_TOKEN")
    if not bearer_token:
        print("Error: X_BEARER_TOKEN missing from .env")
        return

    client = tweepy.Client(bearer_token=bearer_token, wait_on_rate_limit=True)

    print("Fetching US political tweets from X...")
    try:
        response = client.search_recent_tweets(
            query=QUERY,
            max_results=max_results,          # 10–100 per request (API limit)
            tweet_fields=["created_at", "author_id", "source", "text"],
            expansions=["author_id"],
            user_fields=["username", "verified"],
        )
    except tweepy.TweepyException as e:
        print(f"Twitter API error: {e}")
        return

    if not response.data:
        print("No tweets returned.")
        return

    # Build author id → username map from includes
    user_map = {}
    if response.includes and "users" in response.includes:
        for user in response.includes["users"]:
            user_map[user.id] = user.username

    data = []
    for tweet in response.data:
        cleaned = _clean_tweet(tweet.text)
        if len(cleaned) < 20:          # skip very short/empty tweets after cleaning
            continue
        data.append({
            "text":       cleaned,
            "scraped_at": tweet.created_at.strftime('%Y-%m-%d %H:%M:%S')
                          if tweet.created_at else datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "source":     f"@{user_map.get(tweet.author_id, 'unknown')}",
        })

    if not data:
        print("No usable tweets after filtering.")
        return

    df = pd.DataFrame(data)
    os.makedirs("data/new_scraped", exist_ok=True)
    filename = f"data/new_scraped/news_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    df.to_csv(filename, index=False)
    print(f"Saved {len(df)} tweets to {filename}")


if __name__ == "__main__":
    scrape_us_politics()
