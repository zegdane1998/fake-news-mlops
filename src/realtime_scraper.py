import os
import re
import pandas as pd
import tweepy
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Curated list of credible English-language news outlets on X/Twitter
NEWS_ACCOUNTS = [
    "Reuters", "AP", "BBCNews", "BBCWorld",
    "CNN", "CNNPolitics", "nytimes", "washingtonpost",
    "guardian", "AJEnglish", "NBCNews", "CBSNews",
    "ABC", "Bloomberg", "politico", "NPR",
    "thehill", "axios", "ForeignPolicy", "MiddleEastEye",
    "WSJ", "time", "TheEconomist", "Independent",
]

# Political/geopolitical keywords relevant to the thesis domain
KEYWORDS = (
    "Iran OR IRGC OR Tehran OR Khamenei OR \"Strait of Hormuz\" "
    "OR airstrike OR \"nuclear deal\" OR sanctions OR \"Persian Gulf\" "
    "OR Houthis OR \"proxy war\" OR Trump OR Biden OR Congress OR Pentagon"
)

def _build_query():
    accounts = " OR ".join(f"from:{a}" for a in NEWS_ACCOUNTS)
    # X API query limit is 1024 chars; keywords + accounts + filters
    query = f"({KEYWORDS}) ({accounts}) has:links lang:en -is:retweet -is:reply"
    if len(query) > 1024:
        # Fallback: accounts-only query without keyword filter
        query = f"({accounts}) has:links lang:en -is:retweet -is:reply"
    return query


def _clean_text(text):
    text = re.sub(r'http\S+|www\S+', '', text)   # remove URLs
    text = re.sub(r'@\w+', '', text)              # remove @mentions
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def scrape_news(max_results=25):
    bearer_token = os.getenv("X_BEARER_TOKEN")
    if not bearer_token:
        print("Error: X_BEARER_TOKEN missing from .env")
        return

    client = tweepy.Client(bearer_token=bearer_token, wait_on_rate_limit=True)
    query = _build_query()
    print(f"Query ({len(query)} chars): {query[:120]}...")

    try:
        response = client.search_recent_tweets(
            query=query,
            max_results=max_results,
            tweet_fields=["created_at", "author_id", "text"],
            expansions=["author_id"],
            user_fields=["username"],
        )
    except tweepy.TweepyException as e:
        print(f"Twitter API error: {e}")
        return

    if not response.data:
        print("No tweets returned.")
        return

    user_map = {}
    if response.includes and "users" in response.includes:
        for user in response.includes["users"]:
            user_map[user.id] = user.username

    data = []
    for tweet in response.data:
        cleaned = _clean_text(tweet.text)
        if len(cleaned) < 20:
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

    master_path = "data/new_scraped/all_tweets.csv"
    if os.path.exists(master_path):
        master = pd.read_csv(master_path)
        master = pd.concat([master, df], ignore_index=True).drop_duplicates(subset=["text"])
    else:
        master = df
    master.to_csv(master_path, index=False)
    print(f"Master CSV updated: {len(master)} total tweets → {master_path}")


if __name__ == "__main__":
    scrape_news()
