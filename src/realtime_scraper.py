import tweepy
import pandas as pd
import os
from datetime import datetime
from dotenv import load_dotenv

# Load local .env for Eyüpsultan testing
load_dotenv()

def scrape_us_politics():
    bearer_token = os.getenv("X_BEARER_TOKEN")
    if not bearer_token:
        print("Error: X_BEARER_TOKEN missing.")
        return

    # Initialize X API v2 Client
    client = tweepy.Client(bearer_token=bearer_token)
    
    # Query: High-traffic US terms to guarantee data for your demo
    query = "(US News OR White House OR Congress) lang:en -is:retweet"
    
    try:
        print(f"Connecting to X API for query: {query}")
        tweets = client.search_recent_tweets(query=query, max_results=10)
        
        data = []
        if tweets.data:
            for tweet in tweets.data:
                data.append({
                    "text": tweet.text,
                    "scraped_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })
            
            df = pd.DataFrame(data)
            os.makedirs("data/new_scraped", exist_ok=True)
            filename = f"data/new_scraped/tweets_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
            df.to_csv(filename, index=False)
            print(f"Successfully saved {len(df)} US tweets to CSV.")
        else:
            print("No tweets found. X API might be returning empty for this query.")
            
    except tweepy.errors.TooManyRequests:
        print("Rate limit (429) hit. Please wait 15 minutes.")
    except Exception as e:
        print(f"Scraper error: {e}")

if __name__ == "__main__":
    scrape_us_politics()