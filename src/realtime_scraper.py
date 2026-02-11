import tweepy
import pandas as pd
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

def scrape_us_politics():
    # Retrieve the secret we saved in GitHub/local .env
    bearer_token = os.getenv("X_BEARER_TOKEN")
    if not bearer_token:
        print("Error: X_BEARER_TOKEN not found.")
        return

    client = tweepy.Client(bearer_token=bearer_token)
    
    # Query focused on US high-traffic political terms
    query = "(US News OR White House OR Congress) lang:en -is:retweet"
    
    try:
        # Max results 10 is the limit for Free Tier
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
            print(f"Successfully scraped {len(df)} US tweets.")
        else:
            print("No new tweets found for the US query.")
            
    except tweepy.errors.TooManyRequests:
        print("Rate limit reached (429). Wait 15 minutes.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    scrape_us_politics()