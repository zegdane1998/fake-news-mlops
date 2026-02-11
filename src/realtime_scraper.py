import tweepy
import pandas as pd
import os
from datetime import datetime
from dotenv import load_dotenv

# Load for local testing; GitHub Actions will use its Secrets
load_dotenv()

def scrape_political_news():
    # Retrieve token from environment variables
    bearer_token = os.getenv("X_BEARER_TOKEN")
    if not bearer_token:
        raise ValueError("X_BEARER_TOKEN is missing!")

    client = tweepy.Client(bearer_token=bearer_token)
    
    # Query optimized for your thesis focus
    query = "election OR policy OR Istanbul -is:retweet"
    
    try:
        # max_results=10 is mandatory for the X Free Tier
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
            print(f"Successfully saved {len(df)} tweets to {filename}")
        else:
            print("No new tweets found for the current query.")
            
    except tweepy.errors.Forbidden as e:
        print(f"Access Denied (403): Check your X API Free Tier limits. {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    scrape_political_news()