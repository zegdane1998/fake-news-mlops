from ntscraper import Nitter
import pandas as pd
import os
from datetime import datetime

def scrape_with_ntscraper():
    nitter = Nitter()
    # Search for broader political terms to ensure we find news
    query = "politics news Istanbul"
    
    print(f"Starting scrape for: {query}")
    
    try:
        # Scrape 10 tweets from a public Nitter instance
        results = nitter.get_tweets(query, mode='term', number=10)
        
        tweets_data = []
        for tweet in results['tweets']:
            tweets_data.append({
                "text": tweet['text'],
                "scraped_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })

        if tweets_data:
            df = pd.DataFrame(tweets_data)
            os.makedirs("data/new_scraped", exist_ok=True)
            filename = f"data/new_scraped/tweets_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
            df.to_csv(filename, index=False)
            print(f"Successfully saved {len(df)} tweets to {filename}")
        else:
            print("No tweets found. All instances might be down.")

    except Exception as e:
        print(f"Nitter Scraper error: {e}")

if __name__ == "__main__":
    scrape_with_ntscraper()