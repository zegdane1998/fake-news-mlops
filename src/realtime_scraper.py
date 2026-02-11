from ntscraper import Nitter
import pandas as pd
import os
from datetime import datetime

def scrape_with_ntscraper():
    nitter = Nitter()
    # Updated query for US News and Politics
    query = "(Trump) lang:en"
    
    print(f"Starting scrape for US focus: {query}")
    
    try:
        # Pull 10 tweets from a Nitter instance
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
            # Keeping the same naming convention for your pipeline
            filename = f"data/new_scraped/tweets_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
            df.to_csv(filename, index=False)
            print(f"Successfully saved {len(df)} US-focused tweets.")
    except Exception as e:
        print(f"Scraper error: {e}")

if __name__ == "__main__":
    scrape_with_ntscraper()