import snscrape.modules.twitter as sntwitter
import pandas as pd
import os
from datetime import datetime

def scrape_with_snscrape():
    # Define a high-traffic query to guarantee data for your thesis
    query = "politics news Istanbul lang:en"
    tweets_list = []

    print(f"Starting scrape for query: {query}")

    try:
        # Scrape the 10 most recent tweets
        for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
            if i >= 10: # Limiting to 10 for a fast demo
                break
            tweets_list.append({
                "text": tweet.rawContent,
                "scraped_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })

        if tweets_list:
            df = pd.DataFrame(tweets_list)
            os.makedirs("data/new_scraped", exist_ok=True)
            filename = f"data/new_scraped/tweets_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
            df.to_csv(filename, index=False)
            print(f"Successfully saved {len(df)} tweets to {filename}")
        else:
            print("No tweets found. X might be blocking the scraper.")

    except Exception as e:
        print(f"Snscrape error: {e}")

if __name__ == "__main__":
    scrape_with_snscrape()