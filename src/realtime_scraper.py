import os
import pandas as pd
from datetime import datetime
from newsapi import NewsApiClient
from dotenv import load_dotenv

load_dotenv()

def scrape_us_politics():
    api_key = os.getenv("NEWSAPI_KEY")
    if not api_key:
        print("Error: NEWSAPI_KEY missing.")
        return

    newsapi = NewsApiClient(api_key=api_key)

    print("Fetching US political news headlines...")
    response = newsapi.get_everything(
        q='congress OR election OR senate OR "white house" OR trump OR biden',
        language='en',
        sort_by='publishedAt',
        page_size=20
    )

    if response['status'] != 'ok' or not response['articles']:
        print("No articles returned.")
        return

    data = []
    for article in response['articles']:
        title = article.get('title', '')
        if title and '[Removed]' not in title:
            data.append({
                "text": title,
                "scraped_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "source": article.get('source', {}).get('name', 'Unknown')
            })

    df = pd.DataFrame(data)
    os.makedirs("data/new_scraped", exist_ok=True)
    filename = f"data/new_scraped/news_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    df.to_csv(filename, index=False)
    print(f"Saved {len(df)} headlines to {filename}")

if __name__ == "__main__":
    scrape_us_politics()