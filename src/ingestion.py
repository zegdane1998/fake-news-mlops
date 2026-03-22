import pandas as pd
import os

BASE_URL = "https://raw.githubusercontent.com/KaiDMML/FakeNewsNet/master/dataset/"

def _load_dataset(fake_url, real_url, name):
    """Download and label a FakeNewsNet dataset split."""
    print(f"Downloading {name} dataset...")
    df_fake = pd.read_csv(fake_url)
    df_real = pd.read_csv(real_url)
    df_fake['label'] = 0
    df_real['label'] = 1
    combined = pd.concat([df_fake, df_real], ignore_index=True)
    print(f"  {name}: {len(df_fake)} fake + {len(df_real)} real = {len(combined)} total")
    return combined

def load_gossipcop_data():
    os.makedirs('data/raw', exist_ok=True)

    # GossipCop: entertainment/celebrity (broad fake news patterns)
    gossipcop = _load_dataset(
        BASE_URL + "gossipcop_fake.csv",
        BASE_URL + "gossipcop_real.csv",
        "GossipCop"
    )

    # PolitiFact: fact-checked political claims (directly relevant to political fake news)
    politifact = _load_dataset(
        BASE_URL + "politifact_fake.csv",
        BASE_URL + "politifact_real.csv",
        "PolitiFact"
    )

    df_combined = pd.concat([gossipcop, politifact], ignore_index=True)

    # Keep only what the pipeline needs
    df_combined = df_combined[['title', 'label']].dropna(subset=['title'])
    df_combined = df_combined[df_combined['title'].str.strip() != '']

    print(f"\nTotal combined dataset: {len(df_combined)} rows")
    print(f"  Fake (0): {(df_combined['label'] == 0).sum()}")
    print(f"  Real (1): {(df_combined['label'] == 1).sum()}")
    return df_combined

if __name__ == "__main__":
    df = load_gossipcop_data()
    df.to_csv('data/raw/gossipcop_combined.csv', index=False)
    print("Consolidated file saved to data/raw/gossipcop_combined.csv")