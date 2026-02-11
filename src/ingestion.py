import pandas as pd
import os

def load_gossipcop_data():
    # Base URLs for the GossipCop subset of FakeNewsNet
    base_url = "https://raw.githubusercontent.com/KaiDMML/FakeNewsNet/master/dataset/"
    fake_path = base_url + "gossipcop_fake.csv"
    real_path = base_url + "gossipcop_real.csv"
    
    if not os.path.exists('data/raw'):
        os.makedirs('data/raw')

    print("Downloading GossipCop Real and Fake datasets...")
    
    # Load separate files
    df_fake = pd.read_csv(fake_path)
    df_real = pd.read_csv(real_path)
    
    # Add the labels (0 for Fake, 1 for Real)
    df_fake['label'] = 0
    df_real['label'] = 1
    
    # Merge them into one
    df_combined = pd.concat([df_fake, df_real], ignore_index=True)
    
    # Use the 'title' column as your main text feature
    print(f"Loaded {len(df_combined)} rows from GossipCop.")
    print(df_combined[['title', 'label']].head())
    
    return df_combined

if __name__ == "__main__":
    df = load_gossipcop_data()
    # Save the consolidated version for your pipeline
    df.to_csv('data/raw/gossipcop_combined.csv', index=False)
    print("Consolidated file saved to data/raw/gossipcop_combined.csv")