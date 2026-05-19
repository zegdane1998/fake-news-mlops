"""
Controlled drift injection for demonstrating the FakeShield detection pipeline.

The monitor computes PSI and KS statistics between BERTweet's confidence scores
on live tweets vs. the PHEME training reference.  Because BERTweet was fine-tuned
exclusively on PHEME rumour tweets (breaking-news events, political crises), it
produces strongly polarised confidence scores on in-distribution data (peaks near
0 and 1).  When the incoming stream contains tweets from a completely different
domain, the model becomes uncertain → scores cluster near 0.5 → the distribution
shifts → PSI ≥ 0.25 and KS p < 0.05 trigger.

This script injects 200 out-of-distribution tweets (sports / entertainment /
celebrity gossip / finance) into data/new_scraped/drift_injection.csv, then runs
the drift detector so the shift is immediately visible.

Usage:
    python src/inject_drift.py            # inject + run detector
    python src/inject_drift.py --inject-only   # only write the CSV
    python src/inject_drift.py --detect-only   # only run the detector
"""

import argparse
import os
import re
import sys
from datetime import datetime

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)

# ── Out-of-distribution tweets ────────────────────────────────────────────────
# These are from domains completely absent from PHEME:
#   sports results, celebrity news, finance, music, TV shows, lifestyle.
# BERTweet has never seen these topics → model uncertainty → drift signal.

OOD_TWEETS = [
    # Sports
    "Messi scores a hat-trick as Argentina beats Brazil 3-1 in Copa América final",
    "LeBron James announces he will play one more season before retirement",
    "Real Madrid wins Champions League for the 16th time after penalty shootout",
    "Serena Williams comeback: wins first match since retirement announcement",
    "Manchester City breaks Premier League points record with 100 points",
    "Tiger Woods spotted practicing at Augusta ahead of The Masters",
    "NBA trade deadline: Lakers acquire All-Star guard in blockbuster deal",
    "Ronaldo breaks international goals record with 130th strike for Portugal",
    "Super Bowl LVIII draws record 123 million viewers in the US",
    "Lewis Hamilton signs historic $100M deal to drive for Ferrari",
    "Novak Djokovic wins 25th Grand Slam title at Wimbledon",
    "Paris 2024 Olympics: USA tops medal table with 40 gold medals",
    "Formula 1: Max Verstappen clinches fourth consecutive world championship",
    "NBA Finals: Boston Celtics beat Miami Heat 4-1 to win the title",
    "World Cup 2026: FIFA confirms 48-team format across US, Canada and Mexico",
    "Kylian Mbappe scores twice on Real Madrid debut at Santiago Bernabeu",
    "Rafael Nadal announces retirement from professional tennis at 38",
    "Stephen Curry breaks all-time three-point record in Warriors win",
    "Premier League: Arsenal win first title in 20 years on final day drama",
    "Tom Brady to join ESPN as lead NFL analyst after retirement",

    # Celebrity / Entertainment
    "Taylor Swift breaks Spotify record with 100 million streams in a single day",
    "Beyoncé announces world tour with 80 dates across six continents",
    "Brad Pitt and Angelina Jolie finally reach divorce settlement after 8 years",
    "Netflix releases trailer for the most expensive series ever made",
    "Oscar nominations: Everything Everywhere All At Once leads with 11 nods",
    "Billie Eilish wins Grammy for Album of the Year for the third time",
    "New Marvel movie breaks opening weekend box office record with $250M",
    "Kim Kardashian launches new skincare line valued at $1 billion",
    "Adele postpones Las Vegas residency due to back injury surgery",
    "Drake and Kendrick Lamar beef ends with surprise joint concert",
    "The Weeknd announces final album before retiring from music",
    "Jennifer Lopez and Ben Affleck separate after two years of marriage",
    "Game of Thrones prequel breaks HBO viewership record on premiere night",
    "Rihanna confirms she is pregnant with her third child",
    "Disney announces live-action remakes of 12 classic animated films",
    "Harry Styles wins three Brit Awards including Artist of the Year",
    "Netflix loses 2 million subscribers in Q1 earnings shock",
    "Dua Lipa's new album debuts at number one in 47 countries",
    "Spider-Man: No Way Home becomes fifth film to cross $1 billion worldwide",
    "Kanye West legally changes name to Ye in Los Angeles court",

    # Finance / Economy
    "Federal Reserve raises interest rates by 0.75% in surprise move",
    "Bitcoin surges past $80,000 as institutional investment accelerates",
    "Apple becomes first company to reach $4 trillion market cap",
    "US inflation falls to 2.1% for first time since 2021",
    "Amazon announces 15,000 layoffs as part of cost-cutting restructuring",
    "Goldman Sachs posts worst quarterly earnings in a decade",
    "Tesla stock drops 12% after Elon Musk sells $4 billion in shares",
    "Oil prices hit $120 per barrel amid Middle East supply concerns",
    "JPMorgan CEO warns of economic hurricane heading for the US",
    "Eurozone GDP growth beats expectations at 0.6% in Q2",
    "Google parent Alphabet reports $20 billion profit in Q3",
    "IMF upgrades global growth forecast to 3.2% for 2025",
    "US unemployment rate falls to historic low of 3.2%",
    "Meta announces 11,000 job cuts in major restructuring plan",
    "Bank of England raises interest rates to 5.25% to combat inflation",
    "Nvidia becomes second most valuable company in the world",
    "Chinese economy grows 5.2% in 2024 beating government target",
    "Microsoft acquires gaming company for $69 billion in largest tech deal",
    "S&P 500 closes at all-time high above 5,500 points",
    "EU imposes record €4 billion antitrust fine on Google",

    # Technology / Science
    "OpenAI releases GPT-5 with reasoning capabilities that surpass human experts",
    "SpaceX Starship successfully completes first orbital flight and landing",
    "Apple releases Vision Pro 2 with 30% lighter design and all-day battery",
    "Scientists confirm water ice found in permanently shadowed craters on the Moon",
    "World's first fully autonomous taxi service launches in Tokyo",
    "Meta launches new social network to compete with Twitter X",
    "Samsung announces foldable phone with paper-thin display",
    "NASA confirms James Webb telescope images show signs of ancient galaxies",
    "Microsoft Copilot integrated into Windows 12 operating system",
    "World's fastest supercomputer breaks exaflop barrier at US lab",
    "Google DeepMind AI solves 50-year-old protein folding challenge",
    "TikTok reaches 2 billion monthly active users milestone",
    "First human patients receive brain chip implant from Neuralink",
    "Electric vehicles account for 25% of new car sales in Europe",
    "Scientists grow functional human kidney in laboratory for transplant",
    "iPhone 17 features satellite messaging and no charging port",
    "YouTube announces premium subscription price increase to $15.99 per month",
    "World's largest solar farm opens in Saudi Arabia covering 200 square km",
    "Amazon drone delivery service expands to 50 US cities",
    "First commercial quantum computer available for cloud rental at $10 per hour",

    # Lifestyle / Food / Travel
    "Mediterranean diet ranked number one for health for seventh consecutive year",
    "Airbnb bans all indoor security cameras in properties worldwide",
    "New study finds coffee drinkers live longer than non-coffee drinkers",
    "World's best restaurant 2024 named: Noma in Copenhagen tops the list",
    "Travel boom: record 1.4 billion international tourist arrivals in 2024",
    "Plant-based meat sales decline for third consecutive year in the US",
    "Study shows people who sleep 7-8 hours have 30% lower heart disease risk",
    "Paris ranked most visited city in the world ahead of Bangkok and London",
    "Ozempic demand surges as weight loss drug approval expands",
    "New research shows walking 8,000 steps per day reduces mortality risk",

    # Random / Ambiguous (most uncertain for the model)
    "This is amazing you have to see this to believe it",
    "Nobody expected what happened next at the event last night",
    "Scientists are shocked by what they discovered in the deep ocean",
    "You will not believe the price of this new gadget just released",
    "Local restaurant named best in the country by food critics",
    "New app lets you earn money just by walking around the city",
    "Doctors reveal one simple trick that changes everything about dieting",
    "This breed of dog is officially the most popular in the United States",
    "Record number of tourists visit this small European village",
    "Study finds that people who wake up early earn more money on average",
    "The best smartphone of the year has finally been announced",
    "Experts say this is the worst financial decision most people make",
    "New fashion trend takes over social media in just 48 hours",
    "This city just banned cars from its entire downtown area",
    "Scientists discover new species of bird in the Amazon rainforest",
    "The most popular baby names of 2024 have been revealed",
    "New study says remote workers are more productive than office workers",
    "This airline just announced free checked bags for all passengers",
    "World record broken for the longest time without sleep",
    "Researchers find that laughing for 15 minutes burns 40 calories",

    # More sports variety
    "Wimbledon: Carlos Alcaraz defeats Jannik Sinner in five-set final",
    "FC Barcelona signs 17-year-old wonderkid for record €200 million",
    "Boston Red Sox trade franchise player to Yankees in stunning deal",
    "India wins Cricket World Cup final at Lord's Cricket Ground",
    "Olympic 100m record broken at Paris Games with time of 9.79 seconds",
    "Conor McGregor announces comeback fight against current UFC champion",
    "Tour de France 2024: Tadej Pogacar dominates to win for third time",
    "New York Knicks reach NBA Finals for first time since 2000",
    "England wins Euro 2024 on home soil beating Spain in the final",
    "Australian Open final goes to fifth set in six-hour epic match",

    # More entertainment
    "New season of Stranger Things breaks Netflix record for opening weekend",
    "Leonardo DiCaprio finally wins second Academy Award for Best Actor",
    "Michael Jackson hologram tour announced for 2025 across 50 cities",
    "The next James Bond actor officially revealed by producers",
    "Avatar 3 becomes highest-grossing film of all time at $3.5 billion",
    "Glastonbury headliners announced: Coldplay, Dua Lipa, and Arctic Monkeys",
    "Disney Plus raises subscription price to $13.99 per month",
    "Friends reunion special attracts 30 million viewers in one day",
    "New Beatles song released using AI to restore John Lennon vocals",
    "Keanu Reeves confirms John Wick 5 is officially in production",
]


def normalise_tweet(text):
    text = str(text)
    text = re.sub(r"http\S+|www\S+", "HTTPURL", text)
    text = re.sub(r"@\w+", "@USER", text)
    return re.sub(r"\s+", " ", text).strip()


def inject(output_path="data/new_scraped/drift_injection.csv"):
    os.makedirs("data/new_scraped", exist_ok=True)
    df = pd.DataFrame({
        "text":      OOD_TWEETS,
        "source":    "drift_injection",
        "domain":    "OOD",
        "timestamp": datetime.utcnow().isoformat(),
    })
    df.to_csv(output_path, index=False)
    print(f"Injected {len(df)} OOD tweets → {output_path}")
    return output_path


def run_monitor():
    """Import and run the drift monitor against the injected tweets."""
    monitor_path = os.path.join(PROJECT_ROOT, "src", "monitor.py")
    if not os.path.exists(monitor_path):
        print("monitor.py not found — skipping detector run.")
        return

    # Run monitor as a subprocess so it picks up the new CSV
    import subprocess
    result = subprocess.run(
        [sys.executable, monitor_path],
        capture_output=False,
        text=True,
    )
    return result.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inject-only",  action="store_true")
    parser.add_argument("--detect-only",  action="store_true")
    args = parser.parse_args()

    if not args.detect_only:
        inject()

    if not args.inject_only:
        print("\nRunning drift detector...")
        run_monitor()

        # Print summary from drift_report.json if available
        report_path = "metrics/drift_report.json"
        if os.path.exists(report_path):
            import json
            with open(report_path) as f:
                report = json.load(f)
            print("\n── Drift Report ──────────────────────────────")
            print(f"  PSI:           {report.get('psi', 'N/A')}")
            print(f"  PSI status:    {report.get('psi_status', 'N/A')}")
            print(f"  KS statistic:  {report.get('ks_statistic', 'N/A')}")
            print(f"  KS p-value:    {report.get('ks_p_value', 'N/A')}")
            print(f"  Drift flag:    {report.get('drift_detected', 'N/A')}")
            print(f"  Avg confidence:{report.get('avg_confidence', 'N/A')}")
            print("──────────────────────────────────────────────")


if __name__ == "__main__":
    main()
