import pandas as pd
import os

ratios = pd.read_csv("./data/financial_ratios.csv")
sentiment = pd.read_csv("./data/news_sentiment.csv")

combined = pd.merge(ratios, sentiment[["Ticker", "Sentiment Score"]], on="Ticker", how="left")

combined["Sentiment Score"] = combined["Sentiment Score"].fillna(0)

combined["macro_stress"] = combined["Date"].apply(lambda d: 1 if int(d[:4]) >= 2023 else 0)

os.makedirs("./data", exist_ok=True)

combined.to_csv("./data/combined_data.csv", index=False)
print("saved")
