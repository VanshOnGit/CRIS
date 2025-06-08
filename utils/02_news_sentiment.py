import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
from newsapi import NewsApiClient
from transformers import pipeline
from transformers import pipeline
import pandas as pd

newsapi = NewsApiClient(api_key="66a232e334f04f4c80445a3d589a4b2f")
finbert = pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone", framework="pt")

tickers = pd.read_csv("./data/financial_ratios.csv")["Ticker"].unique()

def ticker_to_company(ticker):
    name = ticker.replace(".NS", "")
    name = name.replace("BANK", " Bank")
    name = name.replace("FINANCE", " Finance")
    name = name.replace("TECH", " Tech")
    name = name.replace("HIND", "Hindustan ")
    name = name.upper()
    return name

results = []

def label_to_score(label):
    return {"positive": 1, "neutral": 0, "negative": -1}[label.lower()]

for ticker in tickers:
    company_name = ticker_to_company(ticker)
    print(f"Fetching news for {company_name}")

    try:
        headlines = newsapi.get_everything(
            q=company_name,
            language="en",
            sort_by="relevancy",
            page_size=20
        )["articles"]
    except Exception as e:
        print(f"Error for {company_name}: {e}")
        continue

    titles = [a["title"] for a in headlines if a["title"]]

    if not titles:
        continue

    try:
        sentiments = finbert(titles)
        scores = [label_to_score(s['label']) for s in sentiments]
        avg_score = round(sum(scores) / len(scores), 3)

        results.append({
            "Ticker": ticker,
            "Company": company_name,
            "Sentiment Score": avg_score,
            "Articles Analyzed": len(scores)
        })

    except Exception as e:
        print(f" Sentiment error for {company_name}: {e}")
        continue

df = pd.DataFrame(results)
os.makedirs("./data", exist_ok=True)
df.to_csv("./data/news_sentiment.csv", index=False)
print("saved")
