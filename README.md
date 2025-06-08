# CRIS – Corporate Risk Intelligence System

**Deployed App:** [https://cris-vansh.streamlit.app/](https://cris-vansh.streamlit.app/)


CRIS (Corporate Risk Intelligence System) is an AI-driven financial risk analysis tool that functions as an early warning system for corporate distress. It combines multi-source data — including financial ratios, news sentiment, and macroeconomic stress indicators — to produce a unified risk score for companies. Built using machine learning with SHAP explainability, CRIS helps analysts and stakeholders identify vulnerable firms and understand the key drivers behind their risk profile.

---

## Key Features

- Financial Ratio Engine  
  Automatically calculates liquidity, solvency, and Altman Z-score components using public financial data.

- News Sentiment Analyzer  
  Extracts company-specific sentiment from news articles using FinBERT to quantify short-term market perception.

- Macro Stress Flags  
  Integrates macroeconomic shocks (e.g., interest rate hikes, banking crises) to contextualize firm-level risk.

- Risk Prediction Model  
  Uses XGBoost to assign a probabilistic risk score, identifying companies at high risk of distress.

- SHAP Explainability  
  Highlights the most influential financial or sentiment drivers behind each company’s risk score.

- Historic Collapse Backtesting  
  Validated using real-world case studies (e.g., SVB, Yes Bank, Evergrande).

- Streamlit Dashboard  
  A simple and intuitive UI that summarizes risk scores, feature importance, and recent macro events.

---

## Tech Stack

- Python, Pandas, NumPy, Scikit-learn, XGBoost, SHAP  
- FinBERT (via Hugging Face) for sentiment analysis  
- Streamlit for web-based dashboard  

---

## Use Case

This project simulates what an internal risk monitoring tool could look like — useful for analysts, auditors, regulators, or FinTech platforms needing real-time corporate risk signals.

---

## About Me

I’m Vansh Kumar, a third-year B.Tech student in Artificial Intelligence at IIT Gandhinagar.  
LinkedIn: [https://www.linkedin.com/in/vansh-ai/](https://www.linkedin.com/in/vansh-ai/)
