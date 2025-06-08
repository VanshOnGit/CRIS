import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np
import urllib.parse
import numpy as np
import os
import IPython


st.set_page_config(page_title="CRIS – Risk Intelligence Dashboard", page_icon="📊")

with open("last_update.txt", "r", encoding="utf-16") as f:
    last_update_text = f.read().strip()


# --- Load data and model ---
@st.cache_data
def load_data():
    return pd.read_csv(os.path.join(os.path.dirname(__file__), "combined_data.csv"))


@st.cache_resource
def load_model():
    return joblib.load(os.path.join(os.path.dirname(__file__), "cris_model_78.pkl"))


df = load_data()
model = load_model()

# --- Logo & Title ---
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image(os.path.join(os.path.dirname(__file__), "logo.png"), width=520)

st.markdown("<h1 style='white-space: nowrap;'>Corporate Risk Intelligence System (CRIS)</h1>", unsafe_allow_html=True)

st.markdown("""
Welcome to **CRIS**, a machine-learning based early warning system  
to detect **financial distress risk** for Indian companies using:

- Financial Ratios  
- News Sentiment  
- Macro Stress Indicators              
""")

st.markdown(
    f"""
    <style>
    .footer-update {{
        position: fixed;
        bottom: 8px;
        left: 12px;
        font-size: 0.75rem;
        color: #999999;
    }}
    </style>
    <div class='footer-update'>🕒 {last_update_text}</div>
    """,
    unsafe_allow_html=True
)


# --- Select company ---
ticker_map = {t: t.replace(".NS", "") for t in sorted(df["Ticker"].unique())}

st.markdown("### 👇 Please select a company from the dropdown to begin")
selected_clean = st.selectbox("Company", ["-- Select --"] + list(ticker_map.values()))

if selected_clean == "-- Select --":
    st.markdown("""
    <style>
    .fixed-footer {
        position: fixed;
        bottom: 10px;
        right: 15px;
        background-color: rgba(33, 33, 33, 0.9);
        padding: 10px 15px;
        border-radius: 8px;
        font-size: 0.85em;
        color: white;
        z-index: 999;
    }
    .fixed-footer a {
        color: #00BFFF;
        text-decoration: none;
    }
    </style>

    <div class="fixed-footer">
        👤 <b>Vansh Kumar</b><br>
        🎓 IIT Gandhinagar<br>
        💼 AI & Data Science Intern @ Tata Communications <br>           
        📧 <a href="mailto:kumar.vansh@iitgn.ac.in">kumar.vansh@iitgn.ac.in</a><br>
        🔗 <a href="https://www.linkedin.com/in/vansh-ai/" target="_blank">LinkedIn</a> |
        💻 <a href="https://github.com/VanshOnGit" target="_blank">GitHub</a> <br>
        📝    <a href = "https://forms.gle/MQrrKBxwdFUgMc9g9" target="_blank">Give your Feedback or Suggestions</a>
                
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
<a href="https://forms.gle/MQrrKBxwdFUgMc9g9" target="_blank">
    <button style='padding:10px 20px; font-size:16px; border:none; background-color:#4CAF50; color:white; border-radius:8px; cursor:pointer;'>
        📝 Give Feedback
    </button>
</a>
""", unsafe_allow_html=True)


    st.stop()


selected_ticker = [k for k, v in ticker_map.items() if v == selected_clean][0]

search_query = f"{selected_clean} company"
search_url = f"https://www.google.com/search?q={urllib.parse.quote_plus(search_query)}"
st.markdown(f"[🔍 Search {selected_clean} on Google]({search_url})", unsafe_allow_html=True)

if selected_ticker == "-- Select --":
    st.warning("👆 Please select a company to begin analysis.")
    st.stop()


company_data = df[df["Ticker"] == selected_ticker].sort_values(by="Date", ascending=False)
latest = company_data.iloc[0]

abc = str(selected_ticker)[:-3]

st.title(f"📊 CRIS Dashboard for {abc}")


# --- Financials ---
st.subheader("Financial Indicators")
latest_3yrs = company_data.sort_values("Date", ascending=False).head(3)

financial_cols = ["Date", "Current Ratio", "Quick Ratio", "D/E", "ROA", "ROE", "Interest Coverage"]
financial_data = latest_3yrs[financial_cols]

financial_data = financial_data.replace("None", np.nan)
financial_data = financial_data.dropna(how='all')
financial_data = financial_data.dropna(axis=1, how='all')
financial_data = financial_data.rename(columns={"ROA": "Return on Assets (ROA)"})

st.dataframe(financial_data.set_index("Date"))

with st.expander("📘 What does 'Return on Assets (ROA) mean?"):
    st.markdown("""
**Return on Assets (ROA)** tells how well a company is using its total assets (like buildings, money, machines) to make profit.

- A **higher ROA** means the company is good at using its resources to earn money.
- A **lower ROA** might mean the company owns a lot of assets but isn’t earning much from them.

For Eg: Think of it like this:  
If a company has ₹100 worth of stuff (assets), and it earns ₹10 profit, ROA = 10% or 0.1.

It's a simple way to ask:  
**"How much money are we making from what we already have?"**
""")



# --- Sentiment & Macro ---
st.subheader("🗞️ News Sentiment & Macro Stress")
col1, col2 = st.columns(2)
col1.metric("Sentiment Score", round(latest["Sentiment Score"], 3))
macro_value = latest["macro_stress"]
col2.metric("Macro Stress", "Yes" if macro_value == 1 else "No")

with st.expander("📘 What do Sentiment Score & Macro Stress mean?"):
    st.markdown("""
**Sentiment Score**  
This tells us what the **news and media** are saying about the company.  
- A **positive score** (closer to +1) means good news.  
- A **negative score** (closer to -1) means bad news.  
- Around **0** means mixed or neutral.

---

**Macro Stress**  
This shows whether the company is affected by **big-picture economic problems**, like:  
- High inflation  
- Rising interest rates  
- Slow GDP growth  
- Global financial uncertainty

- If it says **"Yes"**, the company is operating under stress from the economy.
- If it says **"No"**, there’s no major external pressure from the macro environment.

""")

# --- Risk Score ---
st.subheader("Financial Distress Risk")

feature_cols = [
    "Current Ratio", "Quick Ratio", "D/E", "ROA", "ROE", "Interest Coverage",
    "X1", "X2", "X3", "X4", "X5", "Sentiment Score"
]

input_features = latest[feature_cols]
risk_score = model.predict_proba([input_features])[0][1]

if risk_score > 0.7:
    st.error(f"Risk Score: {risk_score:.3f} -----> ⚠️ High Risk")
elif risk_score > 0.4:
    st.warning(f"Risk Score: {risk_score:.3f} -----> ⚠️ Moderate Risk")
else:
    st.success(f"Risk Score: {risk_score:.3f} -----> ✅ Low Risk")

with st.expander("📘 What does Risk Score mean?"):
    st.markdown("""
**Risk Score** shows how likely the company is to fall into financial distress, like running out of money, defaulting on loans, or struggling to survive.
                
The Risk Score is based on many features together....not just Macro Stress
This number is predicted by a machine learning model using many inputs:
- Financial ratios (like ROA, D/E)
- News sentiment
- Macro stress signals
- Other internal features (X1–X5)

---

- **Above 0.7** → ⚠️ High risk (the company may be in trouble soon)  
- **0.4 – 0.7** → ⚠️ Moderate risk  
- **Below 0.4** → ✅ Low risk (healthy and stable)

""")


# --- SHAP explanation ---
st.subheader("SHAP Feature Impact")

X_df = pd.DataFrame([input_features])
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_df)
shap_df = pd.DataFrame(shap_values, columns=X_df.columns)
mean_shap = shap_df.abs().mean()
nonzero_features = mean_shap[mean_shap > 0].index.tolist()
shap_filtered = shap_df[nonzero_features]
X_filtered = X_df[nonzero_features]


IPython.get_ipython = lambda: True  
shap.initjs()

shap.summary_plot(shap_filtered.values, X_filtered, plot_type="bar", show=False)
fig = plt.gcf()
st.pyplot(fig)

with st.expander("📘 What does this SHAP chart show?"):
    st.markdown("""
**SHAP** stands for **SHapley Additive exPlanations**.  
It shows **which features had the biggest effect** on the company's risk score.

> 💭 "What made the model think this company is risky (or not)?"

---

### How to read the chart:
- Each bar is a feature (like ROA, X2, Sentiment Score)
- The longer the bar, the **more that feature influenced** the prediction (risk score)
- SHAP does **not** show the feature’s actual value — only its **impact**
- If a feature doesn’t appear, it means it had **no measurable effect**

---

### What do these features mean?

- **X5, X2**: These are **internal or hidden features** from the dataset that capture financial patterns or behaviors not named directly (like custom ratios, trends, or composite indicators). Their exact meaning may be known only to the dataset creator.

- **ROA (Return on Assets)**: Shows how efficiently the company uses its assets to make profit. Higher ROA = more efficient.

- **Sentiment Score**: Reflects the tone of news and media about the company. Positive score = good news, negative = bad news.

---
""")


st.subheader("Companies with No Macro Stress (latest year only):")
latest_rows = df.sort_values("Date", ascending=False).drop_duplicates("Ticker", keep="first")
macro_stress_free = latest_rows[latest_rows["macro_stress"] == 0]["Ticker"].unique()
st.dataframe(pd.DataFrame(macro_stress_free, columns=["Ticker"]))

if len(macro_stress_free) == 0:
    st.write("None")

st.markdown("""
<div style='line-height: 1.6'>
🛠️ <i>Made with care by</i> <b>Vansh Kumar</b><br>
🎓 <i>B.Tech AI, IIT Gandhinagar</i><br>
💼 <i>AI & Data Science Intern @ Tata Communications</i>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.fixed-footer {
    position: fixed;
    bottom: 10px;
    right: 15px;
    background-color: rgba(33, 33, 33, 0.9);
    padding: 10px 15px;
    border-radius: 8px;
    font-size: 0.85em;
    color: white;
    z-index: 999;
}
.fixed-footer a {
    color: #00BFFF;
    text-decoration: none;
}
</style>

            

<div class="fixed-footer">
    👤 <b>Vansh Kumar</b><br>
    🎓 IIT Gandhinagar<br>
    💼 AI & Data Science Intern @ Tata Communications <br>        
    📧 <a href="mailto:kumar.vansh@iitgn.ac.in">kumar.vansh@iitgn.ac.in</a><br>
    🔗 <a href="https://www.linkedin.com/in/vansh-ai/" target="_blank">LinkedIn</a> | 💻 <a href="https://github.com/VanshOnGit" target="_blank">GitHub</a> <br>
        📝    <a href = "https://forms.gle/MQrrKBxwdFUgMc9g9" target="_blank">Give your Feedback or Suggestions</a>
</div>
""", unsafe_allow_html=True)
st.markdown("""
<a href="https://forms.gle/MQrrKBxwdFUgMc9g9" target="_blank">
    <button style='padding:10px 20px; font-size:16px; border:none; background-color:#4CAF50; color:white; border-radius:8px; cursor:pointer;'>
        📝 Give Feedback
    </button>
</a>
""", unsafe_allow_html=True)

# --- Show last update info bottom-left ---
st.markdown(
    f"""
    <style>
    .footer-update {{
        position: fixed;
        bottom: 8px;
        left: 12px;
        font-size: 0.75rem;
        color: #999999;
    }}
    </style>
    <div class='footer-update'>🕒 {last_update_text}</div>
    """,
    unsafe_allow_html=True
)
