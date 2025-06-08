import pandas as pd
import os

df = pd.read_csv("./data/combined_data.csv")
df["Date"] = pd.to_datetime(df["Date"])

macro_events = [
    "2020-03-11",  # COVID declared pandemic
    "2022-02-24",  # Russia-Ukraine war
    "2022-06-15",  # Fed rate hike
    "2023-03-10",  # SVB collapse
    "2023-10-07",  # Israel-Gaza escalation
]

macro_years = [pd.to_datetime(d).year for d in macro_events]

df["macro_stress"] = df["Date"].dt.year.isin(macro_years).astype(int)

df.to_csv("./data/combined_data.csv", index=False)
print("Updated combined_data.csv with macro_stress column.")
