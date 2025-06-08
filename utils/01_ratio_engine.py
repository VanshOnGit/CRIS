import yfinance as yf
import pandas as pd
import os

def safe_div(x, y):
    return x / y if x is not None and y not in [None, 0] else None

def fetch_ratios_for_all_dates(ticker_symbol):
    ticker = yf.Ticker(ticker_symbol)
    results = []

    try:
        bs = ticker.balance_sheet.T
        is_ = ticker.financials.T
        cf = ticker.cashflow.T
        info = ticker.info
    except Exception as e:
        print(f" Error fetching data for {ticker_symbol}: {e}")
        return []

    for date in bs.index:
        try:
            current_assets = bs.loc[date].get("Total Current Assets")
            current_liabilities = bs.loc[date].get("Total Current Liabilities")
            inventory = bs.loc[date].get("Inventory", 0)
            total_assets = bs.loc[date].get("Total Assets")
            total_liabilities = bs.loc[date].get("Total Liabilities")
            shareholder_equity = bs.loc[date].get("Total Stockholder Equity")
            total_debt = bs.loc[date].get("Short Long Term Debt Total", 0)
            retained_earnings = bs.loc[date].get("Retained Earnings")

            net_income = is_.loc[date].get("Net Income")
            ebit = is_.loc[date].get("Ebit")
            interest_expense = is_.loc[date].get("Interest Expense")
            sales = is_.loc[date].get("Total Revenue")

            market_cap = info.get("marketCap")

            current_ratio = safe_div(current_assets, current_liabilities)
            quick_ratio = safe_div(current_assets - inventory if current_assets and inventory else None, current_liabilities)
            debt_equity = safe_div(total_debt, shareholder_equity)
            roa= safe_div(net_income, total_assets)
            roe= safe_div(net_income, shareholder_equity)
            interest_coverage =safe_div(ebit, abs(interest_expense) if interest_expense else None)

            x1 = safe_div(current_assets - current_liabilities if current_assets and current_liabilities else None, total_assets)
            x2 = safe_div(retained_earnings, total_assets)
            x3 = safe_div(ebit, total_assets)
            x4 = safe_div(market_cap, total_liabilities)
            x5 = safe_div(sales, total_assets)

            results.append({
                "Ticker": ticker_symbol,
                "Date": str(date.date()),
                "Current Ratio": current_ratio,
                "Quick Ratio": quick_ratio,
                "D/E": debt_equity,
                "ROA": roa,
                "ROE": roe,
                "Interest Coverage": interest_coverage,
                "X1": x1,
                "X2": x2,
                "X3": x3,
                "X4": x4,
                "X5": x5
            })

        except Exception as e:
            print(f"Error processing {ticker_symbol} on {date}: {e}")
            continue

    return results


if __name__ == "__main__":
    tickers = [  
        "TCS.NS", "INFY.NS", "HDFCBANK.NS", "RELIANCE.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS",
        "AXISBANK.NS", "TECHM.NS", "WIPRO.NS", "ITC.NS", "LT.NS", "HCLTECH.NS", "BAJFINANCE.NS",
        "MARUTI.NS", "ULTRACEMCO.NS", "ASIANPAINT.NS", "HINDUNILVR.NS", "BHARTIARTL.NS", "COALINDIA.NS",
        "JSWSTEEL.NS", "POWERGRID.NS", "NTPC.NS", "ADANIENT.NS", "ADANIPORTS.NS", "BPCL.NS", "ONGC.NS",
        "DRREDDY.NS", "DIVISLAB.NS", "SUNPHARMA.NS", "TATACONSUM.NS", "BRITANNIA.NS", "EICHERMOT.NS",
        "HDFCLIFE.NS", "BAJAJFINSV.NS", "HEROMOTOCO.NS", "GRASIM.NS", "TITAN.NS", "BAJAJ-AUTO.NS",
        "CIPLA.NS", "UPL.NS", "INDUSINDBK.NS", "M&M.NS", "TATAMOTORS.NS", "SBICARD.NS", "AMBUJACEM.NS",
        "SHREECEM.NS", "DABUR.NS", "PIDILITIND.NS", "GAIL.NS", "ABB.NS", "LUPIN.NS", "ICICIPRULI.NS",
        "IDFCFIRSTB.NS", "AUROPHARMA.NS", "MPHASIS.NS", "NAUKRI.NS", "BANDHANBNK.NS", "PEL.NS",
        "MUTHOOTFIN.NS", "APOLLOHOSP.NS", "HAVELLS.NS", "GLAND.NS", "DMART.NS", "GODREJCP.NS",
        "CHOLAFIN.NS", "BIOCON.NS", "TORNTPHARM.NS", "BOSCHLTD.NS", "SRF.NS", "CANBK.NS", "TRENT.NS",
        "INDIGO.NS", "PAGEIND.NS", "SAIL.NS", "TVSMOTOR.NS", "OBEROIRLTY.NS", "IEX.NS", "CROMPTON.NS",
        "BANKBARODA.NS", "YESBANK.NS", "ZEEL.NS", "FEDERALBNK.NS", "ASHOKLEY.NS", "BHEL.NS",
        "PVRINOX.NS", "RECLTD.NS", "MFSL.NS", "BEL.NS", "BALRAMCHIN.NS", "IDBI.NS", "PNB.NS",
        "IRCTC.NS", "TATAPOWER.NS", "IRFC.NS", "NHPC.NS", "IOC.NS", "LICI.NS", "JSPL.NS", "INDHOTEL.NS"
    ]

    all_ratios = []

    for t in tickers:
        print(f"\n Fetching: {t}")
        ratios = fetch_ratios_for_all_dates(t)
        if not ratios:
            print(f" No data for {t}")
        else:
            print(f" Got {len(ratios)} rows for {t}")
        all_ratios.extend(ratios)

    if not all_ratios:
        print(" No ratios collected. CSV not written.")
    else:
        df = pd.DataFrame(all_ratios)
        os.makedirs("./data", exist_ok=True)
        df.to_csv("./data/financial_ratios.csv", index=False)
        print("saved financial_ratios.csv with", len(df), "rows")
