import pandas as pd
import numpy as np
from tqdm import tqdm
import requests
from time import sleep
from concurrent.futures import ThreadPoolExecutor
import re

# -------------------------------
# 1. Fetch all scheme metadata
# -------------------------------

def get_scheme_metadata(url="https://api.mfapi.in/mf", scheme_name=None, verbose=True):
    """
    Fetch all mutual fund scheme metadata from MFAPI, with optional case-insensitive filtering by scheme name(s).

    Parameters:
        url : API endpoint
        scheme_name : str or list of str, optional
        verbose : bool, whether to print match summary

    Returns:
        List of filtered scheme metadata dictionaries
    """
    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            print("❌ Failed to fetch scheme list.")
            return []

        data = response.json()

        # If no filter is provided, return all
        if not scheme_name:
            if verbose:
                print(f"✅ Fetched all schemes: {len(data)}")
            return data

        # Handle single or multiple names
        if isinstance(scheme_name, str):
            scheme_name = [scheme_name]

        # Compile case-insensitive match pattern
        pattern = "|".join([re.escape(name.lower()) for name in scheme_name])

        # Convert to DataFrame for fast filtering
        df = pd.DataFrame(data)
        mask = df["schemeName"].str.lower().str.contains(pattern, regex=True)

        filtered = df[mask]
        if verbose:
            print(f"🔍 Found {len(filtered)} matching scheme(s) for: {scheme_name}")

        return filtered.to_dict(orient="records")

    except Exception as e:
        print(f"🚨 Exception in get_scheme_metadata: {e}")
        return []

# -------------------------------
# 2. Process and enrich metadata
# -------------------------------
def process_scheme_metadata(raw_data, active_only="Y", isGrowthOnly="Y", isDirect="N"):
    df = pd.DataFrame(raw_data)

    df["isinGrowth"] = df["isinGrowth"].fillna("").astype(str).str.strip()
    df["isinDivReinvestment"] = df["isinDivReinvestment"].fillna("").astype(str).str.strip()

    df["isActive"] = np.where((df["isinGrowth"].str.len() > 0) | (df["isinDivReinvestment"].str.len() > 0), True, False)
    df["isGrowthOnly"] = (df["isinGrowth"].str.len() > 0) & (df["isinDivReinvestment"].str.len() == 0)
    df["isDividendOnly"] = (df["isinDivReinvestment"].str.len() > 0) & (df["isinGrowth"].str.len() == 0)

    # Optimize: Don't compute 'isDirect' unless asked
    df["isDirect"] = False
    if isDirect.upper() == "Y":
        df["isDirect"] = df["schemeName"].str.contains("direct", case=False, na=False)

    if active_only.upper() == "Y":
        df = df[df["isActive"]]
        print(f"🔍 Active schemes found: {len(df)}")

    if isGrowthOnly.upper() == "Y":
        df = df[df["isGrowthOnly"]]
        print(f"🌱 Growth-only schemes found: {len(df)}")

    if isDirect.upper() == "Y":
        df = df[df["isDirect"]]
        print(f"📦 Direct-only schemes found: {len(df)}")

    df["latest_url"] = df["schemeCode"].apply(lambda x: f"https://api.mfapi.in/mf/{x}/latest")
    df["historical_url"] = df["schemeCode"].apply(lambda x: f"https://api.mfapi.in/mf/{x}")

    return df.reset_index(drop=True)

# -------------------------------
# 3. Fetch latest NAV (parallel)
# -------------------------------
def fetch_nav(url, verbose=True):
    try:
        r = requests.get(url, timeout=5)
        if r.status_code != 200:
            if verbose: print(f"❌ HTTP {r.status_code} for {url}")
            return None
        data = r.json()
        if data.get("status") != "SUCCESS":
            return None
        latest_list = data.get("data", [])
        if not latest_list: return None

        latest = latest_list[0]
        if "nav" not in latest or "date" not in latest:
            return None

        meta = data.get("meta", {})
        row = {**meta, **latest}
        row["nav"] = float(row["nav"])
        row["date"] = pd.to_datetime(row["date"], format="%d-%m-%Y")
        return row

    except Exception as e:
        if verbose: print(f"🚨 Exception in fetch_nav: {e} → {url}")
        return None

def get_latest_nav(df_schemes, limit=10, delay=0.2, verbose=True, scheme_name=None, max_workers=10):
    """
    Fetch latest NAVs with optional scheme_name filtering (case-insensitive).
    """
    # Filter schemes if name(s) provided
    if scheme_name:
        if isinstance(scheme_name, str):
            scheme_name = [scheme_name]

        # ✅ Keep case-insensitive user-facing string match using regex
        pattern = "|".join([re.escape(name.lower()) for name in scheme_name])
        mask = df_schemes["schemeName"].str.lower().str.contains(pattern, regex=True)
        filtered_df = df_schemes[mask]

        if filtered_df.empty:
            print(f"❌ No matching schemes found for: {scheme_name}")
            return pd.DataFrame()

        urls = filtered_df["latest_url"]
        print(f"🔍 Found {len(urls)} matching scheme(s)")
    else:
        urls = df_schemes["latest_url"].head(limit)

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for result in tqdm(executor.map(lambda url: fetch_nav(url, verbose), urls), total=len(urls)):
            if result:
                results.append(result)

    df_latest = pd.DataFrame(results)
    print(f"✅ Latest NAVs fetched: {len(df_latest)}")
    return df_latest

# -------------------------------
# 4. Fetch historical NAV (parallel)
# -------------------------------
def fetch_history(url, verbose=True):
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return pd.DataFrame()
        data = r.json()
        if data.get("status") != "SUCCESS":
            return pd.DataFrame()

        nav_data = pd.DataFrame(data["data"])
        if nav_data.empty:
            return pd.DataFrame()

        nav_data["nav"] = nav_data["nav"].astype(float)
        nav_data["date"] = pd.to_datetime(nav_data["date"], format="%d-%m-%Y")

        for k, v in data["meta"].items():
            nav_data[k] = v
        return nav_data

    except Exception as e:
        if verbose: print(f"🚨 Exception in fetch_history: {e}")
        return pd.DataFrame()

def get_historical_nav(df_schemes, limit=5, delay=0.3, verbose=True, scheme_name=None, max_workers=5):
    """
    Fetch historical NAVs with optional case-insensitive scheme_name filtering.
    """
    if scheme_name:
        if isinstance(scheme_name, str):
            scheme_name = [scheme_name]
        pattern = "|".join([re.escape(name.lower()) for name in scheme_name])
        mask = df_schemes["schemeName"].str.lower().str.contains(pattern, regex=True)
        urls = df_schemes[mask]["historical_url"]

        if urls.empty:
            print(f"❌ No matching schemes found for: {scheme_name}")
            return pd.DataFrame()
        print(f"🔍 Found {len(urls)} matching scheme(s)")
    else:
        urls = df_schemes["historical_url"].head(limit)

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for df in tqdm(executor.map(lambda url: fetch_history(url, verbose), urls), total=len(urls)):
            if not df.empty:
                results.append(df)

    if results:
        df_history = pd.concat(results, ignore_index=True)
        print(f"✅ Historical NAV rows fetched: {len(df_history)}")
        return df_history
    else:
        print("⚠️ No historical NAVs found.")
        return pd.DataFrame()

def calculate_cagr(start_value, end_value, num_years):
    if start_value <= 0 or end_value <= 0 or num_years <= 0:
        return None
    return (end_value / start_value) ** (1 / num_years) - 1

def summarize_returns(df_history, years_list=[1, 3, 5, 7, 10]):
    df_history = df_history.copy()
    df_history["date"] = pd.to_datetime(df_history["date"])
    summaries = []

    grouped = df_history.groupby("scheme_code")

    for scheme_code, group in tqdm(grouped, desc="📊 Calculating CAGR and rolling returns"):
        group = group.sort_values("date")
        scheme_name = group["scheme_name"].iloc[0]
        dates = group["date"].values
        navs = group["nav"].values
        start_date = dates[0]
        end_date = dates[-1]
        scheme_age_years = (end_date - start_date).astype('timedelta64[D]').astype(int) / 365.25
        start_nav = navs[0]
        end_nav = navs[-1]

        scheme_summary = {
            "scheme_code": scheme_code,
            "scheme_name": scheme_name,
            "inception_date": pd.to_datetime(start_date).date(),
            "scheme_age_years": round(scheme_age_years, 2),
            "latest_date": pd.to_datetime(end_date).date(),
            "cagr_since_inception": calculate_cagr(start_nav, end_nav, (end_date - start_date).astype('timedelta64[D]').astype(int) / 365.25)
        }

        for y in years_list:
            # A. Point-in-time CAGR
            past_date = end_date - np.timedelta64(int(y * 365.25), 'D')
            idx = np.searchsorted(dates, past_date, side='right') - 1
            if idx < 0:
                scheme_summary[f"cagr_{y}yr"] = None
                scheme_summary[f"rolling_avg_{y}yr"] = None
                scheme_summary[f"rolling_min_{y}yr"] = None
                scheme_summary[f"rolling_max_{y}yr"] = None
                continue

            start_y_nav = navs[idx]
            scheme_summary[f"cagr_{y}yr"] = calculate_cagr(start_y_nav, end_nav, y)

            # B. Rolling CAGR
            rolling_cagrs = []
            for i in range(len(dates)):
                start_i = dates[i]
                end_i = start_i + np.timedelta64(int(y * 365.25), 'D')
                j = np.searchsorted(dates, end_i, side='right') - 1
                if j <= i or j >= len(dates):
                    continue
                nav_start = navs[i]
                nav_end = navs[j]
                rr = calculate_cagr(nav_start, nav_end, y)
                if rr is not None:
                    rolling_cagrs.append(rr)

            if rolling_cagrs:
                rolling_cagrs = np.array(rolling_cagrs)
                scheme_summary[f"rolling_avg_{y}yr"] = np.mean(rolling_cagrs)
                scheme_summary[f"rolling_min_{y}yr"] = np.min(rolling_cagrs)
                scheme_summary[f"rolling_max_{y}yr"] = np.max(rolling_cagrs)
            else:
                scheme_summary[f"rolling_avg_{y}yr"] = None
                scheme_summary[f"rolling_min_{y}yr"] = None
                scheme_summary[f"rolling_max_{y}yr"] = None

        summaries.append(scheme_summary)

    summary_df = pd.DataFrame(summaries)

    # ✅ Keep as float, no % sign
    percentage_cols = [col for col in summary_df.columns if "cagr" in col or "rolling" in col]
    summary_df[percentage_cols] = summary_df[percentage_cols] * 100
    summary_df[percentage_cols] = summary_df[percentage_cols].round(2)

    return summary_df
