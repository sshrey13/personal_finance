import streamlit as st
import pandas as pd
from functions import (
    get_scheme_metadata,
    process_scheme_metadata,
    get_historical_nav,
    summarize_returns
)

st.set_page_config(page_title="Mutual Fund Summary App", layout="wide")
st.title("📈 Mutual Fund Summary Explorer")

# --- Sidebar
st.sidebar.header("🔍 Filters")
scheme_input = st.sidebar.text_input("Scheme name(s) (comma-separated):", placeholder="e.g. Parag Parikh, Quant ELSS")
min_age = st.sidebar.slider("Minimum scheme age (years)", 0, 30, 0)
scheme_names = [s.strip() for s in scheme_input.split(",") if s.strip()] if scheme_input else None

# --- Fetch data
st.info("⏳ Fetching scheme metadata...")
raw_data = get_scheme_metadata(scheme_name=scheme_names)
df_schemes = process_scheme_metadata(raw_data, active_only="Y", isGrowthOnly="Y")
df_schemes["historical_url"] = df_schemes["schemeCode"].apply(lambda x: f"https://api.mfapi.in/mf/{x}")

# --- Main
if not df_schemes.empty:
    st.info("📥 Fetching NAV history...")
    df_history = get_historical_nav(df_schemes, scheme_name=scheme_names, max_workers=10)

    if df_history.empty:
        st.warning("⚠️ No NAV data found.")
    else:
        st.success(f"✅ NAVs for {df_history['scheme_code'].nunique()} schemes.")
        summary_df = summarize_returns(df_history)

        if min_age > 0:
            summary_df = summary_df[summary_df["scheme_age_years"] >= min_age]

        st.write("🎯 Summary:")
        st.dataframe(summary_df)

        st.download_button("📥 Download as CSV", summary_df.to_csv(index=False), "summary.csv", "text/csv")
else:
    st.warning("⚠️ No matching schemes found.")
