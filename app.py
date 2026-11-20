
import streamlit as st
import pandas as pd
import requests
from io import StringIO

st.set_page_config(
    page_title="Catcher Lab",
    page_icon="⚾",
    layout="wide"
)

st.title("CATCHER LAB")
st.caption("MLB Catching Analytics")

DATASETS = {
    "Receiving": "catcher-framing",
    "Blocking": "catcher-blocking",
    "Throwing": "catcher-throwing",
    "Pop Time": "poptime"
}

@st.cache_data(ttl=21600)
def load_data(category, season):
    endpoint = DATASETS[category]
    url = (
        "https://baseballsavant.mlb.com/"
        f"leaderboard/{endpoint}"
    )

    response = requests.get(
        url,
        params={
            "year": season,
            "csv": "true"
        },
        timeout=30
    )

    response.raise_for_status()

    return pd.read_csv(
        StringIO(response.text)
    )

st.sidebar.header("Filters")

season = st.sidebar.selectbox(
    "Season",
    list(range(2026, 2017, -1)),
    index=1
)

category = st.sidebar.radio(
    "Category",
    list(DATASETS.keys())
)

search = st.sidebar.text_input(
    "Search player"
)

st.header(f"{season} MLB {category}")

try:
    with st.spinner("Loading Statcast data..."):
        df = load_data(category, season)

    if search:
        mask = df.astype(str).apply(
            lambda col: col.str.contains(
                search,
                case=False,
                regex=False
            )
        ).any(axis=1)

        df = df[mask]

    st.metric("Records", len(df))

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "Download CSV",
        df.to_csv(index=False),
        file_name=f"{category}_{season}.csv",
        mime="text/csv"
    )

except Exception as e:
    st.error(f"Unable to retrieve data: {e}")

st.caption("Data source: MLB Baseball Savant")
