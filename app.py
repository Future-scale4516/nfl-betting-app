import streamlit as st
import nflreadpy as nfl

st.title("NFL Betting Analysis")

@st.cache_data(ttl=3600)
def get_schedule(season):
    return nfl.load_schedules(seasons=[season]).to_pandas()

schedule = get_schedule(2026)

st.write(f"Games loaded: {len(schedule)}")
st.dataframe(schedule)