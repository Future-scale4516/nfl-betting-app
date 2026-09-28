import streamlit as st
import nflreadpy as nfl

st.title("NFL Betting Analysis")

@st.cache_data(ttl=3600)
def get_schedule(season):
    return nfl.load_schedules(seasons=[season]).to_pandas()

schedule = get_schedule(2026)

reg = schedule[schedule["game_type"] == "REG"]

unplayed = reg[reg["home_score"].isna()]
current_week = int(unplayed["week"].min()) if len(unplayed) else int(reg["week"].max())

week = st.selectbox("Week", sorted(reg["week"].unique()), index=current_week - 1)

cols = ["gameday", "gametime", "away_team", "home_team",
        "spread_line", "total_line", "away_score", "home_score"]
st.dataframe(reg[reg["week"] == week][cols], hide_index=True)