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

@st.cache_data(ttl=3600)
def get_team_epa(season):
    pbp = nfl.load_pbp(seasons=[season]).select(
        ["season_type", "week", "posteam", "defteam", "play_type", "epa"]
    ).to_pandas()

    plays = pbp[
        (pbp["season_type"] == "REG")
        & pbp["play_type"].isin(["pass", "run"])
        & pbp["epa"].notna()
    ]

    offense = plays.groupby("posteam")["epa"].agg(["mean", "count"])
    offense.columns = ["off_epa", "off_plays"]
    defense = plays.groupby("defteam")["epa"].mean().rename("def_epa")

    teams = offense.join(defense)
    teams["net_epa"] = teams["off_epa"] - teams["def_epa"]
    teams = teams.reset_index().rename(columns={"posteam": "team"})
    return teams.sort_values("net_epa", ascending=False)

st.subheader("Team Ratings (EPA per play)")
team_epa = get_team_epa(2026)
st.dataframe(team_epa.round(3), hide_index=True)