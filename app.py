import math

import pandas as pd
import requests
import streamlit as st
import nflreadpy as nfl

st.title("NFL Betting Analysis")

SEASON = 2026
PRIOR_SEASON = 2025

# ---------- Model settings (starting values, tune via backtest) ----------
PLAYS_PER_GAME = 62   # offensive plays per team per game
HOME_FIELD = 1.5      # points; set to 0 for neutral-site games
MARGIN_SD = 13.5      # spread of real margins around the prediction
TOTAL_SD = 10.0       # spread of real totals around the prediction
BLEND_K = 400         # plays of 2026 data before it's 50/50 with 2025
REGRESS = 0.33        # how far 2025 ratings are pulled to league average

# Traffic lights per market: (green edge, amber edge, max plausible model %)
BANDS = {
    "Moneyline": (0.04, 0.02, 0.90),
    "Spread": (0.03, 0.015, 0.70),
    "Total": (0.03, 0.015, 0.70),
}

TEAM_ABBR = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF", "Carolina Panthers": "CAR", "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE", "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC", "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LA", "Miami Dolphins": "MIA", "Minnesota Vikings": "MIN",
    "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT",
    "Seattle Seahawks": "SEA", "San Francisco 49ers": "SF", "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN", "Washington Commanders": "WAS",
}


# ---------- Data ----------
@st.cache_data(ttl=3600)
def get_schedule(season):
    return nfl.load_schedules(seasons=[season]).to_pandas()


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


def blend_ratings(current, prior, k=BLEND_K, regress=REGRESS):
    merged = current.merge(
        prior[["team", "off_epa", "def_epa"]], on="team", suffixes=("_26", "_25")
    )
    w = merged["off_plays"] / (merged["off_plays"] + k)

    for col in ["off_epa", "def_epa"]:
        league_avg = merged[f"{col}_25"].mean()
        prior_reg = merged[f"{col}_25"] * (1 - regress) + league_avg * regress
        merged[col] = w * merged[f"{col}_26"] + (1 - w) * prior_reg

    merged["net_epa"] = merged["off_epa"] - merged["def_epa"]
    merged["weight_2026"] = w
    cols = ["team", "off_epa", "def_epa", "net_epa", "weight_2026"]
    return merged[cols].sort_values("net_epa", ascending=False)


def league_avg_total(schedules):
    s = pd.concat(schedules)
    s = s[(s["game_type"] == "REG") & s["home_score"].notna()]
    return float((s["home_score"] + s["away_score"]).mean())


# ---------- Model ----------
def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def predict_games(games, ratings, lg_total):
    r = ratings.set_index("team")
    lg_off = r["off_epa"].mean()
    lg_def = r["def_epa"].mean()

    rows = []
    for _, g in games.iterrows():
        h, a = g["home_team"], g["away_team"]
        if h not in r.index or a not in r.index:
            continue

        # Expected EPA per play for each offence against this defence
        home_exp = r.at[h, "off_epa"] + (r.at[a, "def_epa"] - lg_def)
        away_exp = r.at[a, "off_epa"] + (r.at[h, "def_epa"] - lg_def)

        hfa = 0 if g["location"] == "Neutral" else HOME_FIELD
        margin = PLAYS_PER_GAME * (home_exp - away_exp) + hfa
        total = lg_total + PLAYS_PER_GAME * ((home_exp - lg_off) + (away_exp - lg_off))

        rows.append({
            "home": h, "away": a,
            "gameday": g["gameday"], "gametime": g["gametime"],
            "neutral": g["location"] == "Neutral",
            "model_margin": margin, "model_total": total,
            "home_win_prob": 1 - norm_cdf(-margin / MARGIN_SD),
            "mkt_spread": g["spread_line"], "mkt_total": g["total_line"],
            "home_score": g["home_score"], "away_score": g["away_score"],
        })
    return pd.DataFrame(rows)


def fmt_spread(home, away, margin):
    if pd.isna(margin):
        return "n/a"
    if margin >= 0:
        return f"{home} -{margin:.1f}"
    return f"{away} -{-margin:.1f}"


# ---------- Odds ----------
@st.cache_data(ttl=1800)
def fetch_odds():
    url = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"
    params = {
        "apiKey": st.secrets["ODDS_API_KEY"],
        "regions": "uk",
        "markets": "h2h,spreads,totals",
        "oddsFormat": "decimal",
    }
    resp = requests.get(url, params=params, timeout=20)
    resp.raise_for_status()
    return resp.json(), resp.headers.get("x-requests-remaining")


def odds_to_rows(events):
    rows = []
    for ev in events:
        home = TEAM_ABBR.get(ev["home_team"])
        away = TEAM_ABBR.get(ev["away_team"])
        if not home or not away:
            continue
        for bk in ev.get("bookmakers", []):
            for mk in bk.get("markets", []):
                for oc in mk.get("outcomes", []):
                    rows.append({
                        "home": home, "away": away,
                        "book": bk["title"], "market": mk["key"],
                        "name": oc["name"], "price": oc["price"],
                        "point": oc.get("point"),
                    })
    return pd.DataFrame(rows)


def best_prices(odds, home, away):
    """Best price per side. Spreads/totals use the most common line only,
    so prices from different lines are never mixed."""
    g = odds[(odds["home"] == home) & (odds["away"] == away)]
    out = {}
    if g.empty:
        return out

    home_name = [k for k, v in TEAM_ABBR.items() if v == home][0]
    away_name = [k for k, v in TEAM_ABBR.items() if v == away][0]

    ml = g[g["market"] == "h2h"]
    if not ml.empty:
        out["ml_home"] = ml[ml["name"] == home_name]["price"].max()
        out["ml_away"] = ml[ml["name"] == away_name]["price"].max()

    sp = g[g["market"] == "spreads"]
    sp_home = sp[sp["name"] == home_name]
    if not sp_home.empty:
        line = sp_home["point"].mode().iloc[0]
        out["spread_line"] = line
        out["spread_home"] = sp_home[sp_home["point"] == line]["price"].max()
        sp_away = sp[(sp["name"] == away_name) & (sp["point"] == -line)]
        out["spread_away"] = sp_away["price"].max() if not sp_away.empty else None

    tot = g[g["market"] == "totals"]
    over = tot[tot["name"] == "Over"]
    if not over.empty:
        line = over["point"].mode().iloc[0]
        out["total_line"] = line
        out["over"] = over[over["point"] == line]["price"].max()
        under = tot[(tot["name"] == "Under") & (tot["point"] == line)]
        out["under"] = under["price"].max() if not under.empty else None

    return out


def traffic_light(market, prob, edge):
    green, amber, ceiling = BANDS[market]
    if prob > ceiling:
        return "🔴"
    if edge >= green:
        return "🟢"
    if edge >= amber:
        return "🟡"
    return "🔴"


def build_bets(preds, odds):
    bets = []
    for _, p in preds.iterrows():
        h, a = p["home"], p["away"]
        game = f"{a} @ {h}"
        prices = best_prices(odds, h, a)
        mu, tot = p["model_margin"], p["model_total"]
        candidates = []

        if prices.get("ml_home"):
            ph = p["home_win_prob"]
            candidates += [
                ("Moneyline", f"{h} to win", prices["ml_home"], ph),
                ("Moneyline", f"{a} to win", prices["ml_away"], 1 - ph),
            ]

        if prices.get("spread_home"):
            line = prices["spread_line"]  # home team's handicap, e.g. -3.5
            p_cover = 1 - norm_cdf((-line - mu) / MARGIN_SD)
            candidates.append(("Spread", f"{h} {line:+g}", prices["spread_home"], p_cover))
            if prices.get("spread_away"):
                candidates.append(("Spread", f"{a} {-line:+g}", prices["spread_away"], 1 - p_cover))

        if prices.get("over"):
            line = prices["total_line"]
            p_over = 1 - norm_cdf((line - tot) / TOTAL_SD)
            candidates.append(("Total", f"Over {line:g}", prices["over"], p_over))
            if prices.get("under"):
                candidates.append(("Total", f"Under {line:g}", prices["under"], 1 - p_over))

        for market, selection, price, prob in candidates:
            if not price or pd.isna(price):
                continue
            implied = 1 / price
            edge = prob - implied
            bets.append({
                "game": game, "market": market, "selection": selection,
                "price": price, "model": prob, "implied": implied,
                "edge": edge, "ev": prob * price - 1,
                "light": traffic_light(market, prob, edge),
            })
    return pd.DataFrame(bets)


# ---------- Page ----------
schedule = get_schedule(SEASON)
prior_schedule = get_schedule(PRIOR_SEASON)
reg = schedule[schedule["game_type"] == "REG"]

unplayed = reg[reg["home_score"].isna()]
current_week = int(unplayed["week"].min()) if len(unplayed) else int(reg["week"].max())

week = st.selectbox("Week", sorted(reg["week"].unique()), index=current_week - 1)
week_games = reg[reg["week"] == week]

ratings = blend_ratings(get_team_epa(SEASON), get_team_epa(PRIOR_SEASON))
lg_total = league_avg_total([schedule, prior_schedule])
preds = predict_games(week_games, ratings, lg_total)

tab_games, tab_bets, tab_ratings = st.tabs(["Model", "Game Bets", "Team Ratings"])

with tab_games:
    st.caption(f"League average total used: {lg_total:.1f} points")
    for _, p in preds.iterrows():
        with st.container(border=True):
            title = f"**{p['away']} @ {p['home']}**"
            if p["neutral"]:
                title += " · neutral site"
            st.markdown(title)
            st.caption(
                f"{p['gameday']} {p['gametime']} · "
                f"Model: {fmt_spread(p['home'], p['away'], p['model_margin'])}, "
                f"total {p['model_total']:.1f}, {p['home']} win {p['home_win_prob']:.0%} · "
                f"Market: {fmt_spread(p['home'], p['away'], p['mkt_spread'])}, "
                f"total {p['mkt_total']}"
            )
            if pd.notna(p["home_score"]):
                st.caption(f"Final: {p['away']} {p['away_score']:.0f} - {p['home_score']:.0f} {p['home']}")

with tab_bets:
    if week != current_week:
        st.info("Live odds are only available for the upcoming week.")
    else:
        if st.button("Fetch odds"):
            st.session_state["odds"] = fetch_odds()

        if "odds" in st.session_state:
            events, remaining = st.session_state["odds"]
            odds = odds_to_rows(events)
            bets = build_bets(preds, odds) if not odds.empty else pd.DataFrame()
            st.caption(f"Odds API credits remaining: {remaining}")

            if bets.empty:
                st.warning("No matching odds found for this week's games.")
            else:
                only_pos = st.checkbox("Only positive edges", value=True)
                sort_by = st.selectbox("Sort by", ["Edge", "Model %", "EV", "Price"])
                sort_col = {"Edge": "edge", "Model %": "model", "EV": "ev", "Price": "price"}[sort_by]

                view = bets[bets["edge"] > 0] if only_pos else bets
                for _, b in view.sort_values(sort_col, ascending=False).iterrows():
                    with st.container(border=True):
                        st.markdown(f"{b['light']} **{b['selection']}** · {b['market']}")
                        st.caption(
                            f"{b['game']} · Odds {b['price']:.2f} · Model {b['model']:.1%} · "
                            f"Implied {b['implied']:.1%} · Edge {b['edge']:+.1%} · EV {b['ev']:+.1%}"
                        )

with tab_ratings:
    st.dataframe(ratings.round(3), hide_index=True)