"""Rebuilds what the model predicted for a finished week and scores it against
what actually happened. Game bets are priced at nflverse's closing lines (free);
player props have no free historical prices, so they are scored as model reads."""
import numpy as np
import pandas as pd
import streamlit as st

import nfl_model as nm
import props as pr
import tracker as tk

GAME_MARKETS = ["Moneyline", "Spread", "Total"]


def closing_odds_rows(games):
    """nflverse closing lines as a one-bookmaker odds table, in the same shape
    the live Odds API parser produces, so the normal bet builder can price it."""
    rows = []
    for _, g in games.iterrows():
        h, a = g["home_team"], g["away_team"]
        hn, an = nm.ABBR_TEAM[h], nm.ABBR_TEAM[a]

        def add(market, name, american, point=None, default=None):
            if pd.isna(american):
                american = default
            if american is None:
                return
            rows.append({"event_id": g["game_id"], "home": h, "away": a,
                         "book": "Closing line", "market": market, "name": name,
                         "price": float(nm.american_to_decimal(american)), "point": point})

        add("h2h", hn, g["home_moneyline"])
        add("h2h", an, g["away_moneyline"])
        if pd.notna(g["spread_line"]):
            add("spreads", hn, g["home_spread_odds"], -float(g["spread_line"]), -110)
            add("spreads", an, g["away_spread_odds"], float(g["spread_line"]), -110)
        if pd.notna(g["total_line"]):
            add("totals", "Over", g["over_odds"], float(g["total_line"]), -110)
            add("totals", "Under", g["under_odds"], float(g["total_line"]), -110)
    return pd.DataFrame(rows)


def game_results(preds, week_games, raw=True):
    """Every moneyline/spread/total selection the model would have priced for
    this week at closing prices, settled against the final score. raw=True uses
    the pure model; raw=False uses the market-weighted blend the app bets from."""
    done = week_games[week_games["home_score"].notna()]
    if done.empty:
        return None
    odds = closing_odds_rows(done)
    params = {**nm.PARAMS, "w_margin": 1.0, "w_total": 1.0} if raw else nm.PARAMS
    bets = nm.build_bets(preds[preds["game_id"].isin(done["game_id"])], odds, params)
    if bets.empty:
        return None
    results = [tk.settle_one(r, done, None) for _, r in bets.iterrows()]
    bets = bets.assign(result=results)
    bets["Hit"] = bets["result"] == "WIN"
    bets["P/L (1u)"] = np.where(bets["result"] == "WIN", bets["price"] - 1,
                                np.where(bets["result"] == "LOSE", -1.0, 0.0))
    bets = bets[bets["result"].isin(["WIN", "LOSE"])]  # pushes are neither
    out = pd.DataFrame({
        "Light": bets["light"], "Market": bets["market"], "Selection": bets["selection"],
        "Game": bets["game"], "Model %": bets["model"] * 100, "Fair %": bets["fair"] * 100,
        "Edge": bets["edge"] * 100, "Odds": bets["price"], "Hit": bets["Hit"],
        "P/L (1u)": bets["P/L (1u)"]})
    score = {g["game_id"]: f"{g['away_team']} {g['away_score']:.0f} - "
                           f"{g['home_score']:.0f} {g['home_team']}" for _, g in done.iterrows()}
    out["Score"] = bets["game_id"].map(score).values
    return out.reset_index(drop=True)


@st.cache_data(ttl=3600, show_spinner="Rebuilding projections...")
def cached_roll(_logs, _key):
    return pr.rolling_table(_logs)


def prop_results(roll, season, week, params=None):
    """What the model projected for each player that week (using only earlier
    games) against what they actually did. Lines are standard thresholds, since
    there are no historical prop lines."""
    p = params or pr.PP
    d = roll[(roll["season"] == season) & (roll["week"] == week)]
    rows = []

    def add(q, label, line_txt, prob, proj, actual, hit):
        rows.append(pd.DataFrame({
            "Player": q["player_display_name"].to_numpy(), "Pos": q["position"].to_numpy(),
            "Market": label, "Line": line_txt, "Model %": prob * 100,
            "Projection": proj, "Actual": actual, "Hit": hit,
            "Game": (q["team"] + " vs " + q["opponent_team"]).to_numpy()})
        )

    for key, (label, _, col, _, _) in pr.MARKETS.items():
        q = d[pr.qualified(d, key)]
        if q.empty:
            continue
        if key == "anytime_td":
            lam = pr.td_lambda(q, p["td_mix"], p["gamma_td"]) * p["td_scale"]
            actual = q["td"].to_numpy(float)
            add(q, label, "1+ TD", 1 - np.exp(-lam), lam, actual, actual >= 1)
            continue
        mu = pr.stat_mu(q, key, p["gamma"])
        y = q[col].to_numpy(float)
        for line in pr.STD_LINES[key]:
            prob = np.array([pr.prop_over_under(m, key, line)[0] for m in mu])
            add(q, label, f"Over {line:g}", prob, mu, y, y > line)
    return pd.concat(rows, ignore_index=True) if rows else None
