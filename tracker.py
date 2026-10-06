"""Pick logging, settlement and closing-line (CLV) tracking.

Picks are stored in a private GitHub Gist so they survive Streamlit Cloud
restarts (a Gist, unlike a repo commit, does not trigger an app redeploy).
If the Gist isn't configured, picks live in the browser session only and can
be downloaded as CSV."""
import io
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st
import nflreadpy as nfl

import nfl_model as nm
import props as pr

COLUMNS = [
    "pick_id", "logged_at", "season", "week", "game_id", "game", "gameday", "gametime",
    "market", "market_key", "side", "line", "player_id", "player", "selection", "book",
    "price", "model", "implied", "stake", "close_price", "clv", "result", "profit",
    "settled_at",
]
GIST_FILE = "picks.csv"


# ---------- Storage ----------
def gist_ready():
    try:
        return bool(st.secrets["GITHUB_TOKEN"]) and bool(st.secrets["GIST_ID"])
    except Exception:
        return False


def _headers():
    return {"Authorization": f"Bearer {st.secrets['GITHUB_TOKEN']}",
            "Accept": "application/vnd.github+json"}


def _clean(df):
    for c in COLUMNS:
        if c not in df.columns:
            df[c] = np.nan
    df = df[COLUMNS].copy()
    for c in ("player_id", "player", "result", "settled_at", "pick_id", "side"):
        df[c] = df[c].astype("object")
    return df


def load_picks(force=False):
    if not force and "picks" in st.session_state:
        return st.session_state["picks"]
    df = pd.DataFrame(columns=COLUMNS)
    if gist_ready():
        r = requests.get(f"https://api.github.com/gists/{st.secrets['GIST_ID']}",
                         headers=_headers(), timeout=20)
        r.raise_for_status()
        f = r.json()["files"].get(GIST_FILE)
        if f and f.get("content", "").strip():
            df = pd.read_csv(io.StringIO(f["content"]), dtype=str)
            for c in ("season", "week", "line", "price", "model", "implied", "stake",
                      "close_price", "clv", "profit"):
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
    st.session_state["picks"] = _clean(df)
    return st.session_state["picks"]


def save_picks(df):
    df = _clean(df)
    st.session_state["picks"] = df
    if gist_ready():
        r = requests.patch(f"https://api.github.com/gists/{st.secrets['GIST_ID']}",
                           headers=_headers(), timeout=20,
                           json={"files": {GIST_FILE: {"content": df.to_csv(index=False)}}})
        r.raise_for_status()


# ---------- Logging ----------
def pick_id(b):
    line = "" if pd.isna(b.get("line")) else b.get("line")
    return f"{b['game_id']}|{b['market_key']}|{b['side']}|{line}|{b.get('player_id', '')}"


def log_pick(bet, stake, season, week):
    """Add a pick. Returns False if it was already logged."""
    df = load_picks()
    pid = pick_id(bet)
    if pid in set(df["pick_id"].dropna()):
        return False
    row = {
        "pick_id": pid, "logged_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "season": season, "week": week, "game_id": bet["game_id"], "game": bet["game"],
        "gameday": bet.get("gameday"), "gametime": bet.get("gametime"),
        "market": bet["market"], "market_key": bet["market_key"], "side": bet["side"],
        "line": bet.get("line"), "player_id": bet.get("player_id", ""),
        "player": bet.get("player", ""), "selection": bet["selection"],
        "book": bet.get("book"), "price": bet["price"], "model": bet["model"],
        "implied": bet["implied"], "stake": stake,
    }
    save_picks(pd.concat([df, pd.DataFrame([row])], ignore_index=True))
    return True


# ---------- Settlement ----------
@st.cache_data(ttl=600)
def _player_stats(season):
    ps = nfl.load_player_stats([season]).to_pandas()
    ps = ps[ps["season_type"] == "REG"]
    ps = ps.assign(
        anytime_td=ps["rushing_tds"].fillna(0) + ps["receiving_tds"].fillna(0),
        rush_rec_yards=ps["rushing_yards"].fillna(0) + ps["receiving_yards"].fillna(0),
    )
    return ps


STAT_COL = {
    "anytime_td": "anytime_td", "receptions": "receptions", "rec_yards": "receiving_yards",
    "rush_yards": "rushing_yards", "pass_yards": "passing_yards", "pass_tds": "passing_tds",
    "rush_rec_yards": "rush_rec_yards",
}


def _outcome(value, line, side):
    """WIN/LOSE/PUSH for an over/under style bet."""
    if value == line:
        return "PUSH"
    over = value > line
    return "WIN" if (over and side == "over") or (not over and side == "under") else "LOSE"


def settle_one(r, sched, stats):
    g = sched[sched["game_id"] == r["game_id"]]
    if g.empty or pd.isna(g.iloc[0]["home_score"]):
        return None
    g = g.iloc[0]
    margin = g["home_score"] - g["away_score"]
    key, side, line = r["market_key"], r["side"], r["line"]

    if key == "moneyline":
        if margin == 0:
            return "PUSH"
        return "WIN" if (margin > 0) == (side == "home") else "LOSE"
    if key == "spread":  # line = home team's handicap
        adj = margin + line
        if adj == 0:
            return "PUSH"
        return "WIN" if (adj > 0) == (side == "home") else "LOSE"
    if key == "total":
        return _outcome(g["home_score"] + g["away_score"], line, side)

    # nflverse can publish final scores before player stats: stay pending until
    # this game's stats exist, otherwise a late stat feed would look like a DNP.
    if stats[stats["game_id"] == r["game_id"]].empty:
        return None
    row = stats[(stats["player_id"] == r["player_id"]) & (stats["game_id"] == r["game_id"])]
    if row.empty:
        return "VOID"  # did not play
    value = float(row.iloc[0][STAT_COL[key]])
    if key == "anytime_td":
        return "WIN" if value >= 1 else "LOSE"
    return _outcome(value, line, side)


def settle_picks(df):
    """Fill in result/profit for every pick whose game has finished."""
    df = df.copy()
    open_idx = df.index[df["result"].isna() | (df["result"] == "")]
    n = 0
    for season in df.loc[open_idx, "season"].dropna().unique():
        sched = nm.get_schedule(int(season))
        stats = _player_stats(int(season))
        for i in open_idx[df.loc[open_idx, "season"] == season]:
            res = settle_one(df.loc[i], sched, stats)
            if res is None:
                continue
            stake, price = df.at[i, "stake"], df.at[i, "price"]
            df.at[i, "result"] = res
            df.at[i, "profit"] = stake * (price - 1) if res == "WIN" else (
                -stake if res == "LOSE" else 0.0)
            df.at[i, "settled_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            n += 1
    return df, n


# ---------- Closing prices / CLV ----------
def _current_price(r, game_odds, prop_odds):
    """Best price now for the same selection at the same line, or None."""
    key, side, line = r["market_key"], r["side"], r["line"]
    if key in ("moneyline", "spread", "total"):
        home, away = [x.strip() for x in r["game"].split("@")][::-1]
        pr_ = nm.best_prices(game_odds, home, away)
        if key == "moneyline":
            got = pr_.get("ml_home" if side == "home" else "ml_away")
        elif key == "spread":
            got = pr_.get("spread_home" if side == "home" else "spread_away") \
                if pr_.get("spread_line") == line else None
        else:
            got = pr_.get("over" if side == "over" else "under") \
                if pr_.get("total_line") == line else None
        return got[0] if got and got[0] else None

    if prop_odds is None or prop_odds.empty:
        return None
    name = {"yes": "Yes", "over": "Over", "under": "Under"}[side]
    m = prop_odds[(prop_odds["game_id"] == r["game_id"])
                  & (prop_odds["market_key"] == key)
                  & (prop_odds["player_norm"] == pr.norm_name(r["player"]))
                  & (prop_odds["name"] == name)]
    if key != "anytime_td":
        m = m[m["point"] == line]
    return float(m["price"].max()) if len(m) else None


def update_closing(df, game_odds, prop_odds):
    """Record the latest available price for every unsettled pick. Press this
    shortly before kickoff so the stored price is close to the closing line."""
    df = df.copy()
    n = 0
    for i in df.index[df["result"].isna() | (df["result"] == "")]:
        p = _current_price(df.loc[i], game_odds, prop_odds)
        if p:
            df.at[i, "close_price"] = p
            df.at[i, "clv"] = df.at[i, "price"] / p - 1
            n += 1
    return df, n


# ---------- Summary ----------
def summarise(df, by="market"):
    s = df[df["result"].isin(["WIN", "LOSE", "PUSH"])].copy()
    if s.empty:
        return pd.DataFrame()
    s["staked"] = s["stake"]
    g = s.groupby(by).agg(bets=("result", "size"),
                          wins=("result", lambda x: (x == "WIN").sum()),
                          staked=("staked", "sum"), profit=("profit", "sum"),
                          avg_model=("model", "mean"), avg_clv=("clv", "mean"))
    g["win_rate"] = g["wins"] / g["bets"]
    g["roi"] = g["profit"] / g["staked"]
    return g[["bets", "wins", "win_rate", "staked", "profit", "roi", "avg_model", "avg_clv"]]
