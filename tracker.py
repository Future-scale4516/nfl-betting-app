"""Pick logging, settlement and closing-line (CLV) tracking.

Picks are stored in a private GitHub Gist so they survive Streamlit Cloud
restarts (a Gist, unlike a repo commit, does not trigger an app redeploy).
If the Gist isn't configured, picks live in the browser session only and can
be downloaded as CSV."""
import io
import json
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
    "settled_at", "legs", "close_fair", "clv_fair", "close_ref",
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
    for c in ("player_id", "player", "result", "settled_at", "pick_id", "side", "legs",
              "close_ref"):
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
                      "close_price", "clv", "profit", "close_fair", "clv_fair"):
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


FOLD_KEY = "acca"


def _num(x):
    """JSON-safe number: NaN/None -> None."""
    try:
        return None if x is None or pd.isna(x) else float(x)
    except (TypeError, ValueError):
        return None


def _leg(b):
    """Normalise a bet row (Game Bets row or Suggested Bets leg) to a stored leg."""
    return {
        "game_id": str(b["game_id"]), "game": str(b["game"]),
        "selection": str(b.get("selection", b.get("label", ""))),
        "market_key": str(b["market_key"]), "side": str(b.get("side", "")),
        "line": _num(b.get("line")), "player_id": str(b.get("player_id", "") or ""),
        "player": str(b.get("player", "") or ""),
        "price": _num(b.get("price", b.get("odds"))), "model": _num(b.get("model")),
    }


def log_fold(legs, stake, season, week):
    """Log an accumulator as ONE pick carrying its legs. `legs` is a list of
    dicts or a DataFrame. Returns False if the same fold is already logged."""
    legs = [_leg(b) for b in (legs.to_dict("records") if hasattr(legs, "to_dict") else legs)]
    if len(legs) < 2:
        raise ValueError("A fold needs at least two legs.")
    pid = "ACCA|" + "&".join(sorted(
        f"{l['game_id']}|{l['market_key']}|{l['side']}|{l['line'] if l['line'] is not None else ''}"
        f"|{l['player_id']}" for l in legs))
    df = load_picks()
    if pid in set(df["pick_id"].dropna()):
        return False
    odds = float(np.prod([l["price"] for l in legs]))
    model = float(np.prod([l["model"] if l["model"] is not None else np.nan for l in legs]))
    row = {
        "pick_id": pid, "logged_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "season": season, "week": week, "game_id": "", "game": f"{len(legs)}-fold",
        "market": "Accumulator", "market_key": FOLD_KEY, "side": "",
        "selection": " + ".join(l["selection"] for l in legs),
        "book": "", "price": odds, "model": model, "implied": 1 / odds, "stake": stake,
        "legs": json.dumps(legs),
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


def fold_legs(r, sched, stats):
    """[(leg dict, WIN/LOSE/PUSH/VOID/None)] for a logged fold."""
    out = []
    for leg in json.loads(r["legs"]):
        out.append((leg, settle_one(pd.Series({**leg, "line": leg["line"] if leg["line"]
                                               is not None else np.nan}), sched, stats)))
    return out


def settle_fold(r, sched, stats):
    """(result, effective_odds) or (None, None) while still alive. One lost leg
    kills the fold immediately; pushed/void legs count as 1.00 like a bookmaker."""
    legs = fold_legs(r, sched, stats)
    res = [s for _, s in legs]
    if "LOSE" in res:
        return "LOSE", 0.0
    if any(s is None for s in res):
        return None, None
    won = [l["price"] for l, s in legs if s == "WIN"]
    if not won:
        return "PUSH", 1.0
    return "WIN", float(np.prod(won))


def settle_picks(df):
    """Fill in result/profit for every pick whose game has finished."""
    df = df.copy()
    open_idx = df.index[df["result"].isna() | (df["result"] == "")]
    n = 0
    for season in df.loc[open_idx, "season"].dropna().unique():
        sched = nm.get_schedule(int(season))
        stats = _player_stats(int(season))
        for i in open_idx[df.loc[open_idx, "season"] == season]:
            if df.at[i, "market_key"] == FOLD_KEY:
                res, price = settle_fold(df.loc[i], sched, stats)
            else:
                res, price = settle_one(df.loc[i], sched, stats), df.at[i, "price"]
            if res is None:
                continue
            stake = df.at[i, "stake"]
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
        if df.at[i, "market_key"] == FOLD_KEY:  # no single closing price for a fold
            continue
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
                          avg_model=("model", "mean"), avg_clv=("clv", "mean"),
                          avg_clv_fair=("clv_fair", "mean"))
    g["win_rate"] = g["wins"] / g["bets"]
    g["roi"] = g["profit"] / g["staked"]
    return g[["bets", "wins", "win_rate", "staked", "profit", "roi", "avg_model", "avg_clv",
              "avg_clv_fair"]]


# ---------- CLV against the de-vigged close (from snapshot.py) ----------
def load_snapshots(prefix, force=False):
    """All snap_<prefix>_*.csv files from the Gist as one DataFrame (empty if none)."""
    key = f"snap_{prefix}"
    if not force and key in st.session_state:
        return st.session_state[key]
    frames = []
    if gist_ready():
        r = requests.get(f"https://api.github.com/gists/{st.secrets['GIST_ID']}",
                         headers=_headers(), timeout=30)
        r.raise_for_status()
        for name, f in r.json()["files"].items():
            if name.startswith(f"snap_{prefix}_") and name.endswith(".csv"):
                text = f.get("content", "")
                if f.get("truncated"):
                    text = requests.get(f["raw_url"], headers=_headers(), timeout=30).text
                if text.strip():
                    frames.append(pd.read_csv(io.StringIO(text)))
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    st.session_state[key] = df
    return df


def _close_rows(snaps, home_abbr, away_abbr, gameday):
    """Snapshot rows for one game: the latest ones before kickoff (the close)."""
    if snaps is None or snaps.empty:
        return None
    hn, an = nm.ABBR_TEAM.get(home_abbr), nm.ABBR_TEAM.get(away_abbr)
    s = snaps[(snaps["home"] == hn) & (snaps["away"] == an)].copy()
    if s.empty:
        return None
    ko = pd.to_datetime(s["commence_utc"], utc=True)
    day = pd.to_datetime(gameday, errors="coerce")
    if pd.notna(day):
        s = s[(ko.dt.tz_localize(None).dt.normalize() - day).abs() <= pd.Timedelta(days=1)]
    s = s[s["mins_to_ko"] <= 130]  # only a snapshot taken within ~2h of kickoff is a close
    if s.empty:
        return None
    return s[s["ts_utc"] == s["ts_utc"].max()]


def _pick_fair(rows, market, name, point=None):
    r = rows[(rows["market"] == market) & (rows["name"] == name)]
    if point is not None:
        r = r[(r["point"].astype(float) - point).abs() < 1e-9]
    return r


def _row_fair(r):
    """(fair prob, label) for a snapshot row: Pinnacle if it lists the pair, else the
    median across books."""
    if pd.notna(r.get("fair_pinn")) and r.get("fair_pinn") != "":
        return float(r["fair_pinn"]), "Pinnacle"
    if pd.notna(r.get("fair_median")) and r.get("fair_median") != "":
        return float(r["fair_median"]), "Market median"
    return None, None


def closing_fair(pick, games, props):
    """(fair probability at the close for this pick's selection, label) or (None, None).
    Spread and total picks whose line has moved are converted to the pick's own line
    with the same key-number-aware distribution the app prices with."""
    h, a = [x.strip() for x in str(pick["game"]).split("@")][::-1]
    key, side, line = pick["market_key"], pick["side"], pick["line"]
    params = nm.PARAMS
    if key in ("moneyline", "spread", "total"):
        rows = _close_rows(games, h, a, pick["gameday"])
        if rows is None:
            return None, None
        if key == "moneyline":
            r = _pick_fair(rows, "h2h", nm.ABBR_TEAM[h if side == "home" else a])
            return _row_fair(r.iloc[0]) if len(r) else (None, None)
        if key == "spread":
            r = rows[(rows["market"] == "spreads") & (rows["name"] == nm.ABBR_TEAM[h])]
            if r.empty:
                return None, None
            r = r.sort_values("n_books", ascending=False).iloc[0]
            f, lab = _row_fair(r)
            if f is None:
                return None, None
            mu = nm.implied_mu(f, -float(r["point"]), params["margin_sd_raw"])
            cond_home = nm.cover_prob(mu, params["margin_sd_raw"], -float(line))[0]
            return (cond_home if side == "home" else 1 - cond_home), lab
        r = rows[(rows["market"] == "totals") & (rows["name"] == "Over")]
        if r.empty:
            return None, None
        r = r.sort_values("n_books", ascending=False).iloc[0]
        f, lab = _row_fair(r)
        if f is None:
            return None, None
        mu = nm.implied_mu(f, float(r["point"]), params["total_sd_raw"], False, 0, 130)
        over = nm.cover_prob(mu, params["total_sd_raw"], float(line), False, 0, 130)[0]
        return (over if side == "over" else 1 - over), lab

    # player props: same line only (a moved prop line is not comparable)
    if props is None or props.empty or key not in pr.MARKETS:
        return None, None
    rows = _close_rows(props, h, a, pick["gameday"])
    if rows is None:
        return None, None
    name = {"yes": "Yes", "over": "Over", "under": "Under"}[side]
    r = rows[(rows["market"] == pr.MARKETS[key][1]) & (rows["name"] == name)
             & (rows["player"].map(pr.norm_name) == pr.norm_name(pick["player"]))]
    if key != "anytime_td":
        r = r[(r["point"].astype(float) - float(line)).abs() < 1e-9]
    if r.empty:
        return None, None
    r = r.iloc[0]
    if pd.notna(r["fair_median"]) and r["fair_median"] != "":
        return float(r["fair_median"]), "Market median"
    return 1 / float(r["median_price"]), "Implied (incl. margin)"


def update_clv_fair(df, games, props):
    """CLV vs the de-vigged close: price * close_fair - 1. Positive = you got a better
    price than the market's own fair price at kickoff. Returns (df, n updated)."""
    df = df.copy()
    df["close_ref"] = df["close_ref"].astype("object")
    for c in ("close_fair", "clv_fair"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    n = 0
    for i in df.index:
        if df.at[i, "market_key"] == FOLD_KEY:
            continue
        f, lab = closing_fair(df.loc[i], games, props)
        if f:
            df.at[i, "close_fair"] = f
            df.at[i, "clv_fair"] = df.at[i, "price"] * f - 1
            df.at[i, "close_ref"] = lab
            n += 1
    return df, n
