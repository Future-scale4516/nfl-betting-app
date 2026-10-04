"""Shared model code for the NFL app: data loading, team ratings, game
predictions, odds handling and the backtest engine."""
import math

import numpy as np
import pandas as pd
import requests
import streamlit as st
import nflreadpy as nfl

try:  # keep nflreadpy from holding full play-by-play files in memory
    from nflreadpy.config import update_config
    update_config(cache_mode="off")
except Exception:
    pass

SEASON = 2026
PRIOR_SEASON = 2025

# ---------- Model settings ----------
# Fitted on the 2024-2025 backtest (see the Backtest page). The fitted weight on
# the model vs the closing line was about zero (-0.07 +/- 0.16 for margins), so
# 0.10 is a small allowance inside that uncertainty, not an estimated edge.
PARAMS = {
    "scale": 46.8,          # points per 1.0 of EPA/play difference
    "hfa": 1.9,             # home-field advantage in points
    "total_scale": 28.9,    # points per 1.0 of combined EPA/play above average
    "margin_sd": 12.5,      # spread of real margins around the used line
    "total_sd": 12.8,       # spread of real totals around the used line
    "k": 400,               # plays of current-season data before 50/50 with last year
    "regress": 0.5,         # how far last year's ratings are pulled to league average
    "w_margin": 0.10,       # weight on model vs market line for margins (1 = pure model)
    "w_total": 0.10,        # same, for totals
    "qb_adj": 2.3,          # points removed from a team missing its starting QB
}

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
ABBR_TEAM = {v: k for k, v in TEAM_ABBR.items()}

# Yard-line buckets used for touchdown expectancy (see props.py)
TD_BUCKETS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 50, 100]


# ---------- Small helpers ----------
def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def norm_cdf_arr(x):
    return 0.5 * (1 + np.vectorize(math.erf, otypes=[float])(np.asarray(x) / math.sqrt(2)))


def american_to_decimal(a):
    a = np.asarray(a, dtype=float)
    return np.where(a > 0, 1 + a / 100, 1 + 100 / np.abs(a))


def ols(X, y):
    """Least squares with standard errors. Returns (beta, se, residuals)."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = max(len(y) - X.shape[1], 1)
    s2 = float(resid @ resid) / dof
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    return beta, se, resid


# ---------- Data ----------
@st.cache_data(ttl=3600)
def get_schedule(season):
    return nfl.load_schedules(seasons=[season]).to_pandas()


def _read_pbp(season):
    cols = ["season_type", "week", "posteam", "defteam", "play_type", "epa",
            "yardline_100", "rusher_player_id", "receiver_player_id",
            "rush_touchdown", "pass_touchdown", "two_point_attempt"]
    pbp = nfl.load_pbp(seasons=[season]).select(cols).to_pandas()
    plays = pbp[
        (pbp["season_type"] == "REG")
        & pbp["play_type"].isin(["pass", "run"])
        & pbp["epa"].notna()
    ]

    # Team-week EPA sums (small table, used for ratings)
    off = plays.groupby(["week", "posteam"])["epa"].agg(off_sum="sum", off_n="count")
    off = off.reset_index().rename(columns={"posteam": "team"})
    de = plays.groupby(["week", "defteam"])["epa"].agg(def_sum="sum", def_n="count")
    de = de.reset_index().rename(columns={"defteam": "team"})
    tw = off.merge(de, on=["week", "team"], how="outer").fillna(0)
    tw["season"] = season

    # Player opportunities by field position (used for touchdown expectancy)
    plays = plays[plays["two_point_attempt"].fillna(0) != 1]
    plays = plays[plays["yardline_100"].notna()].copy()
    plays["bucket"] = pd.cut(plays["yardline_100"], TD_BUCKETS, labels=False)
    rush = plays[(plays["play_type"] == "run") & plays["rusher_player_id"].notna()]
    rush = rush.assign(player_id=rush["rusher_player_id"], kind="rush",
                       td=rush["rush_touchdown"].fillna(0))
    rec = plays[(plays["play_type"] == "pass") & plays["receiver_player_id"].notna()]
    rec = rec.assign(player_id=rec["receiver_player_id"], kind="rec",
                     td=rec["pass_touchdown"].fillna(0))
    opps = pd.concat([rush, rec])
    opps = (opps.groupby(["week", "player_id", "kind", "bucket"])["td"]
            .agg(n="size", td="sum").reset_index())
    opps["season"] = season
    return tw, opps


@st.cache_data(ttl=3600, show_spinner="Loading current season data...")
def _pbp_current(season):
    return _read_pbp(season)


@st.cache_data(show_spinner="Loading historical season data...")
def _pbp_past(season):
    return _read_pbp(season)


def load_season_pbp(season):
    """Returns (team_weeks, player_opportunities) for one season."""
    return (_pbp_current if season >= SEASON else _pbp_past)(season)


def load_team_weeks(seasons):
    return pd.concat([load_season_pbp(s)[0] for s in seasons], ignore_index=True)


# ---------- Ratings ----------
def _rate(tw):
    g = tw.groupby("team")[["off_sum", "off_n", "def_sum", "def_n"]].sum()
    g["off_epa"] = g["off_sum"] / g["off_n"].replace(0, np.nan)
    g["def_epa"] = g["def_sum"] / g["def_n"].replace(0, np.nan)
    return g


def ratings_asof(tw_all, season, week, k=None, regress=None):
    """Team ratings using only games before `week` of `season`, blended with
    the prior season. week=99 means 'everything played so far'."""
    k = PARAMS["k"] if k is None else k
    regress = PARAMS["regress"] if regress is None else regress

    prior = _rate(tw_all[tw_all["season"] == season - 1])
    cur = tw_all[(tw_all["season"] == season) & (tw_all["week"] < week)]
    cur = _rate(cur) if len(cur) else pd.DataFrame(
        columns=["off_n", "off_epa", "def_epa"])

    teams = prior.index
    out = pd.DataFrame(index=teams)
    n = cur["off_n"].reindex(teams).fillna(0) if len(cur) else pd.Series(0.0, index=teams)
    w = n / (n + k)
    for col in ["off_epa", "def_epa"]:
        league = prior[col].mean()
        prior_reg = prior[col] * (1 - regress) + league * regress
        now = cur[col].reindex(teams).fillna(prior_reg) if len(cur) else prior_reg
        out[col] = w * now + (1 - w) * prior_reg
    out["net_epa"] = out["off_epa"] - out["def_epa"]
    out["weight_cur"] = w
    out.index.name = "team"
    return out


def league_avg_total(sched_all, season, week):
    done = sched_all[
        (sched_all["game_type"] == "REG") & sched_all["home_score"].notna()
        & ((sched_all["season"] == season - 1)
           | ((sched_all["season"] == season) & (sched_all["week"] < week)))
    ]
    return float((done["home_score"] + done["away_score"]).mean())


# ---------- Game predictions ----------
def epa_components(ratings, home, away):
    lg_off = ratings["off_epa"].mean()
    lg_def = ratings["def_epa"].mean()
    home_exp = ratings.at[home, "off_epa"] + (ratings.at[away, "def_epa"] - lg_def)
    away_exp = ratings.at[away, "off_epa"] + (ratings.at[home, "def_epa"] - lg_def)
    return home_exp - away_exp, (home_exp - lg_off) + (away_exp - lg_off)


def predict_games(games, ratings, lg_total, params=None, qb_out=None):
    """Predict margin/total for each game. qb_out maps team -> True when its
    starting QB is out (applies the qb_adj penalty)."""
    p = params or PARAMS
    qb_out = qb_out or {}
    rows = []
    for _, g in games.iterrows():
        h, a = g["home_team"], g["away_team"]
        if h not in ratings.index or a not in ratings.index:
            continue
        neutral = g["location"] == "Neutral"
        epa_diff, epa_tot = epa_components(ratings, h, a)
        raw_margin = p["scale"] * epa_diff + (0 if neutral else p["hfa"])
        raw_total = lg_total + p["total_scale"] * epa_tot

        # Missing starting QB: penalise that team in the raw model
        if qb_out.get(h):
            raw_margin -= p["qb_adj"]
        if qb_out.get(a):
            raw_margin += p["qb_adj"]

        mkt_m, mkt_t = g["spread_line"], g["total_line"]
        margin = raw_margin if pd.isna(mkt_m) else mkt_m + p["w_margin"] * (raw_margin - mkt_m)
        total = raw_total if pd.isna(mkt_t) else mkt_t + p["w_total"] * (raw_total - mkt_t)

        rows.append({
            "game_id": g["game_id"], "home": h, "away": a,
            "gameday": g["gameday"], "gametime": g["gametime"], "neutral": neutral,
            "raw_margin": raw_margin, "raw_total": raw_total,
            "model_margin": margin, "model_total": total,
            "home_win_prob": 1 - norm_cdf(-margin / p["margin_sd"]),
            "mkt_spread": mkt_m, "mkt_total": mkt_t,
            "home_score": g["home_score"], "away_score": g["away_score"],
            "qb_out_home": bool(qb_out.get(h)), "qb_out_away": bool(qb_out.get(a)),
        })
    return pd.DataFrame(rows)


def fmt_spread(home, away, margin):
    if pd.isna(margin):
        return "n/a"
    if margin >= 0:
        return f"{home} -{margin:.1f}"
    return f"{away} -{-margin:.1f}"


# ---------- Odds API ----------
ODDS_BASE = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl"


def _odds_key():
    return st.secrets["ODDS_API_KEY"]


def fetch_odds_raw(regions="uk"):
    resp = requests.get(f"{ODDS_BASE}/odds", params={
        "apiKey": _odds_key(), "regions": regions,
        "markets": "h2h,spreads,totals", "oddsFormat": "decimal",
    }, timeout=20)
    resp.raise_for_status()
    return resp.json(), resp.headers.get("x-requests-remaining")


@st.cache_data(ttl=1800)
def fetch_odds():
    return fetch_odds_raw()


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
                        "event_id": ev["id"], "home": home, "away": away,
                        "book": bk["title"], "market": mk["key"],
                        "name": oc["name"], "price": oc["price"],
                        "point": oc.get("point"),
                    })
    return pd.DataFrame(rows)


def _best(df):
    """(price, book) of the highest price in df, or (None, None)."""
    if df is None or df.empty:
        return None, None
    r = df.loc[df["price"].idxmax()]
    return float(r["price"]), r["book"]


def best_prices(odds, home, away):
    """Best price + bookmaker per side. Spreads/totals use the most common
    line only, so prices from different lines are never mixed."""
    g = odds[(odds["home"] == home) & (odds["away"] == away)]
    out = {}
    if g.empty:
        return out
    hn, an = ABBR_TEAM[home], ABBR_TEAM[away]

    ml = g[g["market"] == "h2h"]
    out["ml_home"] = _best(ml[ml["name"] == hn])
    out["ml_away"] = _best(ml[ml["name"] == an])

    sp = g[g["market"] == "spreads"]
    sp_home = sp[sp["name"] == hn]
    if not sp_home.empty:
        line = float(sp_home["point"].mode().iloc[0])
        out["spread_line"] = line
        out["spread_home"] = _best(sp_home[sp_home["point"] == line])
        out["spread_away"] = _best(sp[(sp["name"] == an) & (sp["point"] == -line)])

    tot = g[g["market"] == "totals"]
    over = tot[tot["name"] == "Over"]
    if not over.empty:
        line = float(over["point"].mode().iloc[0])
        out["total_line"] = line
        out["over"] = _best(over[over["point"] == line])
        out["under"] = _best(tot[(tot["name"] == "Under") & (tot["point"] == line)])
    return out


def map_events_to_games(events, games):
    """Odds API event id -> nflverse game_id, matched on home/away team."""
    out = {}
    for ev in events:
        h, a = TEAM_ABBR.get(ev["home_team"]), TEAM_ABBR.get(ev["away_team"])
        row = games[(games["home_team"] == h) & (games["away_team"] == a)]
        if len(row):
            out[ev["id"]] = row.iloc[0]["game_id"]
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


def build_bets(preds, odds, params=None):
    p = params or PARAMS
    bets = []
    for _, r in preds.iterrows():
        h, a = r["home"], r["away"]
        pr = best_prices(odds, h, a)
        if not pr:
            continue
        mu, tot = r["model_margin"], r["model_total"]
        cands = []  # (market, key, side, label, line, (price, book), prob)

        ph = r["home_win_prob"]
        cands += [
            ("Moneyline", "moneyline", "home", f"{h} to win", None, pr.get("ml_home"), ph),
            ("Moneyline", "moneyline", "away", f"{a} to win", None, pr.get("ml_away"), 1 - ph),
        ]
        if pr.get("spread_home") and pr["spread_home"][0]:
            line = pr["spread_line"]  # home team's handicap, e.g. -3.5
            p_cover = 1 - norm_cdf((-line - mu) / p["margin_sd"])
            cands.append(("Spread", "spread", "home", f"{h} {line:+g}", line,
                          pr["spread_home"], p_cover))
            cands.append(("Spread", "spread", "away", f"{a} {-line:+g}", line,
                          pr.get("spread_away"), 1 - p_cover))
        if pr.get("over") and pr["over"][0]:
            line = pr["total_line"]
            p_over = 1 - norm_cdf((line - tot) / p["total_sd"])
            cands.append(("Total", "total", "over", f"Over {line:g}", line, pr["over"], p_over))
            cands.append(("Total", "total", "under", f"Under {line:g}", line,
                          pr.get("under"), 1 - p_over))

        for market, key, side, label, line, pb, prob in cands:
            if not pb or pb[0] is None or pd.isna(pb[0]):
                continue
            price, book = pb
            implied = 1 / price
            edge = prob - implied
            bets.append({
                "game_id": r["game_id"], "home": h, "away": a,
                "game": f"{a} @ {h}", "gameday": r["gameday"], "gametime": r["gametime"],
                "market": market, "market_key": key, "side": side,
                "selection": label, "line": line, "player_id": "", "player": "",
                "price": price, "book": book, "model": prob, "implied": implied,
                "edge": edge, "ev": prob * price - 1,
                "light": traffic_light(market, prob, edge),
            })
    return pd.DataFrame(bets)


# ---------- Backtest ----------
def run_backtest(tw_all, sched_all, seasons, k=None, regress=None):
    """Replay past seasons week by week using only information available
    before each week. Returns one row per game."""
    rows = []
    for season in seasons:
        sch = sched_all[(sched_all["season"] == season) & (sched_all["game_type"] == "REG")]
        for week in sorted(sch["week"].unique()):
            r = ratings_asof(tw_all, season, week, k, regress)
            lg_total = league_avg_total(sched_all, season, week)
            for _, g in sch[sch["week"] == week].iterrows():
                h, a = g["home_team"], g["away_team"]
                if h not in r.index or a not in r.index or pd.isna(g["home_score"]):
                    continue
                d, t = epa_components(r, h, a)
                rows.append({
                    "season": season, "week": week, "game_id": g["game_id"],
                    "home": h, "away": a, "neutral": g["location"] == "Neutral",
                    "epa_diff": d, "epa_tot": t, "lg_total": lg_total,
                    "result": g["home_score"] - g["away_score"],
                    "total": g["home_score"] + g["away_score"],
                    "spread_line": g["spread_line"], "total_line": g["total_line"],
                    "home_ml": g["home_moneyline"], "away_ml": g["away_moneyline"],
                    "home_sp_odds": g["home_spread_odds"], "away_sp_odds": g["away_spread_odds"],
                    "over_odds": g["over_odds"], "under_odds": g["under_odds"],
                    "home_qb": g["home_qb_id"], "away_qb": g["away_qb_id"],
                })
    bt = pd.DataFrame(rows)
    return _flag_qb_changes(bt)


def _flag_qb_changes(bt):
    """Mark games where a team's starting QB differs from its previous game."""
    long = pd.concat([
        bt[["season", "week", "game_id", "home", "home_qb"]].rename(
            columns={"home": "team", "home_qb": "qb"}).assign(side="home"),
        bt[["season", "week", "game_id", "away", "away_qb"]].rename(
            columns={"away": "team", "away_qb": "qb"}).assign(side="away"),
    ]).sort_values(["team", "season", "week"])
    long["prev_qb"] = long.groupby("team")["qb"].shift(1)
    long["changed"] = long["prev_qb"].notna() & (long["qb"] != long["prev_qb"])
    for side in ("home", "away"):
        m = long[long["side"] == side].set_index("game_id")["changed"]
        bt[f"{side}_qb_changed"] = bt["game_id"].map(m).fillna(False).astype(bool)
    return bt


def fit_backtest(bt):
    """Fit scale / home field / SDs from backtest rows, and test whether the
    model adds information beyond the closing line."""
    nn = (~bt["neutral"]).astype(float).to_numpy()
    (scale, hfa), _, res = ols(np.column_stack([bt["epa_diff"], nn]), bt["result"])
    model_margin = scale * bt["epa_diff"].to_numpy() + hfa * nn

    (tscale,), _, tres = ols(bt[["epa_tot"]], bt["total"] - bt["lg_total"])
    model_total = bt["lg_total"].to_numpy() + tscale * bt["epa_tot"].to_numpy()

    ok = bt["spread_line"].notna().to_numpy()
    mkt = bt.loc[ok, "spread_line"].to_numpy()
    X = np.column_stack([np.ones(ok.sum()), mkt, model_margin[ok] - mkt])
    b, se, mres = ols(X, bt.loc[ok, "result"])
    w_margin = float(np.clip(b[2], 0, 1))
    blend = mkt + w_margin * (model_margin[ok] - mkt)

    okt = bt["total_line"].notna().to_numpy()
    mt = bt.loc[okt, "total_line"].to_numpy()
    Xt = np.column_stack([np.ones(okt.sum()), mt, model_total[okt] - mt])
    bt_, set_, _ = ols(Xt, bt.loc[okt, "total"])
    w_total = float(np.clip(bt_[2], 0, 1))
    blend_t = mt + w_total * (model_total[okt] - mt)

    actual = bt.loc[ok, "result"].to_numpy()
    actual_t = bt.loc[okt, "total"].to_numpy()
    return {
        "scale": float(scale), "hfa": float(hfa), "total_scale": float(tscale),
        "margin_sd_model": float(res.std(ddof=2)), "total_sd_model": float(tres.std(ddof=1)),
        "margin_sd": float((actual - blend).std(ddof=1)),
        "total_sd": float((actual_t - blend_t).std(ddof=1)),
        "w_margin": w_margin, "w_margin_raw": float(b[2]), "w_margin_se": float(se[2]),
        "w_total": w_total, "w_total_raw": float(bt_[2]), "w_total_se": float(set_[2]),
        "mae_margin_model": float(np.abs(actual - model_margin[ok]).mean()),
        "mae_margin_market": float(np.abs(actual - mkt).mean()),
        "mae_margin_blend": float(np.abs(actual - blend).mean()),
        "mae_total_model": float(np.abs(actual_t - model_total[okt]).mean()),
        "mae_total_market": float(np.abs(actual_t - mt).mean()),
        "mae_total_blend": float(np.abs(actual_t - blend_t).mean()),
        "n": int(len(bt)),
        "model_margin": model_margin, "model_total": model_total,
    }


def qb_change_effect(bt, fit):
    """Average margin shortfall vs the model when a team changes QB."""
    nn = (~bt["neutral"]).astype(float).to_numpy()
    resid = bt["result"].to_numpy() - fit["model_margin"]
    mk_resid = bt["result"].to_numpy() - bt["spread_line"].to_numpy()
    vals, mvals = [], []
    for side, sign in (("home", 1), ("away", -1)):
        m = bt[f"{side}_qb_changed"].to_numpy()
        vals += list(sign * resid[m])
        mvals += list(sign * mk_resid[m])
    vals, mvals = np.array(vals), np.array(mvals)
    n = len(vals)
    return {
        "n": n,
        "model_shortfall": float(vals.mean()) if n else 0.0,
        "model_se": float(vals.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0,
        "market_shortfall": float(mvals.mean()) if n else 0.0,
    }


def cv_mae(bt):
    """Leave-one-season-out: fit scale/hfa on other seasons, score on this one."""
    errs = []
    for s in bt["season"].unique():
        tr, te = bt[bt["season"] != s], bt[bt["season"] == s]
        nn_tr = (~tr["neutral"]).astype(float)
        beta, _, _ = ols(np.column_stack([tr["epa_diff"], nn_tr]), tr["result"])
        nn_te = (~te["neutral"]).astype(float)
        pred = beta[0] * te["epa_diff"] + beta[1] * nn_te
        errs.append((te["result"] - pred).abs())
    return float(pd.concat(errs).mean())


def simulate_bets(bt, fit, use_blend=True, thresholds=(0.0, 0.03, 0.05)):
    """Bet every game where the model sees an edge over the closing price.
    Returns a table of results by market and edge threshold (pushes skipped)."""
    nn = (~bt["neutral"]).astype(float).to_numpy()
    mm = fit["scale"] * bt["epa_diff"].to_numpy() + fit["hfa"] * nn
    mt = fit["model_total"]
    spread = bt["spread_line"].to_numpy(float)
    tline = bt["total_line"].to_numpy(float)
    if use_blend:
        mm = np.where(np.isnan(spread), mm, spread + fit["w_margin"] * (mm - spread))
        mt = np.where(np.isnan(tline), mt, tline + fit["w_total"] * (mt - tline))
        msd, tsd = fit["margin_sd"], fit["total_sd"]
    else:
        msd, tsd = fit["margin_sd_model"], fit["total_sd_model"]

    def dec(col):
        return np.nan_to_num(american_to_decimal(bt[col].to_numpy(float)), nan=1.909)

    result, total = bt["result"].to_numpy(), bt["total"].to_numpy()
    out = []

    def run(market, p_a, dec_a, dec_b, win_a, push):
        edge_a = p_a - 1 / dec_a
        edge_b = (1 - p_a) - 1 / dec_b
        for thr in thresholds:
            profit, n, wins = 0.0, 0, 0
            for i in range(len(p_a)):
                if np.isnan(p_a[i]) or push[i]:
                    continue
                if edge_a[i] >= edge_b[i] and edge_a[i] > thr:
                    won, d = win_a[i], dec_a[i]
                elif edge_b[i] > edge_a[i] and edge_b[i] > thr:
                    won, d = (not win_a[i]), dec_b[i]
                else:
                    continue
                n += 1
                wins += int(won)
                profit += (d - 1) if won else -1
            out.append({"market": market, "min_edge": f"{thr:.0%}", "bets": n,
                        "win_rate": wins / n if n else np.nan,
                        "roi": profit / n if n else np.nan})

    # Spread: side A = home
    p_home_cover = 1 - norm_cdf_arr((spread - mm) / msd)
    run("Spread", p_home_cover, dec("home_sp_odds"), dec("away_sp_odds"),
        result > spread, result == spread)
    # Total: side A = over
    p_over = 1 - norm_cdf_arr((tline - mt) / tsd)
    run("Total", p_over, dec("over_odds"), dec("under_odds"),
        total > tline, total == tline)
    # Moneyline: side A = home
    p_home = 1 - norm_cdf_arr(-mm / msd)
    run("Moneyline", p_home, dec_ml(bt["home_ml"]), dec_ml(bt["away_ml"]),
        result > 0, result == 0)
    return pd.DataFrame(out)


def dec_ml(series):
    return np.nan_to_num(american_to_decimal(series.to_numpy(float)), nan=2.0)


def calibration_table(bt, fit, bins=(0, .3, .4, .5, .6, .7, 1.0)):
    """Model home-win probability vs what actually happened, plus the
    market's de-vigged probability for comparison."""
    nn = (~bt["neutral"]).astype(float).to_numpy()
    mm = fit["scale"] * bt["epa_diff"].to_numpy() + fit["hfa"] * nn
    p = 1 - norm_cdf_arr(-mm / fit["margin_sd_model"])
    d = pd.DataFrame({
        "p": p, "win": (bt["result"] > 0).astype(float).to_numpy(),
        "tie": (bt["result"] == 0).to_numpy(),
    })
    d = d[~d["tie"]]
    d["bin"] = pd.cut(d["p"], list(bins))
    tab = d.groupby("bin", observed=True).agg(
        games=("win", "size"), model_pct=("p", "mean"), actual_pct=("win", "mean"))

    ph = 1 / dec_ml(bt["home_ml"])
    pa = 1 / dec_ml(bt["away_ml"])
    pm = ph / (ph + pa)
    keep = (bt["result"] != 0).to_numpy()
    y = (bt["result"] > 0).astype(float).to_numpy()[keep]
    brier = {
        "model": float(np.mean((p[keep] - y) ** 2)),
        "market": float(np.mean((pm[keep] - y) ** 2)),
    }
    return tab, brier
