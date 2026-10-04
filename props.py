"""Player prop models: recency-weighted usage projections, touchdown
expectancy from field position, count distributions and prop odds matching."""
import math
import re

import numpy as np
import pandas as pd
import requests
import streamlit as st
import nflreadpy as nfl

import nfl_model as nm

# ---------- Settings (fitted on 2025 rolling backtest, see Backtest page) ----------
PP = {
    "n_hist": 8,          # games of history used
    "half_life": 4.0,     # recent games count more (weight halves every 4 games)
    "m_prior": 3.0,       # pseudo-games of position average blended in (volume stats)
    "m_prior_td": 3.0,    # same for touchdown rates
    "gamma": 0.0,         # game-environment scaling for volume stats (backtest: no help)
    "gamma_td": 0.5,      # game-environment scaling for touchdowns
    "td_mix": 0.7,        # weight on expected TDs (from field position) vs actual TDs
    "td_scale": 0.92,     # calibration multiplier on TD rate
}

# Negative-binomial dispersion per stat (bigger = closer to Poisson).
# For NORMAL_MARKETS it is the SD as a fraction of the projection instead.
DISPERSION = {
    "receptions": 7.0, "rec_yards": 1.5, "rush_yards": 2.0,
    "pass_yards": 0.45, "pass_tds": 15.0, "rush_rec_yards": 3.0,
}

# Markets modelled as a normal distribution instead of negative binomial; the
# DISPERSION value for these is the SD as a fraction of the projection.
NORMAL_MARKETS = {"pass_yards"}

# Multipliers correcting the rolling projections' average over-prediction
# (fitted on 2024-2025: players' quiet games often leave no stat row, which
# inflates receiving averages in particular)
BIAS = {
    "receptions": 0.90, "rec_yards": 0.90, "rush_yards": 0.985,
    "pass_yards": 0.95, "pass_tds": 0.97, "rush_rec_yards": 0.99,
}

# Model probability is pulled this far toward the bookmaker's price before an
# edge is shown (1 = trust the model fully). Tune from Results/CLV data.
PROP_TRUST = 0.5

# market key -> (label, Odds API key, stat column, usage column, min usage)
MARKETS = {
    "anytime_td": ("Anytime TD", "player_anytime_td", None, "touches", 3.0),
    "receptions": ("Receptions", "player_receptions", "receptions", "targets", 3.0),
    "rec_yards": ("Receiving yards", "player_reception_yds", "receiving_yards", "targets", 3.0),
    "rush_yards": ("Rushing yards", "player_rush_yds", "rushing_yards", "carries", 6.0),
    "pass_yards": ("Passing yards", "player_pass_yds", "passing_yards", "attempts", 15.0),
    "pass_tds": ("Passing TDs", "player_pass_tds", "passing_tds", "attempts", 15.0),
    "rush_rec_yards": ("Rush + rec yards", "player_rush_reception_yds",
                       "rush_rec_yards", "touches", 8.0),
}
API_TO_KEY = {v[1]: k for k, v in MARKETS.items()}

# Stricter bands than game markets: (green edge, amber edge, max plausible model %)
PROP_BANDS = {
    "anytime_td": (0.05, 0.025, 0.80), "receptions": (0.06, 0.03, 0.85),
    "rec_yards": (0.07, 0.035, 0.80), "rush_yards": (0.07, 0.035, 0.80),
    "pass_yards": (0.07, 0.035, 0.80), "pass_tds": (0.06, 0.03, 0.85),
    "rush_rec_yards": (0.07, 0.035, 0.80),
}

PROJ_STATS = ["receptions", "receiving_yards", "rushing_yards", "passing_yards",
              "passing_tds", "rush_rec_yards", "xtd", "td", "targets", "carries",
              "attempts", "touches"]
POSITIONS = ["QB", "RB", "WR", "TE"]


# ---------- Data ----------
@st.cache_data(ttl=3600, show_spinner="Loading player data...")
def get_player_logs(seasons):
    seasons = list(seasons)
    ps = nfl.load_player_stats(seasons).to_pandas()
    ps = ps[(ps["season_type"] == "REG") & ps["position"].isin(POSITIONS)].copy()
    ps = ps[["player_id", "player_display_name", "position", "team", "season", "week",
             "game_id", "opponent_team", "attempts", "carries", "targets", "receptions",
             "receiving_yards", "rushing_yards", "passing_yards", "passing_tds"]]
    ps["rush_rec_yards"] = ps["rushing_yards"] + ps["receiving_yards"]
    ps["touches"] = ps["carries"] + ps["targets"]

    # Team implied points per game from the closing lines
    sched = pd.concat([nm.get_schedule(s) for s in seasons])
    sched = sched[sched["game_type"] == "REG"]
    home = pd.DataFrame({"game_id": sched["game_id"], "team": sched["home_team"],
                         "implied": (sched["total_line"] + sched["spread_line"]) / 2})
    away = pd.DataFrame({"game_id": sched["game_id"], "team": sched["away_team"],
                         "implied": (sched["total_line"] - sched["spread_line"]) / 2})
    ps = ps.merge(pd.concat([home, away]), on=["game_id", "team"], how="left")

    # Touchdown expectancy from field position
    opps = pd.concat([nm.load_season_pbp(s)[1] for s in seasons], ignore_index=True)
    rates = opps.groupby(["kind", "bucket"]).agg(n=("n", "sum"), td=("td", "sum"))
    rates["rate"] = rates["td"] / rates["n"]
    o = opps.merge(rates["rate"].reset_index(), on=["kind", "bucket"])
    o["xtd"] = o["n"] * o["rate"]
    tdt = o.groupby(["season", "week", "player_id"]).agg(
        xtd=("xtd", "sum"), td=("td", "sum")).reset_index()
    ps = ps.merge(tdt, on=["season", "week", "player_id"], how="left")
    ps[["xtd", "td"]] = ps[["xtd", "td"]].fillna(0.0)
    ps["implied"] = ps["implied"].fillna(ps["implied"].mean())
    return ps.sort_values(["player_id", "season", "week"]).reset_index(drop=True)


def position_priors(logs):
    q = logs[(logs["touches"] >= 3) | (logs["attempts"] >= 10)]
    return q.groupby("position")[PROJ_STATS].mean()


# ---------- Rolling projections ----------
def _m_vec(p):
    m = np.full(len(PROJ_STATS), float(p["m_prior"]))
    for s_ in ("xtd", "td"):
        m[PROJ_STATS.index(s_)] = p["m_prior_td"]
    return m


def _project(vals, imp, prior, n_hist, half_life, m_vec):
    vals, imp = vals[-n_hist:], imp[-n_hist:]
    n = len(vals)
    w = 0.5 ** (np.arange(n - 1, -1, -1) / half_life)
    wm = (w[:, None] * vals).sum(0) / w.sum()
    env = float((w * imp).sum() / w.sum())
    mu = (n * wm + m_vec * prior) / (n + m_vec)
    return mu, env, n


def rolling_table(logs, params=None, priors=None):
    """For every player-game, project it using only earlier games."""
    p = params or PP
    logs = logs.reset_index(drop=True)
    priors = position_priors(logs) if priors is None else priors
    m_vec = _m_vec(p)
    out_mu = np.full((len(logs), len(PROJ_STATS)), np.nan)
    out_env = np.full(len(logs), np.nan)
    out_n = np.zeros(len(logs), dtype=int)
    pos_idx = {pos: priors.loc[pos].to_numpy(float) for pos in priors.index}
    for _, g in logs.groupby("player_id", sort=False):
        vals = g[PROJ_STATS].to_numpy(float)
        imp = g["implied"].to_numpy(float)
        prior = pos_idx[g["position"].iloc[-1]]
        idx = g.index.to_numpy()
        for i in range(1, len(g)):
            mu, env, n = _project(vals[:i], imp[:i], prior, p["n_hist"],
                                  p["half_life"], m_vec)
            out_mu[idx[i]], out_env[idx[i]], out_n[idx[i]] = mu, env, n
    res = logs.copy()
    for j, s in enumerate(PROJ_STATS):
        res[f"mu_{s}"] = out_mu[:, j]
    res["env_hist"] = out_env
    res["n_hist"] = out_n
    return res


# ---------- Distributions ----------
try:
    from scipy.special import gammaln as _gammaln
except Exception:  # slower fallback if scipy is not installed
    _gammaln = np.vectorize(math.lgamma, otypes=[float])


def nb_logpmf(y, mu, r):
    """Negative binomial log-probability. Larger r -> closer to Poisson."""
    y = np.maximum(np.nan_to_num(np.asarray(y, float)), 0.0)  # negative yards count as 0
    mu = np.maximum(np.asarray(mu, float), 1e-6)
    return (_gammaln(y + r) - _gammaln(r) - _gammaln(y + 1)
            + r * np.log(r / (r + mu)) + y * np.log(mu / (r + mu)))


def nb_cdf(k, mu, r):
    """P(X <= k) for a single mean."""
    if k < 0:
        return 0.0
    ks = np.arange(0, int(k) + 1)
    return float(min(1.0, np.exp(nb_logpmf(ks, mu, r)).sum()))


def over_under(mu, r, line, normal=False):
    """(P(win over), P(win under), P(push)) for a prop line."""
    if normal:
        p_over = 1 - nm.norm_cdf((line - mu) / (r * mu))
        return p_over, 1 - p_over, 0.0
    if line == int(line):  # whole-number line can push
        p_le = nb_cdf(line, mu, r)
        p_push = float(np.exp(nb_logpmf(line, mu, r)))
        return 1 - p_le, p_le - p_push, p_push
    p_le = nb_cdf(math.floor(line), mu, r)
    return 1 - p_le, p_le, 0.0


def td_prob(lam):
    return 1 - math.exp(-lam)


# ---------- Backtest ----------
def _env(imp_now, env_hist, gamma):
    ratio = np.where((env_hist > 0) & (imp_now > 0), imp_now / env_hist, 1.0)
    return np.clip(ratio, 0.6, 1.6) ** gamma


def stat_mu(roll, key, gamma):
    """Projected mean for a market: bias-corrected, with environment adjustment."""
    col = MARKETS[key][2]
    return (roll[f"mu_{col}"].to_numpy() * BIAS[key]
            * _env(roll["implied"].to_numpy(), roll["env_hist"].to_numpy(), gamma))


def td_lambda(roll, mix, gamma):
    base = mix * roll["mu_xtd"] + (1 - mix) * roll["mu_td"]
    return base.to_numpy() * _env(roll["implied"].to_numpy(),
                                  roll["env_hist"].to_numpy(), gamma)


def qualified(roll, key):
    usage_col, thr = MARKETS[key][3], MARKETS[key][4]
    return (roll["n_hist"] >= 2) & (roll[f"mu_{usage_col}"] >= thr)


R_GRID = [0.7, 1, 1.5, 2, 3, 4, 6, 8, 12, 20, 40, 100]
LINE_OFFSETS = (0.7, 0.85, 1.0, 1.15, 1.3)  # lines set at these multiples of the projection


def _cdf_matrix(mu, r, kmax):
    ks = np.arange(kmax + 1)
    lp = (_gammaln(ks[None, :] + r) - _gammaln(r) - _gammaln(ks[None, :] + 1)
          + r * np.log(r / (r + mu))[:, None] + ks[None, :] * np.log(mu / (r + mu))[:, None])
    return np.cumsum(np.exp(lp), axis=1)


def line_outcomes(y, mu, r, offsets=LINE_OFFSETS, normal=False):
    """For lines at several offsets around the projection, return
    (predicted P(over), actual over, offset) arrays stacked over offsets."""
    y = np.maximum(np.nan_to_num(np.asarray(y, float)), 0.0)
    mu = np.maximum(np.asarray(mu, float), 1e-6)
    cdf = None if normal else _cdf_matrix(mu, r, int(max(y.max(), mu.max() * 3)) + 5)
    rows = np.arange(len(mu))
    preds, hits, offs = [], [], []
    for f in offsets:
        line = np.floor(mu * f) + 0.5
        if normal:
            preds.append(1 - nm.norm_cdf_arr((line - mu) / (r * mu)))
        else:
            preds.append(1 - cdf[rows, np.floor(line).astype(int)])
        hits.append((y > line).astype(float))
        offs += [f] * len(mu)
    return np.concatenate(preds), np.concatenate(hits), np.array(offs)


SD_GRID = [0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6]


def best_dispersion(y, mu, normal=False):
    """Pick the dispersion (or SD fraction) that prices over/under lines best."""
    best, best_ll = None, -np.inf
    for r in (SD_GRID if normal else R_GRID):
        p, h, _ = line_outcomes(y, mu, r, normal=normal)
        p = np.clip(p, 1e-6, 1 - 1e-6)
        ll = float(np.mean(h * np.log(p) + (1 - h) * np.log(1 - p)))
        if ll > best_ll:
            best, best_ll = r, ll
    return best, best_ll


def calibrate_over(y, mu, r, normal=False):
    """Predicted vs actual P(over) for lines below/at/above the projection."""
    p, h, offs = line_outcomes(y, mu, r, normal=normal)
    d = pd.DataFrame({"line_vs_projection": offs, "p": p, "hit": h})
    t = d.groupby("line_vs_projection").agg(predicted=("p", "mean"), actual=("hit", "mean"))
    t.index = [f"{o:.0%} of projection" for o in t.index]
    return t


def td_logloss(y_any, lam, scale=1.0):
    p = np.clip(1 - np.exp(-scale * lam), 1e-6, 1 - 1e-6)
    return float(-(y_any * np.log(p) + (1 - y_any) * np.log(1 - p)).mean())


def prop_backtest(roll, season, params=None):
    """Evaluate the rolling model on one season. Returns per-market summary,
    calibration tables and the touchdown calibration."""
    p = params or PP
    d = roll[(roll["season"] == season)]
    rows, cals = [], {}
    for key, (label, _, col, _, _) in MARKETS.items():
        if key == "anytime_td":
            continue
        q = d[qualified(d, key)]
        mu = stat_mu(q, key, p["gamma"])
        y = q[col].to_numpy(float)
        nrm = key in NORMAL_MARKETS
        fit_r, _ = best_dispersion(y, mu, normal=nrm)
        rows.append({"market": label, "player_games": len(q), "avg_projection": mu.mean(),
                     "avg_actual": y.mean(), "mean_abs_error": float(np.abs(y - mu).mean()),
                     "fitted_dispersion": fit_r, "used_dispersion": DISPERSION[key]})
        cals[key] = calibrate_over(y, mu, DISPERSION[key], normal=nrm)
    q = d[qualified(d, "anytime_td")]
    lam = td_lambda(q, p["td_mix"], p["gamma_td"]) * p["td_scale"]
    y_any = (q["td"].to_numpy() >= 1).astype(float)
    base = np.full(len(lam), -math.log(1 - y_any.mean()))
    td = {"player_games": len(q), "avg_predicted": float((1 - np.exp(-lam)).mean()),
          "actual_rate": float(y_any.mean()), "logloss": td_logloss(y_any, lam),
          "logloss_base_rate": td_logloss(y_any, base)}
    tab = pd.DataFrame({"p": 1 - np.exp(-lam), "hit": y_any})
    tab["bin"] = pd.cut(tab["p"], [0, .1, .2, .3, .4, .5, .7, 1.0])
    td_cal = tab.groupby("bin", observed=True).agg(
        rows=("hit", "size"), predicted=("p", "mean"), actual=("hit", "mean"))
    return pd.DataFrame(rows), cals, td, td_cal


# ---------- Injuries ----------
@st.cache_data(ttl=1800)
def get_injuries(season):
    try:
        df = nfl.load_injuries([season]).to_pandas()
    except Exception:
        return pd.DataFrame()
    return df


def injury_map(inj, week):
    """gsis_id -> (status, note). Uses the requested week, else the latest."""
    if inj is None or inj.empty:
        return {}
    wk = inj[inj["week"] == week]
    if wk.empty:
        wk = inj[inj["week"] == inj["week"].max()]
    out = {}
    for _, r in wk.iterrows():
        rs, ps = r["report_status"], r["practice_status"]
        note = r["report_primary_injury"] if pd.notna(r["report_primary_injury"]) else \
            (r["practice_primary_injury"] if pd.notna(r["practice_primary_injury"]) else "")
        if pd.notna(rs) and rs in ("Out", "Doubtful"):
            out[r["gsis_id"]] = ("OUT", f"{rs} - {note}")
        elif pd.notna(rs) and rs == "Questionable":
            out[r["gsis_id"]] = ("Q", f"Questionable - {note}")
        elif pd.notna(ps) and str(ps).startswith("Did Not"):
            out[r["gsis_id"]] = ("DNP", f"Did not practise - {note}")
        elif pd.notna(ps) and str(ps).startswith("Limited"):
            out[r["gsis_id"]] = ("LP", f"Limited in practice - {note}")
    return out


def starting_qb_out(logs, inj_map, teams):
    """team -> True when the QB who threw most in that team's last game is out."""
    res = {}
    qbs = logs[logs["position"] == "QB"]
    for t in teams:
        tq = qbs[qbs["team"] == t]
        if tq.empty:
            continue
        latest = tq.sort_values(["season", "week"]).iloc[-1]
        last_game = tq[(tq["season"] == latest["season"]) & (tq["week"] == latest["week"])]
        starter = last_game.sort_values("attempts").iloc[-1]
        res[t] = inj_map.get(starter["player_id"], ("", ""))[0] == "OUT"
    return res


# ---------- Live projections ----------
def project_week(logs, week_games, roll_params=None):
    """Projections for every relevant skill player in this week's games."""
    p = roll_params or PP
    priors = position_priors(logs)
    pos_idx = {pos: priors.loc[pos].to_numpy(float) for pos in priors.index}
    m_vec = _m_vec(p)
    latest = logs.sort_values(["season", "week"]).groupby("player_id").tail(1)

    rows = []
    for _, g in week_games.iterrows():
        env_imp = {
            g["home_team"]: (g["total_line"] + g["spread_line"]) / 2,
            g["away_team"]: (g["total_line"] - g["spread_line"]) / 2,
        }
        for team, opp, is_home in ((g["home_team"], g["away_team"], True),
                                   (g["away_team"], g["home_team"], False)):
            tg = logs[logs["team"] == team][["season", "week"]].drop_duplicates()
            recent = tg.sort_values(["season", "week"]).tail(4)
            keys = set(zip(recent["season"], recent["week"]))
            cand = latest[latest["team"] == team]
            cand = cand[[(s, w) in keys for s, w in zip(cand["season"], cand["week"])]]
            for _, lr in cand.iterrows():
                hist = logs[logs["player_id"] == lr["player_id"]]
                mu, env_hist, n = _project(
                    hist[PROJ_STATS].to_numpy(float), hist["implied"].to_numpy(float),
                    pos_idx[lr["position"]], p["n_hist"], p["half_life"], m_vec)
                imp_now = env_imp[team]
                ratio = imp_now / env_hist if pd.notna(imp_now) and env_hist > 0 else 1.0
                ratio = float(np.clip(ratio, 0.6, 1.6))
                env, env_td = ratio ** p["gamma"], ratio ** p["gamma_td"]
                row = {
                    "game_id": g["game_id"], "player_id": lr["player_id"],
                    "player": lr["player_display_name"], "pos": lr["position"],
                    "team": team, "opp": opp, "home": g["home_team"], "away": g["away_team"],
                    "is_home": is_home, "n_hist": n, "env": env, "env_td": env_td,
                }
                for s in PROJ_STATS:
                    row[f"mu_{s}"] = mu[PROJ_STATS.index(s)]
                lam = (p["td_mix"] * row["mu_xtd"] + (1 - p["td_mix"]) * row["mu_td"]) \
                    * env_td * p["td_scale"]
                row["lam_td"] = lam
                row["td_prob"] = td_prob(lam)
                rows.append(row)
    return pd.DataFrame(rows)


def market_mu(row, key):
    """Projected mean for one player and market, with the environment factor."""
    return row[f"mu_{MARKETS[key][2]}"] * BIAS[key] * row["env"]


# ---------- Odds ----------
def norm_name(s):
    s = str(s).lower()
    s = re.sub(r"[.'`’-]", "", s)
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", s)
    return re.sub(r"\s+", " ", s).strip()


def fetch_event_props(event_id, api_markets, regions="uk"):
    resp = requests.get(f"{nm.ODDS_BASE}/events/{event_id}/odds", params={
        "apiKey": nm._odds_key(), "regions": regions,
        "markets": ",".join(api_markets), "oddsFormat": "decimal",
    }, timeout=20)
    resp.raise_for_status()
    return resp.json(), resp.headers.get("x-requests-remaining")


def props_to_rows(event, game_id):
    rows = []
    for bk in event.get("bookmakers", []):
        for mk in bk.get("markets", []):
            key = API_TO_KEY.get(mk["key"])
            if not key:
                continue
            for oc in mk.get("outcomes", []):
                rows.append({
                    "event_id": event["id"], "game_id": game_id, "book": bk["title"],
                    "market_key": key,
                    "player_norm": norm_name(oc.get("description", "")),
                    "player_raw": oc.get("description", ""),
                    "name": oc["name"], "point": oc.get("point"), "price": oc["price"],
                })
    return pd.DataFrame(rows)


def prop_light(key, prob, edge):
    green, amber, ceiling = PROP_BANDS[key]
    if prob > ceiling:
        return "🔴"
    if edge >= green:
        return "🟢"
    if edge >= amber:
        return "🟡"
    return "🔴"


def build_prop_bets(proj, odds, trust=None):
    """Price every listed prop against the model. Only the most common line
    per player/market is used, and the best price across bookmakers."""
    trust = PROP_TRUST if trust is None else trust
    if odds.empty or proj.empty:
        return pd.DataFrame()
    proj = proj.assign(player_norm=proj["player"].map(norm_name))
    out = []
    for (gid, pn, key), g in odds.groupby(["game_id", "player_norm", "market_key"]):
        cand = proj[(proj["game_id"] == gid) & (proj["player_norm"] == pn)]
        if cand.empty:
            continue
        pr = cand.iloc[0]
        label = MARKETS[key][0]

        def add(side, line, price, book, p_win, p_push=0.0):
            implied = 1 / price
            used = implied + trust * (p_win / (1 - p_push) - implied)
            edge = used - implied
            out.append({
                "game_id": pr["game_id"], "home": pr["home"], "away": pr["away"],
                "game": f"{pr['away']} @ {pr['home']}",
                "market": label, "market_key": key, "side": side, "line": line,
                "player_id": pr["player_id"], "player": pr["player"],
                "selection": f"{pr['player']} {label} " + (
                    "Yes" if side == "yes" else f"{side.title()} {line:g}"),
                "price": price, "book": book, "model_raw": p_win / (1 - p_push),
                "model": used, "implied": implied, "edge": edge,
                "ev": used * price - 1,
                "light": prop_light(key, used, edge),
            })

        if key == "anytime_td":
            yes = g[g["name"] == "Yes"]
            if yes.empty:
                continue
            best = yes.loc[yes["price"].idxmax()]
            add("yes", None, float(best["price"]), best["book"], pr["td_prob"])
            continue

        mu = market_mu(pr, key)
        r = DISPERSION[key]
        ov = g[g["name"] == "Over"]
        if ov.empty:
            continue
        line = float(ov["point"].mode().iloc[0])
        p_over, p_under, p_push = over_under(mu, r, line, normal=key in NORMAL_MARKETS)
        o = ov[ov["point"] == line]
        b = o.loc[o["price"].idxmax()]
        add("over", line, float(b["price"]), b["book"], p_over, p_push)
        un = g[(g["name"] == "Under") & (g["point"] == line)]
        if not un.empty:
            b = un.loc[un["price"].idxmax()]
            add("under", line, float(b["price"]), b["book"], p_under, p_push)
    return pd.DataFrame(out)
