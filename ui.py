"""Shared page setup and UI helpers used by every page."""
from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import streamlit as st

import nfl_model as nm
import props as pr
import tracker as tk

SEASON, PRIOR = nm.SEASON, nm.PRIOR_SEASON
ET, UK, UTC = ZoneInfo("America/New_York"), ZoneInfo("Europe/London"), ZoneInfo("UTC")

LEGEND = (
    "🟢 2-8 pts = believable value · 🟡 8-15 = treat with caution · "
    "🔴 15+ = almost certainly a bad input or stale price · ⚪ under 2 = no signal. "
    "A suspiciously high raw model probability is also flagged red on its own. "
    "Heads-up: the backtest found the model adds almost nothing beyond the market line, "
    "so on game markets Edge will mostly sit near zero (⚪). EV % is where price "
    "differences between bookmakers show up, so sort or filter by it."
)
METRIC_NOTE = ("Model %: our probability (starts from the market) · Fair %: bookmakers' "
               "de-vigged probability · Edge: model minus fair · EV %: expected return per "
               "unit stake at the best price.")

# Relative weights for splitting a bankroll across markets. Deliberately flat to
# start: unlike the MLB app there is no tracked NFL history yet. Update these
# from the Results page as real results and CLV come in.
TRUST_TIER = {"Anytime TD": 0.5}


# ---------- Page setup ----------
def setup_page(title, icon="🏈"):
    st.set_page_config(page_title=f"NFL Analyser - {title}", page_icon=icon, layout="centered")


@st.cache_data(ttl=1800, show_spinner="Projecting players...")
def cached_projection(week, _logs, _games):
    return pr.project_week(_logs, _games)


def get_context():
    """Week picker + stake in the sidebar, and everything the pages share."""
    sched = nm.get_schedule(SEASON)
    sched_all = pd.concat([nm.get_schedule(PRIOR), sched], ignore_index=True)
    reg = sched[sched["game_type"] == "REG"]
    unplayed = reg[reg["home_score"].isna()]
    current_week = int(unplayed["week"].min()) if len(unplayed) else int(reg["week"].max())
    weeks = sorted(int(w) for w in reg["week"].unique())
    week = st.sidebar.selectbox("Week", weeks, index=weeks.index(current_week), key="week")
    stake = st.sidebar.number_input("Stake for logged picks (£)", 0.5, 1000.0, 1.0, 0.5,
                                    key="stake")
    week_games = reg[reg["week"] == week]
    is_current = week == current_week

    tw = nm.load_team_weeks([PRIOR, SEASON])
    ratings = nm.ratings_asof(tw, SEASON, week)  # only games before the selected week
    lg_total = nm.league_avg_total(sched_all, SEASON, week)
    logs = pr.get_player_logs((PRIOR, SEASON))
    inj = pr.get_injuries(SEASON)
    inj_map = pr.injury_map(inj, week) if is_current else {}
    teams = sorted(set(week_games["home_team"]) | set(week_games["away_team"]))
    qb_out = pr.starting_qb_out(logs, inj_map, teams) if is_current else {}
    preds = nm.predict_games(week_games, ratings, lg_total, qb_out=qb_out)
    return SimpleNamespace(
        week=week, current_week=current_week, is_current=is_current, stake=stake,
        week_games=week_games, ratings=ratings, lg_total=lg_total, logs=logs, inj=inj,
        inj_map=inj_map, qb_out=qb_out, preds=preds, teams=teams)


def get_projection(ctx):
    proj = cached_projection(ctx.week, ctx.logs, ctx.week_games)
    return proj.assign(
        inj=[ctx.inj_map.get(p, ("", ""))[0] for p in proj["player_id"]],
        inj_note=[ctx.inj_map.get(p, ("", ""))[1] for p in proj["player_id"]])


# ---------- Times ----------
def kickoff(gameday, gametime):
    """(UTC datetime for sorting, UK time string, US-date string). nflverse times
    are US Eastern."""
    try:
        dt = datetime.strptime(f"{gameday} {gametime}", "%Y-%m-%d %H:%M").replace(tzinfo=ET)
    except (ValueError, TypeError):
        return datetime(2100, 1, 1, tzinfo=UTC), "TBC", str(gameday)
    return dt.astimezone(UTC), dt.astimezone(UK).strftime("%H:%M"), dt.strftime("%a %b %d")


def attach_times(df, week_games):
    df = df.copy()
    t = week_games.set_index("game_id")
    if "gameday" not in df.columns:
        df["gameday"] = df["game_id"].map(t["gameday"])
        df["gametime"] = df["game_id"].map(t["gametime"])
    ks = [kickoff(d, g) for d, g in zip(df["gameday"], df["gametime"])]
    df["_ct"] = [k[0] for k in ks]
    df["Start"] = [k[1] for k in ks]
    df["US Date"] = [k[2] for k in ks]
    return df


# ---------- Game filter ----------
def game_filter(ctx, key="gf"):
    """Checkbox per game inside an expander. Returns the selected game_ids.
    Shared across pages, so ticking games once applies everywhere."""
    games = ctx.week_games.sort_values(["gameday", "gametime"])
    for gid in games["game_id"]:
        st.session_state.setdefault(f"{key}_{gid}", True)

    def _set_all(value):
        for gid in games["game_id"]:
            st.session_state[f"{key}_{gid}"] = value

    with st.expander("🏟️ Filter games", expanded=False):
        c1, c2 = st.columns(2)
        c1.button("Select all", on_click=_set_all, args=(True,), key=f"{key}_all")
        c2.button("Clear all", on_click=_set_all, args=(False,), key=f"{key}_none")
        for _, g in games.iterrows():
            _, uk_time, us_date = kickoff(g["gameday"], g["gametime"])
            st.checkbox(f"{g['away_team']} @ {g['home_team']} · {uk_time} ({us_date})",
                        key=f"{key}_{g['game_id']}")
    selected = [gid for gid in games["game_id"] if st.session_state[f"{key}_{gid}"]]
    st.caption(f"{len(selected)} of {len(games)} games selected · times are UK, dates are US")
    return selected


# ---------- Cards ----------
def sort_picker(df, options, key):
    """options: list of (label, column, ascending); first is the default."""
    labels = [o[0] for o in options]
    choice = st.selectbox("Sort by", labels, key=key)
    _, col, asc = next(o for o in options if o[0] == choice)
    return df.sort_values(col, ascending=asc).reset_index(drop=True)


def light_header(df):
    n = df["light"].value_counts()
    st.markdown(f"### 🟢 {n.get('🟢', 0)} green · 🟡 {n.get('🟡', 0)} amber · "
                f"🔴 {n.get('🔴', 0)} red · ⚪ {n.get('⚪', 0)} no signal")


GAME_SORTS = [("Start time", "_ct", True), ("EV % (high to low)", "ev", False),
              ("Edge (high to low)", "edge", False),
              ("Model % (high to low)", "model", False), ("Odds (high to low)", "price", False)]


def bet_card(ctx, b, prefix, loggable=True):
    """One pick as a compact card, with a Log pick button."""
    with st.container(border=True):
        st.markdown(f"{b['light']} **{b['selection']} @ {b['price']:.2f}**")
        st.caption(f"{b['game']} · {b['Start']} ({b['US Date']})")
        st.caption(f"Model %: {b['model']:.1%} · Fair %: {b['fair']:.1%} · "
                   f"Edge: {b['edge'] * 100:+.1f} pts · EV %: {b['ev']:+.1%}")
        extra = f" · ⚠️ {b['inj_note']}" if b.get("inj_note") else ""
        st.caption(f"{b['reason']} · Best price: {b['book']}{extra}")
        if loggable and st.button("Log pick", key=f"{prefix}_log_{tk.pick_id(b)}"):
            try:
                ok = tk.log_pick(b.to_dict(), ctx.stake, SEASON, ctx.week)
                st.toast("Pick logged" if ok else "Already logged")
            except Exception as e:
                st.error(f"Could not save pick: {e}")


def pool_filter(df, mode):
    """Legs eligible for accumulators / suggestions."""
    if mode == "green":
        return df[df["light"] == "🟢"]
    if mode == "green_amber":
        return df[df["light"].isin(["🟢", "🟡"])]
    return df[(df["ev"] > 0) & (df["light"] != "🔴")]  # positive EV, plausible


POOL_LABELS = {
    "ev": "Positive EV (best price beats the market's fair price)",
    "green_amber": "Green + amber edges (2-15 pts)",
    "green": "Green edges only (2-8 pts)",
}


# ---------- Props fetching ----------
def fetch_props(ctx, sel_games, market_keys, regions):
    """Fetch prop prices for the selected games only. Returns (rows, credits left)."""
    if "odds" not in st.session_state:
        st.session_state["odds"] = nm.fetch_odds()
    events, _ = st.session_state["odds"]
    emap = nm.map_events_to_games(events, ctx.week_games)
    frames, rem = [], None
    for ev in events:
        gid = emap.get(ev["id"])
        if gid not in sel_games:
            continue
        try:
            data, rem = pr.fetch_event_props(
                ev["id"], [pr.MARKETS[k][1] for k in market_keys], regions)
            frames.append(pr.props_to_rows(data, gid))
        except Exception as e:
            st.warning(f"{ev['away_team']} @ {ev['home_team']}: {e}")
    po = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return po, rem


# ---------- Combos ----------
def best_combo(cands, n):
    """The n highest Model % picks from different games."""
    if cands is None or cands.empty:
        return None
    chosen, used = [], set()
    for _, r in cands.sort_values("model", ascending=False).iterrows():
        if r["game_id"] in used:
            continue
        chosen.append({
            "label": r["selection"], "game": r["game"], "game_id": r["game_id"],
            "odds": float(r["price"]), "model": float(r["model"]), "fair": float(r["fair"]),
            "reason": r["reason"], "market_key": r["market_key"], "side": r["side"],
            "line": r["line"], "player_id": r["player_id"], "player": r["player"]})
        used.add(r["game_id"])
        if len(chosen) == n:
            break
    if len(chosen) < n:
        return None
    return {"legs": chosen,
            "odds": float(np.prod([c["odds"] for c in chosen])),
            "prob": float(np.prod([c["model"] for c in chosen])),
            "fair": float(np.prod([c["fair"] for c in chosen]))}


def suggest_stakes(bankroll, markets):
    weights = {m: TRUST_TIER.get(m, 1.0) for m in markets}
    total = sum(weights.values())
    return {m: round(bankroll * w / total, 2) for m, w in weights.items()} if total else {}


STATUS_BADGE = {"WIN": "✅ Won", "LOSE": "❌ Lost", "PUSH": "➖ Push", "VOID": "↩️ Void",
                None: "⏳ Pending"}
COMBO_BADGE = {"won": "✅ Combo won!", "lost": "❌ Combo lost", "alive": "🟢 Still alive"}


def combo_status(combo):
    """Settles each leg against finished games / published player stats."""
    sched, stats = nm.get_schedule(SEASON), tk._player_stats(SEASON)
    legs = [tk.settle_one(pd.Series(leg), sched, stats) for leg in combo["legs"]]
    if "LOSE" in legs:
        overall = "lost"
    elif all(s in ("WIN", "PUSH", "VOID") for s in legs):
        overall = "won"
    else:
        overall = "alive"
    return overall, legs
