import numpy as np
import pandas as pd
import streamlit as st

import nfl_model as nm
import props as pr
import tracker as tk

st.set_page_config(page_title="NFL Betting Analysis", layout="centered")
st.title("NFL Betting Analysis")

SEASON, PRIOR = nm.SEASON, nm.PRIOR_SEASON

# ---------- Week selection ----------
sched = nm.get_schedule(SEASON)
sched_all = pd.concat([nm.get_schedule(PRIOR), sched], ignore_index=True)
reg = sched[sched["game_type"] == "REG"]
unplayed = reg[reg["home_score"].isna()]
current_week = int(unplayed["week"].min()) if len(unplayed) else int(reg["week"].max())
weeks = sorted(int(w) for w in reg["week"].unique())
week = st.selectbox("Week", weeks, index=weeks.index(current_week))
week_games = reg[reg["week"] == week]
is_current = week == current_week

# ---------- Model inputs ----------
tw = nm.load_team_weeks([PRIOR, SEASON])
ratings = nm.ratings_asof(tw, SEASON, week)  # only games before the selected week
lg_total = nm.league_avg_total(sched_all, SEASON, week)

logs = pr.get_player_logs((PRIOR, SEASON))
inj = pr.get_injuries(SEASON)
inj_map = pr.injury_map(inj, week) if is_current else {}
teams = sorted(set(week_games["home_team"]) | set(week_games["away_team"]))
qb_out = pr.starting_qb_out(logs, inj_map, teams) if is_current else {}
preds = nm.predict_games(week_games, ratings, lg_total, qb_out=qb_out)


@st.cache_data(ttl=1800, show_spinner="Projecting players...")
def cached_projection(week, _logs, _games):
    return pr.project_week(_logs, _games)


LEGEND = (
    "🟢 3+ pts = believable value (moneyline needs 4+) · 🟡 1.5-3 = treat with caution · "
    "🔴 under 1.5 = no signal. A suspiciously high raw model probability is also flagged "
    "red on its own, even with a small edge. Props use slightly wider bands. Edge is in "
    "percentage points of probability; Implied % is 1 / price, so it includes the "
    "bookmaker's margin."
)


def render_bets(bets, prefix, stake, default_pos=True, reasons=None):
    """Sortable edge cards with a Log pick button on each."""
    if bets.empty:
        st.warning("No priced selections found.")
        return
    bets = bets.copy()
    if "gameday" not in bets.columns:
        t = week_games.set_index("game_id")
        bets["gameday"] = bets["game_id"].map(t["gameday"])
        bets["gametime"] = bets["game_id"].map(t["gametime"])
    bets["start"] = bets["gameday"].astype(str) + " " + bets["gametime"].astype(str)
    only_pos = st.checkbox("Only positive edges", value=default_pos, key=f"{prefix}_pos")
    sort_by = st.selectbox("Sort by", ["Start time", "Edge", "Model %", "EV", "Price"],
                           key=f"{prefix}_sort")
    col = {"Start time": "start", "Edge": "edge", "Model %": "model",
           "EV": "ev", "Price": "price"}[sort_by]
    view = bets[bets["edge"] > 0] if only_pos else bets
    if view.empty:
        st.info("No positive edges right now.")
    for _, b in view.sort_values(col, ascending=(col == "start")).iterrows():
        day = pd.to_datetime(b["gameday"]).strftime("%a %b %d")
        with st.container(border=True):
            st.markdown(f"{b['light']} **{b['selection']} @ {b['price']:.2f}**")
            st.caption(f"{b['game']} · {b['gametime']} ({day})")
            st.caption(f"Model %: {b['model']:.1%} · Implied %: {b['implied']:.1%} · "
                       f"Edge: {b['edge'] * 100:+.1f} pts · EV %: {b['ev']:+.1%}")
            why = (reasons or {}).get(b["game_id"], "")
            st.caption(f"Best price: {b['book']}" + (f" · {why}" if why else ""))
            if st.button("Log pick", key=f"{prefix}_log_{tk.pick_id(b)}"):
                try:
                    ok = tk.log_pick(b.to_dict(), stake, SEASON, week)
                    st.toast("Pick logged" if ok else "Already logged")
                except Exception as e:
                    st.error(f"Could not save pick: {e}")


tab_model, tab_bets, tab_props, tab_inj, tab_rat = st.tabs(
    ["Model", "Game Bets", "Player Props", "Injuries", "Team Ratings"])

# ---------- Model ----------
with tab_model:
    st.caption(
        f"League average total used: {lg_total:.1f} · Backtest weight on the model vs the "
        f"market line: {nm.PARAMS['w_margin']:.0%} (margins), {nm.PARAMS['w_total']:.0%} (totals)"
    )
    for _, p in preds.iterrows():
        with st.container(border=True):
            title = f"**{p['away']} @ {p['home']}**"
            if p["neutral"]:
                title += " · neutral site"
            st.markdown(title)
            st.caption(
                f"{p['gameday']} {p['gametime']} · "
                f"Model: {nm.fmt_spread(p['home'], p['away'], p['raw_margin'])}, "
                f"total {p['raw_total']:.1f} · "
                f"Market: {nm.fmt_spread(p['home'], p['away'], p['mkt_spread'])}, "
                f"total {p['mkt_total']} · "
                f"Used for bets: {nm.fmt_spread(p['home'], p['away'], p['model_margin'])}, "
                f"{p['home']} win {p['home_win_prob']:.0%}"
            )
            flags = [f"{t} starting QB out" for t, on in
                     ((p["home"], p["qb_out_home"]), (p["away"], p["qb_out_away"])) if on]
            if flags:
                st.caption("⚠️ " + " · ".join(flags))
            if pd.notna(p["home_score"]):
                st.caption(f"Final: {p['away']} {p['away_score']:.0f} - "
                           f"{p['home_score']:.0f} {p['home']}")

# ---------- Game bets ----------
with tab_bets:
    st.caption(LEGEND)
    st.caption(
        "The 2024-25 backtest found the model does not beat closing lines, so bets are "
        "priced off a mostly market-based line. Edges here are mainly price differences "
        "between bookmakers. See the Backtest page."
    )
    reasons = {
        p["game_id"]: f"Line used {nm.fmt_spread(p['home'], p['away'], p['model_margin'])}, "
                      f"total {p['model_total']:.1f} (market "
                      f"{nm.fmt_spread(p['home'], p['away'], p['mkt_spread'])}, "
                      f"{p['mkt_total']})"
        for _, p in preds.iterrows()
    }
    if is_current:
        stake = st.number_input("Stake for logged picks (£)", 0.5, 1000.0, 1.0, 0.5, key="stake")
        c1, c2 = st.columns(2)
        if c1.button("Fetch game odds"):
            try:
                st.session_state["odds"] = nm.fetch_odds()
            except Exception as e:
                st.error(f"Odds fetch failed: {e}")
        if c2.button("Force refresh"):
            try:
                nm.fetch_odds.clear()
                st.session_state["odds"] = nm.fetch_odds()
            except Exception as e:
                st.error(f"Odds fetch failed: {e}")

    t_ml, t_sp, t_tot, t_likely = st.tabs(
        ["💰 Money Line", "📏 Spread", "📊 Totals", "🎯 Most Likely"])

    game_bets = pd.DataFrame()
    if is_current and "odds" in st.session_state:
        events, remaining = st.session_state["odds"]
        odds = nm.odds_to_rows(events)
        if not odds.empty:
            game_bets = nm.build_bets(preds, odds)

    for tab, market, prefix in ((t_ml, "Moneyline", "g_ml"), (t_sp, "Spread", "g_sp"),
                                (t_tot, "Total", "g_tot")):
        with tab:
            if not is_current:
                st.info("Live odds are only available for the upcoming week.")
            elif game_bets.empty:
                st.info("Press Fetch game odds above.")
            else:
                render_bets(game_bets[game_bets["market"] == market], prefix, stake,
                            default_pos=False, reasons=reasons)

    with t_likely:
        st.caption("Highest model win probabilities this week (no odds needed).")
        likely = preds.assign(
            fav=lambda d: np.where(d["home_win_prob"] >= 0.5, d["home"], d["away"]),
            prob=lambda d: np.maximum(d["home_win_prob"], 1 - d["home_win_prob"]))
        for _, p in likely.sort_values("prob", ascending=False).iterrows():
            with st.container(border=True):
                st.markdown(f"**{p['fav']} to win · {p['prob']:.0%}**")
                st.caption(f"{p['away']} @ {p['home']} · {p['gametime']} "
                           f"({pd.to_datetime(p['gameday']).strftime('%a %b %d')})")
                st.caption(reasons[p["game_id"]])

# ---------- Player props ----------
with tab_props:
    if not is_current:
        st.info("Player props are only projected for the upcoming week.")
    else:
        proj = cached_projection(week, logs, week_games)
        status = [inj_map.get(pid, ("", ""))[0] for pid in proj["player_id"]]
        note = [inj_map.get(pid, ("", ""))[1] for pid in proj["player_id"]]
        proj = proj.assign(inj=status, inj_note=note)
        active = proj[proj["inj"] != "OUT"]
        st.caption(f"{len(active)} players projected · {len(proj) - len(active)} ruled out "
                   "(Out/Doubtful) and excluded.")

        mk = st.selectbox("Market", list(pr.MARKETS), key="props_market",
                          format_func=lambda k: pr.MARKETS[k][0])
        label, _, col, usage_col, thr = pr.MARKETS[mk]

        # Model view (no odds needed)
        st.markdown("**Model projections**")
        v = active[active[f"mu_{usage_col}"] >= thr].copy()
        if mk == "anytime_td":
            v["rank_val"] = v["td_prob"]
        else:
            v["rank_val"] = [pr.market_mu(r, mk) for _, r in v.iterrows()]
        for _, r in v.sort_values("rank_val", ascending=False).head(25).iterrows():
            with st.container(border=True):
                flag = f" ⚠️ {r['inj_note']}" if r["inj"] else ""
                st.markdown(f"**{r['player']}** · {r['pos']} {r['team']} vs {r['opp']}{flag}")
                if mk == "anytime_td":
                    st.caption(f"Model {r['td_prob']:.0%} · fair odds {1 / r['td_prob']:.2f} · "
                               f"{r['n_hist']} games of history")
                else:
                    st.caption(f"Projection {r['rank_val']:.1f} · {r['n_hist']} games of history")

        # Prop odds
        st.markdown("**Price against bookmakers**")
        stake_p = st.number_input("Stake for logged picks (£)", 0.5, 1000.0, 1.0, 0.5,
                                  key="stake_props")
        fetch_mk = st.multiselect("Markets to fetch", list(pr.MARKETS),
                                  default=["anytime_td", "receptions", "rec_yards"],
                                  format_func=lambda k: pr.MARKETS[k][0])
        regions = st.radio("Bookmaker regions", ["uk", "uk,us"], horizontal=True,
                           help="US books list more props but cost double and can't be bet from the UK.")
        n_games = len(week_games)
        n_reg = len(regions.split(","))
        st.caption(f"Costs about {n_games * len(fetch_mk) * n_reg} Odds API credits.")

        if st.button("Fetch prop odds") and fetch_mk:
            try:
                events, _ = st.session_state.get("odds") or nm.fetch_odds()
                emap = nm.map_events_to_games(events, week_games)
                frames, rem = [], None
                with st.spinner("Fetching prop odds..."):
                    for ev in events:
                        gid = emap.get(ev["id"])
                        if not gid:
                            continue
                        try:
                            data, rem = pr.fetch_event_props(
                                ev["id"], [pr.MARKETS[k][1] for k in fetch_mk], regions)
                            frames.append(pr.props_to_rows(data, gid))
                        except Exception as e:
                            st.warning(f"{ev['away_team']} @ {ev['home_team']}: {e}")
                po = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
                st.session_state["prop_odds"] = (po, rem)
            except Exception as e:
                st.error(f"Prop odds fetch failed: {e}")

        if "prop_odds" in st.session_state:
            po, rem = st.session_state["prop_odds"]
            st.caption(f"Odds API credits remaining: {rem}")
            if po.empty:
                st.warning("No prop prices came back. UK bookmakers often have few NFL props "
                           "in the API; try the uk,us region to see what's listed.")
            else:
                bets = pr.build_prop_bets(active, po)
                if not bets.empty:
                    bets = bets[bets["market_key"] == mk]
                st.caption(f"Model probability is pulled {1 - pr.PROP_TRUST:.0%} of the way "
                           "toward the bookmaker price before edges are shown.")
                render_bets(bets, "p", stake_p)

# ---------- Injuries ----------
with tab_inj:
    if not is_current:
        st.info("Injury reports are shown for the upcoming week.")
    elif inj.empty:
        st.warning("Injury data is unavailable from nflverse right now.")
    else:
        for t, on in qb_out.items():
            if on:
                st.error(f"{t}: starting QB is ruled out. The model penalises {t} by "
                         f"{nm.PARAMS['qb_adj']:.1f} points.")
        wk = inj[(inj["week"] == week) & inj["team"].isin(teams)
                 & inj["position"].isin(["QB", "RB", "WR", "TE"])]
        if wk.empty:
            st.info("No skill-position injury entries for these games yet.")
        else:
            rows = []
            for _, r in wk.iterrows():
                s, n = inj_map.get(r["gsis_id"], ("", ""))
                if s:
                    rows.append({"team": r["team"], "player": r["full_name"],
                                 "pos": r["position"], "status": s, "detail": n})
            if rows:
                order = {"OUT": 0, "Q": 1, "DNP": 2, "LP": 3}
                df = pd.DataFrame(rows)
                df["o"] = df["status"].map(order)
                st.dataframe(df.sort_values(["o", "team"]).drop(columns="o"), hide_index=True)
                st.caption("OUT = Out or Doubtful · Q = Questionable · DNP/LP = did not "
                           "participate / limited in practice. Game-day statuses appear "
                           "later in the week.")
            else:
                st.info("Nobody flagged so far this week.")

# ---------- Ratings ----------
with tab_rat:
    st.dataframe(ratings.round(3).reset_index(), hide_index=True)
