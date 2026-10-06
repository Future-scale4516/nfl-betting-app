import pandas as pd
import streamlit as st

import props as pr
import ui

ui.setup_page("Player Props", "🎰")
ctx = ui.get_context()

st.title("🎰 Player Props")
st.caption("Projects every relevant skill player from recent usage (touchdowns also use "
           "field position and game environment), then prices the props against "
           "bookmakers. Props are checked against real outcomes in the backtest but not "
           "against bookmaker prices, so treat edges as leads to verify, and let CLV on "
           "the Results page tell you whether they hold up.")

if not ctx.is_current:
    st.info("Player props are only projected for the upcoming week. Switch the week in "
            "the sidebar.")
    st.stop()

sel_games = ui.game_filter(ctx)
proj = ui.get_projection(ctx)
active = proj[proj["inj"] != "OUT"]
st.caption(f"{len(active)} players projected · {len(proj) - len(active)} ruled out "
           "(Out/Doubtful) and excluded.")

TAB_LABELS = {
    "anytime_td": "🎯 Anytime TD", "receptions": "🙌 Receptions", "rec_yards": "📏 Rec Yds",
    "rush_yards": "🏃 Rush Yds", "pass_yards": "🎯 Pass Yds", "pass_tds": "🏈 Pass TDs",
    "rush_rec_yards": "➕ Rush+Rec",
}

# ---------- Edges ----------
st.markdown("## 🎰 Player Prop Edges")
fetch_mk = st.multiselect("Markets to fetch", list(pr.MARKETS),
                          default=["anytime_td", "receptions", "rec_yards"],
                          format_func=lambda k: pr.MARKETS[k][0])
regions = st.radio("Bookmaker regions", ["uk", "uk,us"], horizontal=True,
                   help="US books list more props but cost double and can't be bet from "
                        "the UK.")
cost = len(sel_games) * len(fetch_mk) * len(regions.split(","))
st.caption(f"Projected cost: about {cost} Odds API credits for the selected games.")

if st.button("Find player prop edges") and fetch_mk and sel_games:
    try:
        with st.spinner("Fetching prop prices for the selected games..."):
            po, rem = ui.fetch_props(ctx, sel_games, fetch_mk, regions)
        if po.empty:
            st.warning("No prop prices came back. UK bookmakers often have few NFL props in "
                       "the API; try the uk,us region to see what's listed.")
            st.session_state.pop("prop_edges", None)
        else:
            bets = pr.build_prop_bets(active, po)
            if bets.empty:
                st.warning("Prices came back but none matched projected players.")
                st.session_state.pop("prop_edges", None)
            else:
                # Fetch + store only; display lives in the persistent block below.
                st.session_state["prop_edges"] = ui.attach_times(bets, ctx.week_games)
                st.session_state["prop_edges_week"] = ctx.week
                st.session_state["prop_edges_remaining"] = rem
    except Exception as e:
        st.error(f"Prop odds fetch failed: {e}")

pe = st.session_state.get("prop_edges")
if isinstance(pe, pd.DataFrame) and not pe.empty \
        and st.session_state.get("prop_edges_week") == ctx.week:
    pe = pe[pe["game_id"].isin(sel_games)]
    if st.session_state.get("prop_edges_remaining"):
        st.caption(f"Odds API credits remaining: {st.session_state['prop_edges_remaining']}")
    if pe.empty:
        st.info("None of the fetched props are in the selected games.")
    else:
        ui.light_header(pe)
        st.caption(ui.LEGEND)
        st.caption(f"Model % is pulled {1 - pr.PROP_TRUST:.0%} of the way back to the "
                   "bookmaker's fair price before edges are shown (props are unvalidated "
                   "against prices). Anytime TD is one-sided, so its Fair % is the "
                   "price-implied probability, vig included.")
        only_pos = st.checkbox("Only positive EV", key="p_pos_ev")
        view = pe[pe["ev"] > 0] if only_pos else pe
        keys = [k for k in pr.MARKETS if k in set(view["market_key"])] or \
               [k for k in pr.MARKETS if k in set(pe["market_key"])]
        tabs = st.tabs([TAB_LABELS[k] for k in keys])
        for tab, key in zip(tabs, keys):
            with tab:
                sub = view[view["market_key"] == key]
                if sub.empty:
                    st.write("No selections for this market with the current filters.")
                    continue
                sub = ui.sort_picker(sub, ui.GAME_SORTS, key=f"sort_p_{key}")
                if len(sub) > 60:
                    st.caption(f"Showing the first 60 of {len(sub)}. Use the game filter, "
                               "sort order or positive-EV box to narrow.")
                for _, b in sub.head(60).iterrows():
                    ui.bet_card(ctx, b, f"p_{key}")
        st.caption(ui.METRIC_NOTE)

# ---------- Most likely ----------
st.markdown("## 🔮 Most Likely — best players to achieve a market")
st.caption("Pure model view for the selected games, no odds needed. Anytime TD ranks "
           "chance of 1+ TD; the yardage and reception markets rank chance of going over "
           "the line you pick.")
tabs = st.tabs([TAB_LABELS[k] for k in pr.MARKETS])
active_sel = active[active["game_id"].isin(sel_games)]
for tab, key in zip(tabs, pr.MARKETS):
    with tab:
        line = None
        if key != "anytime_td":
            line = st.selectbox("Line", pr.STD_LINES[key], key=f"ml_line_{key}",
                                format_func=lambda x: f"Over {x:g}")
        top = pr.most_likely(active_sel, key, line).head(30)
        if top.empty:
            st.write("No players meet the usage threshold in the selected games.")
            continue
        for _, r in top.iterrows():
            with st.container(border=True):
                flag = f" ⚠️ {r['inj_note']}" if r["inj"] else ""
                st.markdown(f"**{r['player']}** · {r['pos']} {r['team']} vs {r['opp']}{flag}")
                if key == "anytime_td":
                    st.caption(f"Model %: {r['prob']:.1%} · Fair odds: {1 / r['prob']:.2f} · "
                               f"Expected TDs {r['proj']:.2f} · {r['n_hist']} games of history")
                else:
                    st.caption(f"Model % over {line:g}: {r['prob']:.1%} · Projection "
                               f"{r['proj']:.1f} · {r['n_hist']} games of history")
