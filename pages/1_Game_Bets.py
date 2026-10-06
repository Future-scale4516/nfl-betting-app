import numpy as np
import pandas as pd
import streamlit as st

import nfl_model as nm
import ui

ui.setup_page("Game Bets", "🎯")
ctx = ui.get_context()

st.markdown("## 🎯 Game Bets — Money Line · Spread · Totals")
st.caption("Starts from the bookmakers' de-vigged UK prices, then moves a small, backtested "
           "step toward the model (the backtest found the model adds little beyond the "
           "market line). Positive EV % means the best available price beats the market's "
           "fair price. Always confirm the live price at your book before staking — "
           "lines move.")

if not ctx.is_current:
    st.info("Live odds are only available for the upcoming week. Switch the week in the "
            "sidebar.")
    st.stop()

sel_games = ui.game_filter(ctx)


def analyse(force=False):
    try:
        if force:
            nm.fetch_odds.clear()
        events, remaining = nm.fetch_odds()
    except Exception as e:
        st.error(f"Odds fetch failed: {e}")
        return
    st.session_state["odds"] = (events, remaining)
    odds = nm.odds_to_rows(events)
    bets = nm.build_bets(ctx.preds, odds) if not odds.empty else pd.DataFrame()
    if bets.empty:
        st.warning("No game odds available right now.")
        st.session_state.pop("game_edges", None)
        return
    # Fetch + store only. The display lives in the persistent block below,
    # gated on session_state, so sort/filter widgets don't make the list vanish.
    st.session_state["game_edges"] = ui.attach_times(bets, ctx.week_games)
    st.session_state["game_edges_week"] = ctx.week
    st.session_state["game_edges_remaining"] = remaining


c1, c2 = st.columns(2)
if c1.button("Analyse game bets (UK odds)"):
    analyse()
if c2.button("Refresh odds"):
    analyse(force=True)
st.caption("Odds are cached for 30 minutes to protect your quota; Refresh odds bypasses "
           "the cache.")

gdf = st.session_state.get("game_edges")
if isinstance(gdf, pd.DataFrame) and not gdf.empty \
        and st.session_state.get("game_edges_week") == ctx.week:
    gdf = gdf[gdf["game_id"].isin(sel_games)]
    if st.session_state.get("game_edges_remaining"):
        st.caption(f"Odds API credits remaining: {st.session_state['game_edges_remaining']}")
    if gdf.empty:
        st.info("No games selected, or none priced yet. Tick some games in the filter.")
        st.stop()

    ui.light_header(gdf)
    st.caption(ui.LEGEND)
    only_pos = st.checkbox("Only positive EV (price-shopping view)", key="g_pos_ev")
    view = gdf[gdf["ev"] > 0] if only_pos else gdf

    def show_market(tab, market_name, prefix):
        with tab:
            sub = view[view["market"] == market_name]
            if sub.empty:
                st.write("No selections for this market with the current filters.")
                return
            sub = ui.sort_picker(sub, ui.GAME_SORTS, key=f"sort_{prefix}")
            for _, b in sub.iterrows():
                ui.bet_card(ctx, b, prefix)

    def show_most_likely(tab):
        with tab:
            st.caption("Highest model probabilities, ignoring price. Remember Model % here "
                       "is mostly the market's own view, so this is 'what the market and "
                       "model both expect', not 'where's the value'. A high % is not a "
                       "guarantee.")
            picks = st.multiselect("Markets to include:", ["Moneyline", "Spread", "Total"],
                                   default=["Moneyline", "Spread", "Total"], key="ml_pick")
            sub = gdf[gdf["market"].isin(picks)]
            if sub.empty:
                st.write("No selections match the chosen markets.")
                return
            sub = ui.sort_picker(sub, [("Model % (high to low)", "model", False),
                                       ("Start time", "_ct", True)], key="sort_likely")
            for _, b in sub.iterrows():
                with st.container(border=True):
                    st.markdown(f"**{b['selection']} · {b['market']}**")
                    st.caption(f"{b['game']} · {b['Start']} ({b['US Date']})")
                    st.caption(f"Model %: {b['model']:.1%}")

    ml_tab, sp_tab, tot_tab, likely_tab = st.tabs(
        ["💰 Money Line", "📏 Spread", "📊 Totals", "🎯 Most Likely"])
    show_market(ml_tab, "Moneyline", "g_ml")
    show_market(sp_tab, "Spread", "g_sp")
    show_market(tot_tab, "Total", "g_tot")
    show_most_likely(likely_tab)
    st.caption(ui.METRIC_NOTE)

    # ---------- Accumulator builder ----------
    st.markdown("### 🎟️ Accumulator builder")
    st.caption("Build a multi-fold from the selections in your chosen pool. Combined odds, "
               "the model's chance of every leg landing, and what the bet is worth if the "
               "market is right are worked out for you.")
    mode = st.radio("Leg pool", list(ui.POOL_LABELS), format_func=ui.POOL_LABELS.get,
                    key="acca_pool")
    pool = ui.pool_filter(gdf, mode).sort_values("ev", ascending=False).reset_index(drop=True)
    if pool.empty:
        st.write("No selections in this pool on the analysed slate. Try the "
                 "positive-EV pool.")
    else:
        pool["pick"] = pool.apply(
            lambda r: f"{r['selection']} @ {r['price']:.2f} · {r['market']} · "
                      f"{r['game']} (EV {r['ev']:+.1%})", axis=1)
        chosen = st.multiselect("Choose your legs:", pool["pick"].tolist())
        stake = st.number_input("Stake (£)", min_value=0.0, value=5.0, step=0.5,
                                key="acca_stake")
        if chosen:
            sel = pool[pool["pick"].isin(chosen)]
            odds = float(np.prod(sel["price"]))
            model = float(np.prod(sel["model"]))
            fair = float(np.prod(sel["fair"]))
            ret = stake * odds
            a1, a2, a3 = st.columns(3)
            a1.metric("Legs", len(sel))
            a2.metric("Combined odds", f"{odds:.2f}")
            a3.metric(f"Return on £{stake:.2f}", f"£{ret:.2f}", f"+£{ret - stake:.2f}")
            b1, b2, b3 = st.columns(3)
            b1.metric("Model: chance it lands", f"{model * 100:.1f}%")
            b2.metric("Market fair chance", f"{fair * 100:.1f}%")
            b3.metric("EV if market is right", f"{(fair * odds - 1) * 100:+.1f}%")
            dupes = [g for g, c in sel["game"].value_counts().items() if c > 1]
            if dupes:
                st.warning("⚠️ Multiple legs from the same game (" + ", ".join(dupes) +
                           "). Those outcomes are correlated, so the chances above are "
                           "off and most bookmakers need a same-game multi instead.")
            for i, (_, b) in enumerate(sel.iterrows()):
                ui.bet_card(ctx, b, f"acca{i}")
            st.caption("Chances assume the legs are independent (true across different "
                       "games). Every extra leg multiplies the bookmaker's margin along "
                       "with the odds, so a fold is high-variance even when each leg is "
                       "fairly priced. 'EV if market is right' shows that cost directly.")
