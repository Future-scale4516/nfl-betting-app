import pandas as pd
import streamlit as st

import nfl_model as nm
import props as pr
import ui

ui.setup_page("Suggested Bets", "📋")
ctx = ui.get_context()

st.title("📋 Suggested Bets")
st.caption("Auto-built doubles and trebles per market, ranked by the model's probability "
           "(Model %), never two legs from the same game. Stakes are split by market "
           "trust tier, which starts flat because there is no tracked NFL history yet. "
           "The backtest found no proven edge on game markets, so keep stakes small and "
           "use the Results page to see what actually holds up.")

if not ctx.is_current:
    st.info("Suggestions are only built for the upcoming week. Switch the week in the "
            "sidebar.")
    st.stop()

sel_games = ui.game_filter(ctx)

col1, col2 = st.columns(2)
bankroll = col1.number_input("Bankroll to allocate (£)", min_value=1.0, value=10.0, step=1.0)
pool_mode = col2.selectbox("Leg pool", list(ui.POOL_LABELS), format_func=ui.POOL_LABELS.get)
prop_mk = st.multiselect("Prop markets to include (each costs credits)", list(pr.MARKETS),
                         default=["anytime_td", "receptions"],
                         format_func=lambda k: pr.MARKETS[k][0])
regions = st.radio("Prop bookmaker regions", ["uk", "uk,us"], horizontal=True)
st.caption(f"Projected prop cost: about {len(sel_games) * len(prop_mk) * len(regions.split(','))} "
           "credits. Game-line odds are cached for 30 minutes.")

if st.button("Build suggested bets") and sel_games:
    results = {}
    try:
        with st.spinner("Scanning every market and building combos..."):
            events, remaining = nm.fetch_odds()
            st.session_state["odds"] = (events, remaining)
            odds = nm.odds_to_rows(events)
            gdf = nm.build_bets(ctx.preds, odds) if not odds.empty else pd.DataFrame()
            frames = []
            if not gdf.empty:
                frames.append(gdf)
            if prop_mk:
                po, rem = ui.fetch_props(ctx, sel_games, prop_mk, regions)
                if not po.empty:
                    pbets = pr.build_prop_bets(ui.get_projection(ctx).query("inj != 'OUT'"), po)
                    if not pbets.empty:
                        frames.append(pbets)
        if frames:
            allb = pd.concat(frames, ignore_index=True)
            allb = allb[allb["game_id"].isin(sel_games)]
            pool = ui.pool_filter(allb, pool_mode)
            for market in allb["market"].unique():
                sub = pool[pool["market"] == market]
                results[market] = {
                    "double": ui.best_combo(sub, 2), "treble": ui.best_combo(sub, 3),
                    "note": "" if not sub.empty else "No qualifying picks in this market."}
        st.session_state["suggested"] = results
        st.session_state["suggested_bankroll"] = bankroll
        st.session_state["suggested_week"] = ctx.week
        st.session_state.pop("suggested_status", None)
    except Exception as e:
        st.error(f"Build failed: {e}")

results = st.session_state.get("suggested")
if results is not None and st.session_state.get("suggested_week") == ctx.week:
    available = [m for m, r in results.items() if r["double"] or r["treble"]]
    if not available:
        st.warning("No markets have enough qualifying picks right now. Try the "
                   "positive-EV pool, include more games, or check again closer to "
                   "kickoff once prices settle.")
    else:
        stakes = ui.suggest_stakes(st.session_state.get("suggested_bankroll", bankroll),
                                   available)
        check = st.button("🔴 Check status")
        st.caption("Settles each leg from finished games and published player stats. "
                   "One lost leg kills the whole combo. Player stats can lag the final "
                   "score by some hours, so props may show Pending for a while.")
        if check:
            with st.spinner("Checking results..."):
                st.session_state["suggested_status"] = {
                    (m, t): ui.combo_status(r[t]) for m, r in results.items()
                    for t in ("double", "treble") if r[t]}
        status = st.session_state.get("suggested_status", {})

        for market in sorted(results):
            r = results[market]
            st.divider()
            if not (r["double"] or r["treble"]):
                st.markdown(f"### {market}")
                st.write(f"No qualifying picks. {r['note']}")
                continue
            stake = stakes.get(market, 0.0)
            st.markdown(f"### {market} — suggested stake £{stake:.2f}")
            if market == "Anytime TD":
                st.caption("Touchdown scorer folds are long shots: stake is deliberately "
                           "half-weighted.")
            cols = st.columns(2)
            for col, kind, label in ((cols[0], "double", "Double"), (cols[1], "treble", "Treble")):
                combo = r[kind]
                with col:
                    st.markdown(f"**{label}**")
                    if combo is None:
                        st.caption("Not enough distinct-game picks for this size.")
                        continue
                    legs_st = None
                    if (market, kind) in status:
                        overall, legs_st = status[(market, kind)]
                        st.markdown(f"**{ui.COMBO_BADGE[overall]}**")
                    for i, leg in enumerate(combo["legs"]):
                        st.write(f"• **{leg['label']}** ({leg['game']}) @ {leg['odds']:.2f} "
                                 f"— {leg['model']:.1%} model")
                        st.caption(leg["reason"])
                        if legs_st:
                            st.caption(ui.STATUS_BADGE[legs_st[i]])
                    m1, m2 = st.columns(2)
                    m1.metric("Combined odds", f"{combo['odds']:.2f}")
                    m2.metric("Chance to land", f"{combo['prob'] * 100:.1f}%")
                    st.metric(f"Return on £{stake:.2f}", f"£{stake * combo['odds']:.2f}")
                    st.caption(f"Worth {combo['fair'] * combo['odds'] - 1:+.1%} if the "
                               "market's fair prices are right.")
        st.divider()
        st.caption("Combined odds and stakes are starting points, not instructions: "
                   "confirm live prices at your book before staking. Suggestions are not "
                   "saved between visits. To track a combo, log its legs on the Game Bets "
                   "or Player Props pages.")
