import pandas as pd
import streamlit as st

import nfl_model as nm
import props as pr
import results_lib as rl
import tracker as tk
import ui

ui.setup_page("Results", "📋")
ctx = ui.get_context()

st.title("📋 Results")
st.caption("How the model's predictions turned out for the week picked in the sidebar: what it "
           "projected, what actually happened, and whether it landed. Covers game bets "
           "(Money Line / Spread / Total) and player props. Free nflverse data, no Odds API "
           "credits used. Your own logged picks and CLV are further down.")

bet_type = st.radio("Bet type", ["Game bets", "Player props"], horizontal=True,
                    help="Two separate pipelines that use different data. The numbers below "
                         "reflect only the type you pick here.")
is_game = bet_type == "Game bets"

if is_game:
    model_mode = st.radio(
        "Which prediction to score", ["Pure model", "Blend the app bets from"], horizontal=True,
        help="Pure model is the model's own view. The blend is the market-weighted view "
             "Game Bets uses (mostly the bookmakers' price). Game bets are priced at "
             "nflverse's closing lines, since historical UK prices aren't free.")
else:
    st.caption("Props have no free historical prices, so these are model reads: the "
               "projection the model made before kickoff (using only earlier games) against "
               "what the player actually did. Over lines are standard thresholds.")

if st.button("Load results"):
    with st.spinner("Rebuilding predictions and checking them against results..."):
        if is_game:
            rdf = rl.game_results(ctx.preds, ctx.week_games, raw=model_mode == "Pure model")
            note = ("Priced at nflverse closing lines. Pushes are excluded." if rdf is not None
                    else "")
        else:
            roll = rl.cached_roll(ctx.logs, f"{nm.SEASON}-{ctx.current_week}")
            rdf = rl.prop_results(roll, nm.SEASON, ctx.week)
            note = "Projection uses only games before this week." if rdf is not None else ""
    st.session_state["res_df"] = rdf
    st.session_state["res_note"] = note
    st.session_state["res_key"] = (ctx.week, bet_type)

rdf = st.session_state.get("res_df")
if st.session_state.get("res_key") == (ctx.week, bet_type):
    if rdf is None or rdf.empty:
        st.warning("No finished games with results for this week yet. Pick an earlier week in "
                   "the sidebar.")
    else:
        st.caption(st.session_state.get("res_note", ""))
        floor = st.slider("Only show picks the model rated at least this likely (%)",
                          0, 90, 50 if is_game else 30, 5,
                          help="The model prices every selection. Filtering to its more "
                               "confident calls shows how the picks you'd have considered "
                               "performed.")
        sub_all = rdf[rdf["Model %"] >= floor]
        if sub_all.empty:
            st.warning("No picks at or above that confidence level.")
        else:
            hits, total = int(sub_all["Hit"].sum()), len(sub_all)
            rate, avg_model = hits / total * 100, sub_all["Model %"].mean()
            if is_game:
                pl = sub_all["P/L (1u)"].sum()
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Landed", f"{hits}/{total}")
                c2.metric("Hit rate", f"{rate:.1f}%")
                c3.metric("P/L (1u stakes)", f"{pl:+.2f}u")
                c4.metric("ROI", f"{pl / total * 100:+.1f}%")
                st.caption("P/L assumes a flat 1-unit stake on every pick shown, at closing "
                           "prices. One week is a data point, not a verdict.")
            else:
                c1, c2, c3 = st.columns(3)
                c1.metric("Landed", f"{hits}/{total}")
                c2.metric("Actual hit rate", f"{rate:.1f}%")
                c3.metric("Model predicted", f"{avg_model:.1f}%")
            if abs(rate - avg_model) <= 5:
                st.success(f"Model said {avg_model:.1f}%, reality was {rate:.1f}%: closely "
                           "calibrated this week.")
            elif rate < avg_model:
                st.warning(f"Model said {avg_model:.1f}% but only {rate:.1f}% landed: "
                           "overconfident this week.")
            else:
                st.info(f"Model said {avg_model:.1f}% and {rate:.1f}% landed: underconfident "
                        "this week.")
            st.caption("One week is a small sample. The Backtest page judges calibration over "
                       "two full seasons.")

            markets = rl.GAME_MARKETS if is_game else [v[0] for v in pr.MARKETS.values()]
            markets = [m for m in markets if m in set(sub_all["Market"])]
            tabs = st.tabs(["📊 All"] + markets)

            def table(tab, market=None):
                with tab:
                    sub = sub_all if market is None else sub_all[sub_all["Market"] == market]
                    if sub.empty:
                        st.write("No picks in this market at that confidence level.")
                        return
                    h, t = int(sub["Hit"].sum()), len(sub)
                    hdr = (f"**{h}/{t} landed ({h / t * 100:.1f}%)** · model predicted "
                           f"{sub['Model %'].mean():.1f}%")
                    if "P/L (1u)" in sub.columns:
                        hdr += f" · P/L {sub['P/L (1u)'].sum():+.2f}u"
                    st.markdown(hdr)
                    d = sub.copy()
                    d["Result"] = d["Hit"].map({True: "✅", False: "❌"})
                    cols = (["Result", "Light", "Market", "Selection", "Model %", "Fair %",
                             "Edge", "Odds", "P/L (1u)", "Game", "Score"] if is_game else
                            ["Result", "Player", "Pos", "Market", "Line", "Model %",
                             "Projection", "Actual", "Game"])
                    if market is not None and "Market" in cols:
                        cols.remove("Market")
                    st.dataframe(
                        d[cols].sort_values("Model %", ascending=False).reset_index(drop=True),
                        hide_index=True,
                        column_config={
                            "Result": st.column_config.TextColumn("", width="small"),
                            "Model %": st.column_config.NumberColumn("Model %", format="%.1f"),
                            "Fair %": st.column_config.NumberColumn("Fair %", format="%.1f"),
                            "Edge": st.column_config.NumberColumn("Edge", format="%.1f"),
                            "Odds": st.column_config.NumberColumn("Odds", format="%.2f"),
                            "P/L (1u)": st.column_config.NumberColumn("P/L", format="%.2f"),
                            "Projection": st.column_config.NumberColumn("Proj", format="%.1f"),
                            "Actual": st.column_config.NumberColumn("Actual", format="%.0f"),
                        })

            table(tabs[0])
            for tab, m in zip(tabs[1:], markets):
                table(tab, m)
            st.download_button(
                "Download results CSV", sub_all.to_csv(index=False).encode("utf-8"),
                f"nfl_{'game' if is_game else 'props'}_results_week{ctx.week}.csv", "text/csv")
else:
    st.info("Pick a week in the sidebar and click **Load results**.")

# ---------------------------------------------------------------- my picks
st.divider()
st.header("🧾 My logged picks and CLV")
st.caption("Picks you logged with the Log pick buttons, settled against real results. CLV "
           "compares the price you took with the latest price before kickoff. Consistently "
           "positive CLV is the best early sign of a real edge.")

if tk.gist_ready():
    st.success("Picks are saved to your private GitHub Gist.")
else:
    st.warning("Gist storage is not set up, so picks only live in this browser session.")

try:
    picks = tk.load_picks()
except Exception as e:
    st.error(f"Could not load picks: {e}")
    st.stop()

c1, c2, c3 = st.columns(3)
if c1.button("Reload"):
    try:
        picks = tk.load_picks(force=True)
    except Exception as e:
        st.error(f"Reload failed: {e}")
if c2.button("Settle results"):
    try:
        new, n = tk.settle_picks(picks)
        tk.save_picks(new)
        picks = new
        st.success(f"Settled {n} pick(s).")
    except Exception as e:
        st.error(f"Settling failed: {e}")

regions = st.radio("Regions for closing prices", ["uk", "uk,us"], horizontal=True,
                   help="Use the same regions you fetched the pick from.")
if c3.button("Update closing prices"):
    try:
        open_ = picks[picks["result"].isna() | (picks["result"] == "")]
        if open_.empty:
            st.info("No open picks.")
        else:
            events, rem = nm.fetch_odds_raw(regions)
            game_odds = nm.odds_to_rows(events)
            prop_rows = pd.DataFrame()
            props_open = open_[~open_["market_key"].isin(["moneyline", "spread", "total"])]
            if not props_open.empty:
                sched = pd.concat([nm.get_schedule(int(s)) for s in props_open["season"].unique()])
                gid_to_event = {g: e for e, g in nm.map_events_to_games(events, sched).items()}
                frames = []
                for gid, grp in props_open.groupby("game_id"):
                    ev = gid_to_event.get(gid)
                    if not ev:
                        continue
                    api = sorted({pr.MARKETS[k][1] for k in grp["market_key"]})
                    data, rem = pr.fetch_event_props(ev, api, regions)
                    frames.append(pr.props_to_rows(data, gid))
                if frames:
                    prop_rows = pd.concat(frames, ignore_index=True)
            new, n = tk.update_closing(picks, game_odds, prop_rows)
            tk.save_picks(new)
            picks = new
            st.success(f"Updated {n} closing price(s). Credits remaining: {rem}")
    except Exception as e:
        st.error(f"Closing price update failed: {e}")

if picks.empty:
    st.info("No picks logged yet. Use the Log pick buttons on the Game Bets and Player "
            "Props pages.")
    st.stop()


def show(df):
    settled = df[df["result"].isin(["WIN", "LOSE", "PUSH"])]
    open_n = int((df["result"].isna() | (df["result"] == "")).sum())
    staked, profit = settled["stake"].sum(), settled["profit"].sum()
    clv = df["clv"].dropna()
    a, b, c, d = st.columns(4)
    a.metric("Picks", len(df))
    a.caption(f"{open_n} open")
    b.metric("Profit (£)", f"{profit:+.2f}")
    c.metric("ROI", f"{profit / staked:+.1%}" if staked else "n/a")
    d.metric("Avg CLV", f"{clv.mean():+.1%}" if len(clv) else "n/a")
    d.caption(f"{len(clv)} with closing price")
    cols = ["logged_at", "week", "selection", "game", "book", "price", "close_price", "clv",
            "model", "stake", "result", "profit"]
    st.dataframe(df.sort_values("logged_at", ascending=False)[cols], hide_index=True)


pmarkets = [m for m in ["Moneyline", "Spread", "Total"] + [v[0] for v in pr.MARKETS.values()]
            if m in set(picks["market"])]
ptabs = st.tabs(["📊 All"] + pmarkets)
with ptabs[0]:
    show(picks)
    summ = tk.summarise(picks)
    if not summ.empty:
        st.markdown("**By market**")
        st.dataframe(summ.round(3))
for tab, m in zip(ptabs[1:], pmarkets):
    with tab:
        show(picks[picks["market"] == m])
st.download_button("Download picks CSV", picks.to_csv(index=False), "nfl_picks.csv", "text/csv")
