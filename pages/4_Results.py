import pandas as pd
import streamlit as st

import nfl_model as nm
import props as pr
import tracker as tk
import ui

ui.setup_page("Results", "📋")
st.title("📋 Results")

if tk.gist_ready():
    st.success("Picks are saved to your private GitHub Gist.")
else:
    st.warning("Gist storage is not set up, so picks only live in this browser session. "
               "Download the CSV before closing the tab, or add GITHUB_TOKEN and GIST_ID "
               "to your Streamlit secrets.")

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
                emap = nm.map_events_to_games(events, sched)
                gid_to_event = {g: e for e, g in emap.items()}
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
    st.info("No picks logged yet. Use the Log pick buttons on the Game Bets and "
            "Player Props tabs.")
    st.stop()

# ---------- Summary ----------
def show(df):
    """Headline numbers + the picks table for a set of picks."""
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


st.caption("CLV compares the price you took with the latest price before kickoff. "
           "Consistently positive CLV is the best early sign of a real edge, well before "
           "win/loss results settle down.")
markets = [m for m in ["Moneyline", "Spread", "Total"] + [v[0] for v in pr.MARKETS.values()]
           if m in set(picks["market"])]
tabs = st.tabs(["📊 All"] + markets)
with tabs[0]:
    show(picks)
    summ = tk.summarise(picks)
    if not summ.empty:
        st.markdown("**By market**")
        st.dataframe(summ.round(3))
for tab, m in zip(tabs[1:], markets):
    with tab:
        show(picks[picks["market"] == m])
st.download_button("Download CSV", picks.to_csv(index=False), "nfl_picks.csv", "text/csv")
