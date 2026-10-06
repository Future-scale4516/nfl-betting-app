import pandas as pd
import streamlit as st

import nfl_model as nm
import ui

ui.setup_page("Model")
ctx = ui.get_context()

st.title("🏈 NFL Analyser")
st.caption("Model view, injuries and team ratings. Use the pages in the sidebar for "
           "Game Bets, Player Props, Suggested Bets and Results. The week picker and "
           "stake in the sidebar apply to every page.")

tab_model, tab_inj, tab_rat = st.tabs(["Model", "Injuries", "Team Ratings"])

# ---------- Model ----------
with tab_model:
    st.caption(
        f"League average total used: {ctx.lg_total:.1f} · Weight on the model vs the "
        f"market line: {nm.PARAMS['w_margin']:.0%} (margins), {nm.PARAMS['w_total']:.0%} "
        "(totals). The backtest found the model adds little beyond the market."
    )
    for _, p in ctx.preds.iterrows():
        _, uk_time, us_date = ui.kickoff(p["gameday"], p["gametime"])
        with st.container(border=True):
            title = f"**{p['away']} @ {p['home']}**"
            if p["neutral"]:
                title += " · neutral site"
            st.markdown(title)
            st.caption(
                f"{uk_time} ({us_date}) · "
                f"Model: {nm.fmt_spread(p['home'], p['away'], p['raw_margin'])}, "
                f"total {p['raw_total']:.1f} · "
                f"Market: {nm.fmt_spread(p['home'], p['away'], p['mkt_spread'])}, "
                f"total {p['mkt_total']} · "
                f"Blend used: {nm.fmt_spread(p['home'], p['away'], p['model_margin'])}, "
                f"{p['home']} win {p['home_win_prob']:.0%}"
            )
            flags = [f"{t} starting QB out" for t, on in
                     ((p["home"], p["qb_out_home"]), (p["away"], p["qb_out_away"])) if on]
            if flags:
                st.caption("⚠️ " + " · ".join(flags))
            if pd.notna(p["home_score"]):
                st.caption(f"Final: {p['away']} {p['away_score']:.0f} - "
                           f"{p['home_score']:.0f} {p['home']}")

# ---------- Injuries ----------
with tab_inj:
    if not ctx.is_current:
        st.info("Injury reports are shown for the upcoming week.")
    elif ctx.inj.empty:
        st.warning("Injury data is unavailable from nflverse right now.")
    else:
        for t, on in ctx.qb_out.items():
            if on:
                st.error(f"{t}: starting QB is ruled out. The model penalises {t} by "
                         f"{nm.PARAMS['qb_adj']:.1f} points.")
        wk = ctx.inj[(ctx.inj["week"] == ctx.week) & ctx.inj["team"].isin(ctx.teams)
                     & ctx.inj["position"].isin(["QB", "RB", "WR", "TE"])]
        rows = []
        for _, r in wk.iterrows():
            s, n = ctx.inj_map.get(r["gsis_id"], ("", ""))
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
    st.dataframe(ctx.ratings.round(3).reset_index(), hide_index=True)
