import numpy as np
import pandas as pd
import streamlit as st

import nfl_model as nm
import props as pr

st.title("Backtest")
st.caption("Replays past seasons week by week using only information that was available "
           "before each week, then compares the model with what actually happened.")

TEST_SEASONS = [2024, 2025]
HISTORY = [2023, 2024, 2025]


@st.cache_data(show_spinner="Replaying 2024-2025 game by game...")
def game_backtest(k, regress):
    tw = nm.load_team_weeks(HISTORY)
    sched = pd.concat([nm.get_schedule(s) for s in HISTORY], ignore_index=True)
    bt = nm.run_backtest(tw, sched, TEST_SEASONS, k, regress)
    fit = nm.fit_backtest(bt)
    return bt, fit, nm.qb_change_effect(bt, fit), nm.calibration_table(bt, fit), \
        nm.simulate_bets(bt, fit, use_blend=False)


@st.cache_data(show_spinner="Searching settings (about a minute)...")
def grid_search():
    tw = nm.load_team_weeks(HISTORY)
    sched = pd.concat([nm.get_schedule(s) for s in HISTORY], ignore_index=True)
    rows = []
    for k in (100, 200, 400, 800, 1600):
        for rg in (0.0, 0.2, 0.33, 0.5, 0.7):
            bt = nm.run_backtest(tw, sched, TEST_SEASONS, k, rg)
            rows.append({"k": k, "regress": rg, "avg_error_pts": nm.cv_mae(bt)})
    return pd.DataFrame(rows).sort_values("avg_error_pts")


@st.cache_data(show_spinner="Replaying player props (about a minute)...")
def props_backtest(season):
    logs = pr.get_player_logs(tuple(HISTORY))
    roll = pr.rolling_table(logs)
    return pr.prop_backtest(roll, season)


# ---------- Game markets ----------
st.header("Game markets")
p = nm.PARAMS
bt, fit, qb, (cal, brier), sim = game_backtest(p["k"], p["regress"])
st.caption(f"{fit['n']} games, 2024-2025 · blend settings k={p['k']}, regress={p['regress']}")

st.subheader("Does the model beat the closing line?")
st.dataframe(pd.DataFrame({
    "": ["Spread (avg points off)", "Total (avg points off)"],
    "Model alone": [fit["mae_margin_model"], fit["mae_total_model"]],
    "Closing line": [fit["mae_margin_market"], fit["mae_total_market"]],
    "Blend used in app": [fit["mae_margin_blend"], fit["mae_total_blend"]],
}).round(2), hide_index=True)
st.markdown(
    f"Extra information the model adds beyond the line (0 = none, 1 = the model is right "
    f"and the line is wrong): **margins {fit['w_margin_raw']:+.2f} ± {fit['w_margin_se']:.2f}**, "
    f"**totals {fit['w_total_raw']:+.2f} ± {fit['w_total_se']:.2f}**. "
    "Values within about two standard errors of zero mean the model cannot be told apart "
    "from adding nothing.")

st.subheader("Settings fitted from this backtest")
st.dataframe(pd.DataFrame({
    "setting": ["scale", "hfa", "total_scale", "margin_sd (model alone)",
                "total_sd (model alone)", "qb_adj"],
    "fitted": [fit["scale"], fit["hfa"], fit["total_scale"], fit["margin_sd_model"],
               fit["total_sd_model"], -qb["model_shortfall"]],
    "in app now": [p["scale"], p["hfa"], p["total_scale"], p["margin_sd"], p["total_sd"],
                   p["qb_adj"]],
}).round(2), hide_index=True)
st.caption(f"qb_adj: teams starting a different QB than in their previous game fell short "
           f"of the model by {-qb['model_shortfall']:.1f} ± {qb['model_se']:.1f} points "
           f"({qb['n']} team-games), but only {-qb['market_shortfall']:.1f} short of the "
           "market line, so the market already prices QB changes in.")

st.subheader("Moneyline calibration (model alone)")
st.dataframe(cal.round(3))
st.caption(f"Brier score (lower is better): model {brier['model']:.3f}, "
           f"market {brier['market']:.3f}. Model is over-confident in the tails.")

st.subheader("Betting every edge at closing prices (model alone)")
st.dataframe(sim.round(3), hide_index=True)
st.caption("Pushes skipped. Closing prices are the hardest benchmark, since they reflect "
           "all the late information. Totals are about break-even; spreads and moneylines lose.")

if st.button("Search blend settings (k, regress)"):
    st.dataframe(grid_search().round(3), hide_index=True)
    st.caption("Average points of error with settings fitted on one season and scored on "
               "the other. Small differences here are noise.")

# ---------- Props ----------
st.header("Player props")
st.caption("There is no free historical prop price data, so props can only be checked "
           "against what happened, not against the market. Calibration below means: when "
           "the model says 60%, does it happen about 60% of the time?")
season = st.radio("Season to test", TEST_SEASONS, index=1, horizontal=True)
if st.button("Run props backtest"):
    summ, cals, td, td_cal = props_backtest(season)
    st.subheader("Projection accuracy")
    st.dataframe(summ.round(2), hide_index=True)
    st.caption("avg_projection should match avg_actual. Dispersion controls how wide the "
               "outcome range is (for passing yards it is the SD as a fraction of the "
               "projection).")

    st.subheader("Anytime TD")
    st.markdown(
        f"{td['player_games']} player-games · model averages {td['avg_predicted']:.1%} vs "
        f"actual {td['actual_rate']:.1%} · log-loss {td['logloss']:.3f} vs "
        f"{td['logloss_base_rate']:.3f} for a flat base rate (lower is better)")
    st.dataframe(td_cal.round(3))

    for key, tab in cals.items():
        st.subheader(f"{pr.MARKETS[key][0]} calibration")
        st.caption("Probability of going over lines set below / at / above the projection.")
        st.dataframe(tab.round(3))
