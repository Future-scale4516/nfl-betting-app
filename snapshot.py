"""Price snapshots for CLV. Runs from a GitHub Action every 30 minutes (no
Streamlit needed). For each NFL game it records the market at four points before
kickoff (about 72h, 24h, 2h and the last run before kickoff = the close) as a
compact table: best price, median price, and the de-vigged fair probability, plus
Pinnacle's own fair probability when the eu region lists it. Rows go into the same
private Gist as your picks, as snap_games_<year>W<week>.csv (and props at the
close, snap_props_<year>W<week>.csv).

Environment: ODDS_API_KEY, GITHUB_TOKEN (gist scope), GIST_ID.
Optional: SNAP_REGIONS (default uk,eu), SNAP_PROP_MARKETS, SNAP_PROP_REGIONS."""
import csv
import io
import os
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import requests

SPORT = "americanfootball_nfl"
BASE = f"https://api.the-odds-api.com/v4/sports/{SPORT}"
KEY = os.environ["ODDS_API_KEY"]
TOKEN, GIST = os.environ["GITHUB_TOKEN"], os.environ["GIST_ID"]
REGIONS = os.environ.get("SNAP_REGIONS", "uk,eu")
PROP_MARKETS = [m for m in os.environ.get(
    "SNAP_PROP_MARKETS", "player_anytime_td,player_receptions,player_reception_yds"
).split(",") if m]
PROP_REGIONS = os.environ.get("SNAP_PROP_REGIONS", "uk")

# (name, minutes before kickoff the window opens, window width in minutes)
WINDOWS = [("T72", 4320, 30), ("T24", 1440, 30), ("T2", 120, 30), ("T0", 0, 30)]
COMMISSION = {"betfair_ex_uk": 0.05, "betfair_ex_eu": 0.05, "betfair_ex_au": 0.05,
              "matchbook": 0.02, "smarkets": 0.02}  # same as the app: net of commission
GAME_COLS = ["ts_utc", "window", "mins_to_ko", "event_id", "commence_utc", "home", "away",
             "market", "name", "point", "n_books", "best_price", "best_book",
             "median_price", "fair_median", "fair_pinn"]
PROP_COLS = ["ts_utc", "window", "mins_to_ko", "event_id", "commence_utc", "home", "away",
             "market", "player", "name", "point", "n_books", "best_price", "best_book",
             "median_price", "fair_median"]


def net(book_key, price):
    c = COMMISSION.get(book_key, 0.0)
    return 1 + (price - 1) * (1 - c) if c else price


def get(url, **params):
    r = requests.get(url, params={"apiKey": KEY, **params}, timeout=30)
    r.raise_for_status()
    return r.json(), r.headers.get("x-requests-remaining")


def nfl_week_key(commence):
    d = (commence - timedelta(days=2)).isocalendar()
    return f"{d[0]}W{d[1]:02d}"


# ---------- Gist ----------
H = {"Authorization": f"Bearer {TOKEN}", "Accept": "application/vnd.github+json"}


def gist_files():
    r = requests.get(f"https://api.github.com/gists/{GIST}", headers=H, timeout=30)
    r.raise_for_status()
    return r.json()["files"]


def read_rows(files, name):
    f = files.get(name)
    if not f:
        return []
    text = f.get("content", "")
    if f.get("truncated"):
        text = requests.get(f["raw_url"], headers=H, timeout=30).text
    return list(csv.DictReader(io.StringIO(text))) if text.strip() else []


def to_csv(rows, cols):
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


def merge(old, new, cols):
    """Keep the first row per (event, window, market, name, point[, player]) for most
    windows, but let the latest T0 row replace earlier ones (closest to kickoff)."""
    ident = lambda r: tuple(r.get(c, "") for c in cols if c in (
        "event_id", "window", "market", "player", "name", "point"))
    out = {ident(r): r for r in old}
    for r in new:
        k = ident(r)
        if k not in out or r["window"] == "T0":
            out[k] = r
    return list(out.values())


# ---------- Market summaries ----------
def devig_pairs(books, market):
    """per book: {(name, point): fair prob} by de-vigging each two-way pair."""
    fair = {}
    for bk in books:
        mk = next((m for m in bk["markets"] if m["key"] == market), None)
        if not mk:
            continue
        oc = mk["outcomes"]
        for o in oc:
            if market == "h2h":
                opp = [x for x in oc if x["name"] != o["name"]]
            elif market == "spreads":
                opp = [x for x in oc if x["name"] != o["name"]
                       and abs(x.get("point", 0) + o.get("point", 0)) < 1e-9]
            else:
                opp = [x for x in oc if x["name"] != o["name"] and x.get("point") == o.get("point")]
            if len(opp) == 1:
                a, b = 1 / net(bk["key"], o["price"]), 1 / net(bk["key"], opp[0]["price"])
                fair[(bk["key"], o["name"], o.get("point"))] = a / (a + b)
    return fair


def game_rows(ev, odds_ev, window, mins, now):
    books = odds_ev.get("bookmakers", [])
    out = []
    for market in ("h2h", "spreads", "totals"):
        fair = devig_pairs(books, market)
        grp = defaultdict(list)
        for bk in books:
            mk = next((m for m in bk["markets"] if m["key"] == market), None)
            for o in (mk["outcomes"] if mk else []):
                grp[(o["name"], o.get("point"))].append((bk["key"], bk["title"],
                                                         net(bk["key"], o["price"])))
        for (name, point), quotes in grp.items():
            if len(quotes) < 2 and market != "h2h":
                continue  # lines only one book offers are noise for CLV
            best = max(quotes, key=lambda q: q[2])
            fs = [fair[(q[0], name, point)] for q in quotes if (q[0], name, point) in fair]
            pinn = fair.get(("pinnacle", name, point))
            out.append({
                "ts_utc": now.isoformat(timespec="seconds"), "window": window,
                "mins_to_ko": round(mins), "event_id": ev["id"],
                "commence_utc": ev["commence_time"], "home": ev["home_team"],
                "away": ev["away_team"], "market": market, "name": name,
                "point": "" if point is None else point, "n_books": len(quotes),
                "best_price": round(best[2], 4), "best_book": best[1],
                "median_price": round(statistics.median(q[2] for q in quotes), 4),
                "fair_median": round(statistics.median(fs), 5) if fs else "",
                "fair_pinn": round(pinn, 5) if pinn is not None else ""})
    return out


def prop_rows(ev, data, window, mins, now):
    books = data.get("bookmakers", [])
    out = []
    for market in {m["key"] for b in books for m in b["markets"]}:
        grp = defaultdict(list)
        for bk in books:
            mk = next((m for m in bk["markets"] if m["key"] == market), None)
            for o in (mk["outcomes"] if mk else []):
                grp[(o.get("description", ""), o["name"], o.get("point"))].append(
                    (bk["key"], bk["title"], net(bk["key"], o["price"]), o))
        pair_fair = {}
        for (player, name, point), quotes in grp.items():
            opp = {"Over": "Under", "Under": "Over"}.get(name)
            if not opp:
                continue
            mine = {q[0]: q[2] for q in quotes}
            other = {q[0]: q[2] for q in grp.get((player, opp, point), [])}
            f = [(1 / mine[b]) / (1 / mine[b] + 1 / other[b]) for b in mine if b in other]
            if f:
                pair_fair[(player, name, point)] = statistics.median(f)
        # keep only the main line per player/market: the point most books quote
        main = {}
        for (player, name, point), quotes in grp.items():
            cur = main.get((player, name))
            if cur is None or len(quotes) > cur[1]:
                main[(player, name)] = (point, len(quotes))
        for (player, name, point), quotes in grp.items():
            if main[(player, name)][0] != point:
                continue
            best = max(quotes, key=lambda q: q[2])
            fair = pair_fair.get((player, name, point))
            out.append({
                "ts_utc": now.isoformat(timespec="seconds"), "window": window,
                "mins_to_ko": round(mins), "event_id": ev["id"],
                "commence_utc": ev["commence_time"], "home": ev["home_team"],
                "away": ev["away_team"], "market": market, "player": player, "name": name,
                "point": "" if point is None else point, "n_books": len(quotes),
                "best_price": round(best[2], 4), "best_book": best[1],
                "median_price": round(statistics.median(q[2] for q in quotes), 4),
                "fair_median": round(fair, 5) if fair is not None else ""})
    return out


def main():
    now = datetime.now(timezone.utc)
    events, _ = get(f"{BASE}/events")  # free: costs no credits
    due = {}  # event id -> (window name, minutes)
    for ev in events:
        ko = datetime.fromisoformat(ev["commence_time"].replace("Z", "+00:00"))
        mins = (ko - now).total_seconds() / 60
        for name, start, width in WINDOWS:
            if start <= mins < start + width:
                due[ev["id"]] = (ev, name, mins)
    if not due:
        print("nothing due")
        return
    print("due:", [(v[0]["away_team"] + " @ " + v[0]["home_team"], v[1]) for v in due.values()])
    odds, rem = get(f"{BASE}/odds", regions=REGIONS, markets="h2h,spreads,totals",
                    oddsFormat="decimal", eventIds=",".join(due))
    files = gist_files()
    updates = {}
    by_file = defaultdict(list)
    for o in odds:
        ev, window, mins = due[o["id"]]
        wk = nfl_week_key(datetime.fromisoformat(ev["commence_time"].replace("Z", "+00:00")))
        by_file[f"snap_games_{wk}.csv"] += game_rows(ev, o, window, mins, now)
    for fname, rows in by_file.items():
        updates[fname] = {"content": to_csv(merge(read_rows(files, fname), rows, GAME_COLS),
                                            GAME_COLS)}
    # Props: once per game, at the close window, first run only
    if PROP_MARKETS:
        pfiles = defaultdict(list)
        for eid, (ev, window, mins) in due.items():
            if window != "T0":
                continue
            wk = nfl_week_key(datetime.fromisoformat(ev["commence_time"].replace("Z", "+00:00")))
            fname = f"snap_props_{wk}.csv"
            have = read_rows(files, fname)
            if any(r["event_id"] == eid for r in have):
                continue
            data, rem = get(f"{BASE}/events/{eid}/odds", regions=PROP_REGIONS,
                            markets=",".join(PROP_MARKETS), oddsFormat="decimal")
            pfiles[fname] += prop_rows(ev, data, window, mins, now)
        for fname, rows in pfiles.items():
            updates[fname] = {"content": to_csv(read_rows(files, fname) + rows, PROP_COLS)}
    if updates:
        r = requests.patch(f"https://api.github.com/gists/{GIST}", headers=H, timeout=30,
                           json={"files": updates})
        r.raise_for_status()
    print("saved", {k: len(v["content"]) for k, v in updates.items()}, "credits left", rem)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # make failures visible in the Actions log
        print("snapshot failed:", repr(e))
        sys.exit(1)
