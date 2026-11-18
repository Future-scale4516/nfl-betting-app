# Snapshot job: one-time setup (about 5 minutes)

What it does: every 30 minutes (every 10 near kickoffs) GitHub runs `snapshot.py`, which records
the market's prices at about 72h, 24h, 2h and the close before each NFL game, plus prop prices at
the close. They are saved to your private Gist, and the Results page uses them to measure CLV
against the de-vigged closing price (Pinnacle where available). Cost: roughly 6 Odds API credits per
game window and about 40 per week for props, well inside the 20k plan.

1. In your repo on GitHub, add `snapshot.py` at the top level (next to app.py).
2. Add `snapshot.yml` at `.github/workflows/snapshot.yml` (GitHub: Add file > Create new file, type
   the path including the folders, paste the contents).
3. Repo Settings > Secrets and variables > Actions > New repository secret. Add three:
   - `ODDS_API_KEY`  (same key as in your Streamlit secrets)
   - `GIST_TOKEN`    (same token as GITHUB_TOKEN in your Streamlit secrets, gist scope)
   - `GIST_ID`       (same as in your Streamlit secrets)
4. Actions tab > "Price snapshots" > Run workflow once. The log should say "nothing due" or "saved".
   If it says "snapshot failed", paste me the log line.
5. After a game has kicked off, open Results > "Update CLV from snapshots".

Notes
- If the repo is private, runs use GitHub free minutes (about 1,500 to 1,900 a month at this
  schedule, inside the 2,000 free). Public repos are unlimited. Streamlit Cloud ignores these files.
- Snapshots only exist from the day you switch this on, so CLV builds up from then.
