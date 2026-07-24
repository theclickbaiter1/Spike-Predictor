# Agent prompt: Portfolio media for Spike Detector

Copy this entire prompt into a coding agent opened on this repo.

---

## Mission

Produce portfolio-ready media for **sstasbih.com** project page `/projects/spike-detector`. Drop finished files into:

`/Users/sstasbih/Desktop/Projects/sstasbih.com/public/assets/projects/spike-detector/`

Then uncomment the matching `<Media>` blocks in:

`/Users/sstasbih/Desktop/Projects/sstasbih.com/src/pages/projects/spike-detector.astro`

## Story the media must tell

Pre-market binary spike classifier (not a price predictor): two-stage XGBoost + FinBERT + stat-mech layer, automated paper trading via Alpaca, Telegram alerts, EOD validation.

## Deliverables (required)

You already have strong candidates under `output/`. Copy (and lightly polish if needed):

1. **`trend.png`**
   - Source: latest `output/hist_eval/**/validation-*/trend.png` (pick the most recent clean chart).
   - Caption intent: rolling validation metrics from live paper trading.

2. **`backtest.png`**
   - Source: `output/backtest_strategy.png` or `output/backtest_results.png` (prefer strategy with TP/SL if readable).

3. **`thumb.jpg`**
   - 16:9 crop of the trend chart or a clean watchlist still.

## Optional (high value)

4. **`telegram-watchlist.png`** — redacted morning Telegram watchlist (blur chat IDs, exact dollar sizes if sensitive).
5. **`pipeline.png`** — export the README mermaid architecture to a PNG (or redraw simply).

## Constraints

- Prefer existing real charts over regenerating unless files are broken.
- Redact account numbers and large P&L callouts if they look like live money.
- No em dashes in captions.
- After copy, update `spike-detector.astro` Media `src` paths to `/assets/projects/spike-detector/...`.

## Done when

`trend.png`, `backtest.png`, and `thumb.jpg` exist in the portfolio folder and Media components are uncommented.
