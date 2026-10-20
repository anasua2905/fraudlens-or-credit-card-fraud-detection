# Demo script (about 6 minutes)

A live walk-through of the deployed system for the final presentation. Slide 10 of the deck hands over to this
demo; the speaker notes of every slide carry the talk itself.

## Before the session (5 minutes)

1. Start the deployment on the demo laptop (or use the hosted dashboard):

   ```powershell
   powershell -ExecutionPolicy Bypass -File scripts\deploy\run_local.ps1 -Public
   ```

   Note the two links it prints (dashboard and API docs), or open `data\deploy\deployment.json`.
2. Open three browser tabs: the dashboard, the API docs (`…/docs`), and the GitHub repository on branch `final-a4`.
3. In the dashboard sidebar keep the defaults: *Test* period, *Model default (min_cost)* threshold, $15 review cost.

## 1. Overview: does it meet the brief? (60 s)

* **Overview** tab: point at the KPI tiles: 555,719 test transactions, 2,851 alerts, **96.2% recall**, **72.4%
  precision**, **$964k net benefit**.
* Read the "What this period says" sentence: 14.7 alerts a day, 98.7% of fraud value caught.
* The daily chart: green bars are caught fraud, amber false alerts, red line missed fraud (almost flat at zero).

## 2. Alert queue: an analyst's morning (90 s)

* **Alert queue** tab: 531 cards account for the 2,851 alerts; reviewing by card cuts the queue by 81%.
* Pick the top card: its timeline shows a burst of high scores within two days; the table lists each alert with
  its **reasons** (SHAP contributions and plain rules such as "Amount is 13× this card's usual spend").
* Record a verdict in the feedback form ("fraud"), and mention it flows into monitoring and retraining.

## 3. Live scoring: the real-time path (90 s)

* **Live scoring** tab: leave *new card*, amount 875, *shopping_net*; click **Score transaction**.
* Click **Simulate a card-testing burst**: six rising amounts three minutes apart; the score climbs past the
  threshold and the reasons explain why (5 transactions in 24 h, 3 minutes since the last, 7× usual spend).
* Switch to the **API docs** tab: `POST /score` → *Try it out* → *Execute* with the example transaction; same score,
  same reasons, as JSON. Show `GET /metrics` (Prometheus counters increase with each call).

## 4. Threshold & cost: the manager's decision (45 s)

* **Threshold & cost** tab: the dotted line marks the current operating point at the net-benefit peak.
* Change the preset to *Alert budget (50/day)*: recall rises to 99.2% but precision falls to 24% and net benefit
  drops; the box explains it only pays if a review costs under $2.02.
* Set the preset back to *Model default*.

## 5. Monitoring and the model (45 s)

* **Monitoring** tab: monthly recall stays above the 85% floor; drift share crossed 30% in August and December
  (seasonal spending), so retraining is recommended for review.
* **Model** tab: why HistGradientBoosting was chosen (simplest non-inferior model), and the SHAP importance chart.

## 6. Repository and platform (30 s)

* GitHub, branch `final-a4`: README (lifecycle table), `notebooks/01-08` (executed), `platform/docker-compose.yml`
  (MinIO, NiFi, Hive Metastore, Trino, ClickHouse, Superset, MLflow, Prometheus, Grafana, OpenMetadata).
* Mention the CI workflows (tests, platform end to end, image deploy) and the GitHub Project board.

## If something goes wrong

| Problem | Fallback |
|---|---|
| Public link not loading | use `http://127.0.0.1:8501` on the laptop, or restart with `run_local.ps1 -Public` (new links) |
| Dashboard slow on first load | it loads 667k scored transactions once (about 10 s); later tabs are instant |
| No internet at all | run `streamlit run dashboard/app.py` locally; the deck has screenshots of every step |
