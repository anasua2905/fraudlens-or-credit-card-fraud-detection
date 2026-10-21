# Demo script (about 7 minutes)

A live walk-through of the deployed system for the final presentation. Slide 10 of the deck hands over to this
demo; the speaker notes of every slide carry the talk itself.

## Before the session (5 minutes)

1. Start the deployment on the demo laptop (or use the hosted dashboard):

   ```powershell
   powershell -ExecutionPolicy Bypass -File scripts\deploy\run_local.ps1 -Public
   ```

   Note the two links it prints (dashboard and API docs) and the sign-in details (an `analyst` and a `viewer`
   account, plus API keys), or open `data\deploy\deployment.json` and `data\deploy\access.json`.
2. Open three browser tabs: the dashboard, the API docs (`…/docs`), and the GitHub repository on branch `final-a4`.
3. Sign in to the dashboard as `analyst`. In the sidebar keep the defaults: *Test* period, *Model default
   (min_cost)* threshold, $15 review cost.

## 1. Overview: does it meet the brief? (60 s)

* **Overview** tab: point at the KPI tiles: 555,719 test transactions, 2,851 alerts, **96.2% recall**, **72.4%
  precision**, and an **estimated $964k net benefit**. Say what it assumes: $15 per review and 90% of caught fraud
  recovered, on simulated data; it is an estimate for comparing operating points, not demonstrated savings.
* Read the "What this period says" sentence: 14.7 alerts a day, 98.7% of fraud value caught.
* The daily chart: green bars are caught fraud, amber false alerts, red line missed fraud (almost flat at zero).

## 2. A new transaction, end to end (90 s)

* **Live transaction** tab: pick a known card, amount 875, *shopping_net*, 30 minutes after its last transaction;
  click **Score transaction**.
* Walk the four boxes: **1** the transaction as received; **2** the features the online feature store generated
  from the card's history (13× its usual spend, minutes since the last transaction, 24-hour velocity); **3** the
  model's score against the threshold, its SHAP contributions and the counterfactual ("would have been approved at $X");
  **4** routing: *ALERT, case C-… opened in the review queue*.
* Click **Simulate a card-testing burst**: six rising amounts three minutes apart; the score climbs past the
  threshold as the velocity features rise; each alert opens a case.

## 3. The analyst's decision and what it feeds (90 s)

* **Case review** tab: the alerts are waiting as open cases. Open the top case: score vs threshold, evidence,
  contributions, the exact model inputs.
* Record **Confirmed fraud** with a note; the case closes, signed with the analyst's name (show the audit log).
* Point at **How these decisions inform monitoring**: live precision with its 95% interval against the 40% floor,
  missed fraud from disputes on approved transactions, backlog and time to decision, and the labelled examples
  exported for retraining. The same numbers are in the API (`GET /monitoring/feedback`) and in Grafana, with a
  Prometheus alert if live precision drops below the floor.
* Optional: sign out and in as `viewer`: the decision buttons are disabled. In the **API docs** tab, `POST /score`
  without a key returns 401; with the scorer key it returns the score and a `case_id`.

## 4. Threshold & cost: the manager's decision (45 s)

* **Threshold & cost** tab: the dotted line marks the current operating point at the net-benefit peak.
* Change the preset to *Alert budget (50/day)*: recall rises to 99.2% but precision falls to 24% and the estimated
  net benefit drops; the box explains it only pays if a review costs under $2.02.
* Set the preset back to *Model default*.

## 5. Monitoring and the model (45 s)

* **Monitoring** tab: monthly recall stays above the 85% floor, with 95% intervals; the chi-square test finds no
  more month-to-month variation than chance. Drift share crossed 30% in August and December (seasonal spending),
  so retraining is recommended for review.
* **Model** tab: why HistGradientBoosting was chosen; SHAP importance; the explanation-quality check (removing
  the top-3 SHAP features drops the score far more than removing random ones); and how far the results transfer:
  at a 0.1% real-world fraud rate precision would be about 40%, which is why a shadow-mode trial comes first.

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
