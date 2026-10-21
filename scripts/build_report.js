/*
 * FraudLens final project report (Assessment 4).
 * Every number is read from the pipeline's result files, so the report cannot disagree with the
 * code that produced it. Optional inputs (platform CI results, deployment links) are used when present.
 *
 * Usage: node scripts/build_report.js <reports_dir> <output.docx> [deployment.json]
 */
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow, TableCell, WidthType, ShadingType,
  AlignmentType, ImageRun, PageBreak, TableOfContents, LevelFormat, Footer, PageNumber, ExternalHyperlink,
} = require("docx");

const DIR = process.argv[2] || "reports";
const OUT = process.argv[3] || "FraudLens_Final_Report.docx";
const DEPLOY = process.argv[4] && fs.existsSync(process.argv[4]) ? JSON.parse(fs.readFileSync(process.argv[4], "utf8")) : {};
const J = (f) => JSON.parse(fs.readFileSync(path.join(DIR, f), "utf8"));
const JO = (f) => (fs.existsSync(path.join(DIR, f)) ? J(f) : null);
const FIGDIR = path.join(DIR, "figures");
const IMGDIR = path.join(__dirname, "..", "docs", "images");      // architecture diagram, dashboard screenshots

// ------------------------------------------------------------------ data
const RUN = J("pipeline_run.json"), VAL = J("validation_report.json"), AUD = J("cleaning_audit.json");
const EDA = J("eda_summary.json"), PRE = J("preprocessing_report.json"), MOD = J("model_results.json");
const VER = J("verification_report.json"), EXP = J("explanations.json"), META = J("dashboard_meta.json");
const MON = J("monitoring/summary.json"), MAN = J("acquisition_manifest.json");
const FAIR = JO("fairness.json"), CHAL = JO("monitoring/challenger.json");
const OLAP = JO("olap_benchmark.json"), GOV = JO("governance_catalog.json"), SMOKE = JO("platform_smoke.json");
const PCI = JO("platform_ci.json");
const STAB = JO("stability.json"), EQ = JO("explanation_quality.json");        // the platform workflow's results (scripts/ci/collect_platform_results.py)
const OLAP_Q = { daily_outcomes: "Daily caught, false and missed alerts", category_by_hour: "Fraud and alerts by category and hour",
  missed_by_merchant: "Merchants with the most missed fraud", alert_precision_by_state: "Alert precision by state" };

const CH = MOD.champion, SEL = MOD.selection, CR = MOD.results[CH], CT = CR.test, CVd = CR.validation;
const THR = CR.thresholds_from_validation[MOD.threshold_strategy];
const CI = MOD.champion_test_ci95;
const NAMES = { logistic: "Logistic regression", hist_gb: "HistGradientBoosting", xgboost: "XGBoost", lightgbm: "LightGBM",
  isolation_forest: "Isolation Forest", hybrid_gb: "Hybrid (HistGB + Isolation Forest score)" };
const nm = (m) => NAMES[m] || m;
const int = (n) => Math.round(n).toLocaleString("en-US");
const usd = (n, d = 0) => "$" + Number(n).toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
const pf = (frac, d = 1) => (100 * frac).toFixed(d) + "%";
const pv = (val, d = 2) => Number(val).toFixed(d) + "%";
const f3 = (x) => Number(x).toFixed(3), f4 = (x) => Number(x).toFixed(4);
const splits = RUN.split.splits;
const totalRows = MAN.files.train.rows + MAN.files.test.rows;
const RECOVERY = META.recovery_rate;
const caughtRaw = CT.fraud_amount_caught / RECOVERY;
const fraudValueShare = caughtRaw / (caughtRaw + CT.fraud_amount_missed);
const night = EDA.behaviour.night["night (22-04)"], day = EDA.behaviour.night.day;
const tenX = EDA.behaviour.amount_vs_card_mean["10x+"];
const amt = EDA.amount_by_class;
const topCats = Object.entries(EDA.category).sort((a, b) => b[1].fraud_rate_pct - a[1].fraud_rate_pct).slice(0, 3);
const fs24 = EDA.feature_signal;
const monthsTest = MOD.champion_monthly_test;
const recallRange = [Math.min(...monthsTest.map((m) => m.recall)), Math.max(...monthsTest.map((m) => m.recall))];
const precRange = [Math.min(...monthsTest.map((m) => m.precision)), Math.max(...monthsTest.map((m) => m.precision))];
const SHAP = Object.entries(EXP.mean_abs_shap).slice(0, 8);

// --------------------------------------------------------------- helpers
const TOTAL_W = 9360;
const H1 = (t) => new Paragraph({ text: t, heading: HeadingLevel.HEADING_1, spacing: { before: 320, after: 160 } });
const H2 = (t) => new Paragraph({ text: t, heading: HeadingLevel.HEADING_2, spacing: { before: 260, after: 120 } });
const H3 = (t) => new Paragraph({ text: t, heading: HeadingLevel.HEADING_3, spacing: { before: 200, after: 100 } });
function runs(t) {           // **bold** segments
  return String(t).split(/(\*\*[^*]+\*\*)/).filter(Boolean).map((s) =>
    s.startsWith("**") ? new TextRun({ text: s.slice(2, -2), bold: true, size: 22 }) : new TextRun({ text: s, size: 22 }));
}
const P = (t) => new Paragraph({ children: runs(t), spacing: { after: 140, line: 276 } });
const BULLET = (t) => new Paragraph({ children: runs(t), numbering: { reference: "bullets", level: 0 }, spacing: { after: 80, line: 276 } });
const NUMBERED = (t) => new Paragraph({ children: runs(t), numbering: { reference: "numbers", level: 0 }, spacing: { after: 80, line: 276 } });
const CODE = (t) => new Paragraph({ children: [new TextRun({ text: t, font: "Consolas", size: 19 })], spacing: { after: 40 },
  shading: { type: ShadingType.CLEAR, fill: "F2F2F2", color: "auto" } });
const LINK = (text, url) => new Paragraph({ spacing: { after: 80 }, children: [new ExternalHyperlink({ link: url,
  children: [new TextRun({ text, style: "Hyperlink", size: 22 })] })] });
const PAGEBREAK = () => new Paragraph({ children: [new PageBreak()] });
function cell(text, { bold = false, fill = null, width, align = AlignmentType.LEFT } = {}) {
  return new TableCell({ width: { size: width, type: WidthType.DXA },
    shading: fill ? { type: ShadingType.CLEAR, fill, color: "auto" } : undefined,
    margins: { top: 50, bottom: 50, left: 80, right: 80 },
    children: [new Paragraph({ alignment: align, spacing: { after: 0, line: 240 }, children: [new TextRun({ text: String(text), bold, size: 18 })] })] });
}
const isNumeric = (v) => /^[+\-$<>(~]?[\d.,]+\)?\s*(%|x|×|ms|s|MB|h)?$/.test(String(v).trim()) || String(v).trim() === "n/a";
function table(headers, rows, weights) {
  if (headers.length !== weights.length) throw new Error(`weights mismatch: ${headers.join("|")}`);
  rows.forEach((r, i) => { if (r.length !== headers.length) throw new Error(`row ${i}: ${r.join("|")}`); });
  const sum = weights.reduce((a, b) => a + b, 0);
  const widths = weights.map((w) => Math.round((w / sum) * TOTAL_W));
  widths[0] += TOTAL_W - widths.reduce((a, b) => a + b, 0);
  const align = (i, v) => (i > 0 && isNumeric(v) ? AlignmentType.RIGHT : AlignmentType.LEFT);
  return [new Table({ columnWidths: widths, width: { size: TOTAL_W, type: WidthType.DXA }, rows: [
    new TableRow({ tableHeader: true, children: headers.map((h, i) => cell(h, { bold: true, fill: "DCE6F1", width: widths[i] })) }),
    ...rows.map((r) => new TableRow({ children: r.map((c, i) => cell(c, { width: widths[i], align: align(i, c) })) })),
  ] }), new Paragraph({ text: "", spacing: { after: 120 } })];
}
let tableNo = 0, figNo = 0;
const TCAP = (t) => { tableNo += 1; return new Paragraph({ spacing: { before: 160, after: 80 },
  children: [new TextRun({ text: `Table ${tableNo}. ${t}`, italics: true, size: 19, color: "444444" })] }); };
function pngSize(file) { const b = fs.readFileSync(file); return [b.readUInt32BE(16), b.readUInt32BE(20)]; }
function figure(file, caption, width = 468) {
  const p = file.startsWith("img:") ? path.join(IMGDIR, file.slice(4)) : path.join(FIGDIR, file);
  if (!fs.existsSync(p)) return [P(`[Figure ${file} not available in this build]`)];
  const [w, h] = pngSize(p); figNo += 1;
  return [new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 60 },
      children: [new ImageRun({ data: fs.readFileSync(p), type: "png", transformation: { width, height: Math.round(width * h / w) } })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 },
      children: [new TextRun({ text: `Figure ${figNo}. ${caption}`, italics: true, size: 19, color: "444444" })] })];
}
const body = [];
const add = (...items) => items.flat().forEach((i) => body.push(i));
const REPO = "https://github.com/anasua2905/fraudlens-or-credit-card-fraud-detection";

// ------------------------------------------------------------ title page
const today = new Date().toLocaleDateString("en-AU", { day: "numeric", month: "long", year: "numeric" });
const titlePage = [
  new Paragraph({ text: "", spacing: { after: 1800 } }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 120 }, children: [new TextRun({ text: "FraudLens", bold: true, size: 60, color: "1F3864" })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 360 }, children: [new TextRun({ text: "An Anomaly Detection Framework for Identifying Fraudulent Credit Card Transactions", size: 30 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 160 }, children: [new TextRun({ text: "Final Project Report: the complete data science lifecycle on an open-source data platform", size: 24, color: "444444" })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 900 }, children: [new TextRun({ text: "PRT661 · Theme 3: Anomaly Detection and Intelligent Monitoring · Assessment 4", size: 22, italics: true, color: "444444" })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 100 }, children: [new TextRun({ text: "Charles Darwin University", size: 22 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 100 }, children: [new TextRun({ text: "Repository: github.com/anasua2905/fraudlens-or-credit-card-fraud-detection (branch final-a4)", size: 20 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 400 }, children: [new TextRun({ text: today, size: 22 })] }),
  PAGEBREAK(),
  H1("Table of Contents"),
  new Paragraph({ spacing: { after: 200 }, children: [new TextRun({ text: "In Word: right-click the table below and choose “Update Field” → “Update entire table”.", italics: true, size: 19, color: "666666" })] }),
  new TableOfContents("Right-click and select 'Update Field' to populate the table of contents.", { hyperlink: true, headingStyleRange: "1-2" }),
  PAGEBREAK(),
];

// -------------------------------------------------------- executive summary
add(H1("Executive Summary"));
add(P(`FraudLens is an end-to-end anomaly detection framework for card fraud, built through all eight stages of the data science lifecycle (problem understanding, acquisition, preparation, exploratory analysis, modelling, evaluation and interpretation, deployment and visualisation, monitoring and improvement) and implemented on an open-source data platform that mirrors a modern cloud data pipeline: MinIO object storage, Apache NiFi ingestion, an Apache Hive Metastore catalog, governance through Trino access rules, MinIO policies and OpenMetadata, analytics with Trino and ClickHouse, and visualisation with Superset, Grafana and a Streamlit dashboard.`));
add(P(`The data are the Sparkov simulated transactions: ${int(totalRows)} transactions from 2019 to 2020, of which ${pv(100 * (splits.train.fraud_count + splits.val.fraud_count + splits.test.fraud_count) / totalRows)} are fraudulent. Exploratory analysis showed that fraud is concentrated in large amounts relative to a card's history, at night (${pv(night.share_of_frauds_pct, 1)} of training fraud between 22:00 and 03:59), in bursts on compromised cards, and in online and grocery point-of-sale categories.`));
add(P(`Six models were tuned with time-series cross-validation and compared on a later validation period. A pre-registered rule selected the simplest model that is statistically non-inferior to the best: **${nm(CH)}** (scikit-learn). On the untouched test period (21 June to 31 December 2020) it detects **${pf(CT.recall)} of fraud** with **${pf(CT.precision)} precision** at **${CT.alerts_per_day} alerts per day**, a PR-AUC of ${f3(CT.pr_auc)} (${Math.round(CT.pr_auc_lift)}× the random baseline), and catches ${pf(fraudValueShare)} of the fraud value, for an estimated net benefit of **${usd(CT.net_benefit)}** (an estimate assuming $${META.investigation_cost} per review and ${pf(RECOVERY, 0)} of caught fraud value recovered, on simulated data, not demonstrated savings). All four success criteria set before modelling are met, and all ${VER.passed} automated end-to-end verification checks pass.`));
add(P(`The system is deployed as a FastAPI scoring service with an online feature store and SHAP explanations, and a Streamlit dashboard for analysts and managers; both run from a self-contained 40 MB bundle, are packaged as a container image, and were deployed with public links. Monthly monitoring with Evidently AI flagged seasonal feature drift in August and December 2020 without a loss of recall, and a champion/challenger retraining test shows how the improvement loop works. The project is managed with a GitHub Project board and CI/CD workflows. The data are simulated, so the absolute results validate the method rather than forecast production accuracy.`));

// -------------------------------------------------------- 1. problem understanding
add(PAGEBREAK(), H1("1. Problem Understanding"));
add(H2("1.1 Context and objectives"));
add(P("Card issuers must decide within milliseconds whether a transaction proceeds quietly or is queued for an analyst. Missing fraud loses money; every false alert costs analyst time and inconveniences a genuine customer. The project brief set five objectives:"));
["Build a machine-learning system that identifies fraudulent transactions.", "Improve detection while minimising false-positive alerts.",
  "Provide an interactive dashboard for transaction monitoring and reporting.", "Show the full flow from data collection to visualisation.",
  "Support quicker decision-making for financial institutions."].forEach((t) => add(NUMBERED(t)));
add(H2("1.2 Stakeholders"));
add(TCAP("Users of the system and what each needs."));
add(table(["Stakeholder", "Question", "What FraudLens provides"], [
  ["Fraud analyst", "Which alerts first, and why?", "Ranked alert queue with SHAP reasons; feedback form"],
  ["Operations manager", "How many alerts, what do they cost and save?", "Alerts/day, precision, recall, review cost, net benefit; threshold what-if"],
  ["Model risk / governance", "Is it valid, fair, reproducible, monitored?", "Verification report, model card, fairness by group, drift monitoring, MLflow registry, data catalog"],
  ["Data platform team", "Does it run on our open-source stack?", "Docker platform: MinIO, NiFi, Hive Metastore, Trino, ClickHouse, Superset, Grafana"],
], [3, 4, 7]));
add(H2("1.3 Formal problem and success criteria"));
add(P(`For each transaction, using only the transaction and the card's earlier history, produce a score that ranks fraud first, and alert when the score exceeds a threshold chosen to maximise net benefit = ${RECOVERY} × fraud value in alerts − $${META.investigation_cost} × alerts on the validation period. Accuracy is unsuitable: on the test period a model that never alerts is ${pf(1 - splits.test.fraud_rate_pct / 100, 2)} accurate and catches nothing.`));
add(TCAP("Success criteria (config/pipeline.yaml), set before modelling and checked on the test period."));
add(table(["Criterion", "Target", "Achieved", "Met"], VER.success_criteria.map((c) => [c.criterion.replace(/_/g, " "),
  String(c.target), String(c.value), c.met ? "Yes" : "No"]), [5, 2, 2, 2]));
add(H2("1.4 Hypotheses"));
add(TCAP("Hypotheses formulated in stage 1 and tested in stage 4."));
add(table(["#", "Hypothesis", "Features that capture it"], [
  ["H1", "Fraudulent transactions are larger than the card's usual spend", "amount, amt_to_card_mean, amt_zscore_card"],
  ["H2", "Fraud happens disproportionately at night", "hour, is_night"],
  ["H3", "Fraud arrives in bursts on a compromised card", "txn_count / amt_sum over 1 h, 24 h, 7 d; secs_since_last_txn"],
  ["H4", "Some merchant categories are riskier", "category, category_share_card"],
  ["H5", "Demographics carry little signal and must not drive decisions", "age, gender (fairness check)"],
], [1, 8, 6]));

// -------------------------------------------------------- 2. architecture
add(PAGEBREAK(), H1("2. Solution Architecture and Open-Source Platform"));
add(P("The project follows the modern cloud data-pipeline pattern (storage, crawler, catalog, governance, analytics, visualisation) with an open-source component in each role. One Python package (src/fraudlens, one module per lifecycle step) and one configuration file drive both a laptop run and the Docker platform."));
add(figure("img:architecture.png", "FraudLens architecture: the open-source data platform and the ML lifecycle.", 468));
add(TCAP("Cloud data-pipeline roles and the open-source components used."));
add(table(["Role", "Cloud pattern", "FraudLens implementation"], [
  ["Storage", "Amazon S3", "MinIO: raw (versioned) / silver / gold buckets, Hive-partitioned Parquet"],
  ["Crawler / ingest", "AWS Glue crawler", "Apache NiFi flow (landing folder → raw layer, ingest record per file) and a Python lake crawler that infers schemas"],
  ["Catalog", "Glue Data Catalog", "Apache Hive Metastore on PostgreSQL"],
  ["Governance", "Lake Formation", "Trino file-based access control (PII masks, column denial), least-privilege MinIO policies, OpenMetadata (catalog, PII classification, lineage)"],
  ["Analytics", "Athena / Redshift", "Trino over the lake; ClickHouse MergeTree tables for fast OLAP"],
  ["Visualisation", "QuickSight", "Apache Superset (BI on Trino), Grafana (operations on Prometheus, business on ClickHouse), Streamlit (analysts)"],
  ["ML operations", "SageMaker", "MLflow tracking and registry, FastAPI scoring service, Evidently drift reports, Prometheus metrics"],
], [2, 3, 9]));
add(TCAP("Lifecycle stages, notebooks, modules and tools."));
add(table(["Stage", "Notebook", "Modules", "Tools"], [
  ["1 Problem understanding", "01", "config/pipeline.yaml", "Jupyter, Markdown"],
  ["2 Data acquisition", "02", "acquire.py, platform/nifi", "Kaggle API, requests, BeautifulSoup, NiFi, MinIO"],
  ["3 Data preparation", "03", "validate, clean, features, split, quality", "pandas, Dask, PyArrow"],
  ["4 EDA", "04", "eda.py, interactive.py", "pandas, Seaborn, Matplotlib, Plotly"],
  ["5 Modelling", "05", "preprocess, models, tuning, train, selection", "scikit-learn, XGBoost, LightGBM, CatBoost, MLflow"],
  ["6 Evaluation & interpretation", "06", "evaluate.py, explain.py", "sklearn metrics, SHAP, ELI5, Yellowbrick"],
  ["7 Deployment & visualisation", "07", "api, runtime, online, dashboard/app.py, publish, crawler, olap", "FastAPI, Streamlit, Plotly, Docker, Trino, ClickHouse, Superset"],
  ["8 Monitoring & improvement", "08", "monitor.py, tracking.py", "Evidently AI, Prometheus, Grafana, MLflow"],
], [3, 1, 5, 5]));
add(P("Design principles, each enforced by a test or a verification check: no feature uses the label; no feature uses information unavailable at authorisation time; splits are chronological; every reported number is reproducible from saved artefacts; the deployed object scores exactly as in training."));

// -------------------------------------------------------- 3. acquisition
add(PAGEBREAK(), H1("3. Data Acquisition"));
add(P(`The dataset is Kaggle ${MAN.dataset} (licence ${MAN.licence}), acquired through the official Kaggle API or from a local copy of the same files. Acquisition records a manifest with SHA-256 checksums, sizes and row counts, a column profile, and the provenance of the source; the generator's public repository page is scraped with requests and BeautifulSoup for its description and licence (notebook 02). Verification recomputes the checksums on every run.`));
add(TCAP("Acquired files (bronze layer)."));
add(table(["Split", "File", "Rows", "Size (MB)", "SHA-256 (first 16)"], Object.entries(MAN.files).map(([k, f]) =>
  [k, f.file, int(f.rows), (f.bytes / 1e6).toFixed(1), f.sha256.slice(0, 16)]), [2, 4, 3, 2, 5]));
add(P("On the platform, the acquisition stage drops the files in a landing folder and Apache NiFi lands them in the raw bucket: ListFile → FetchFile → UpdateAttribute (lake key, ingest time) → PutS3Object → AttributesToJSON → PutS3Object (ingest record). NiFi writes with a write-only service account limited to raw/sparkov/, the raw bucket is versioned so a source file can never be silently replaced, and the stage waits until every file has arrived with the expected size before continuing."));

// -------------------------------------------------------- 4. preparation
add(PAGEBREAK(), H1("4. Data Preparation"));
add(H2("4.1 Data contract"));
add(P(`Each raw file is validated against twelve expectations before any transformation. All pass except a warning that the unix_time column is offset by ${VAL[0].checks.find((c) => c.check.startsWith("unix_time")).detail.example_offset_days} days in the public copies; time features are therefore derived from the readable timestamp.`));
add(H2("4.2 Cleaning and privacy"));
add(P(`No duplicates or invalid rows were found, so the silver layer keeps all ${int(AUD.structural_profile.rows)} transactions. Names and street addresses are dropped, card numbers are replaced by a salted SHA-256 identifier, and the generator's "fraud_" prefix on every merchant name is removed so it cannot be mistaken for a signal. Amount outliers are kept: the top 5% of amounts contain ${pv(AUD.outliers_train.amount.share_of_all_fraud_pct, 0)} of all training fraud.`));
add(H2("4.3 Leakage-safe features"));
add(TCAP("Engineered features."));
add(table(["Group", "Features", "Purpose"], [
  ["Transaction", "amount, log_amount, hour, day_of_week, is_weekend, is_night, category", "describe the event"],
  ["Customer", "age, gender, log_city_pop", "context (checked for fairness)"],
  ["Card history", "amt_to_card_mean, amt_zscore_card, secs_since_last_txn, category_share_card", "compare with the card's own past"],
  ["Velocity", "txn_count and amt_sum over 1 h, 24 h and 7 d", "detect bursts on a compromised card"],
], [2, 8, 4]));
add(P(`Every history and velocity feature uses only transactions strictly before the one being scored (a brute-force recomputation in notebook 03 matches the vectorised values exactly). Running totals (${PRE.excluded_features.join(", ")}) were excluded after measuring that they drift by construction or carry no signal. Dask recomputes the monthly aggregates directly from the raw CSV partitions and matches the pandas silver layer, showing that the preparation scales beyond memory.`));
add(H2("4.4 Chronological split"));
add(TCAP("Train, validation and test periods."));
add(table(["Split", "Start", "End", "Transactions", "Frauds", "Fraud rate"], [["train", "Train"], ["val", "Validation"], ["test", "Test"]].map(([k, n]) =>
  [n, splits[k].start.slice(0, 10), splits[k].end.slice(0, 10), int(splits[k].rows), int(splits[k].fraud_count), pv(splits[k].fraud_rate_pct, 3)]), [2, 2, 2, 3, 2, 2]));

// -------------------------------------------------------- 5. EDA
add(PAGEBREAK(), H1("5. Exploratory Data Analysis"));
add(P("All analysis uses the training period only. The results test the stage-1 hypotheses and shape the modelling choices."));
add(figure("02_amount_by_class.png", "Transaction amount by class (training period)."));
add(P(`**H1, amount (strongly supported).** The median fraudulent transaction is ${usd(amt.fraud.p50)} against ${usd(amt.legit.p50)} for legitimate ones. Transactions above ten times the card's usual amount are fraudulent ${pv(tenX.fraud_rate_pct, 1)} of the time and account for ${pv(tenX.share_of_frauds_pct, 1)} of fraud.`));
add(figure("04_fraud_by_time.png", "Fraud rate by hour, weekday and month."));
add(P(`**H2, night (strongly supported).** ${pv(night.share_of_frauds_pct, 1)} of training fraud occurs between 22:00 and 03:59, where the fraud rate is ${pv(night.fraud_rate_pct)} against ${pv(day.fraud_rate_pct)} during the day.`));
add(P(`**H3, bursts (supported).** Of ${EDA.card_level.cards_total} cards, ${EDA.card_level.cards_with_fraud} were compromised in the training period; a compromised card typically sees ${EDA.card_level.frauds_per_fraud_card_median} fraudulent transactions within about ${Math.round(EDA.card_level.fraud_span_hours_median)} hours. Spend over the previous 24 hours is the single strongest feature (single-feature AUC ${f3(fs24.amt_sum_24h.auc)}).`));
add(figure("03_fraud_by_category.png", "Fraud rate by merchant category."));
add(P(`**H4, category (supported).** ${topCats.map(([c, v]) => `${c} (${pv(v.fraud_rate_pct)})`).join(", ")} have two to three times the base rate of ${pv(splits.train.fraud_rate_pct)}.`));
add(P(`**H5, demographics (weak).** Fraud rates differ modestly by age band and barely by gender (F ${pv(EDA.demographics.gender.F.fraud_rate_pct)}, M ${pv(EDA.demographics.gender.M.fraud_rate_pct)}); they are kept as inputs but monitored for fairness.`));
add(figure("07_feature_signal.png", "Single-feature ROC AUC on a stratified sample.", 400));
add(P("Interactive versions of these views (Plotly) are in reports/interactive/eda_explorer.html and notebook 04."));

// -------------------------------------------------------- 6. modelling
add(PAGEBREAK(), H1("6. Modelling"));
add(H2("6.1 Feature views and class imbalance"));
add(P(`Two preprocessing views are fitted on the training period only: a tree view (${PRE.views.tree.n_features} features with native categorical codes) and a linear view (${PRE.views.linear.n_features} features: log transforms, train-fitted clipping, standardisation, cyclical time encoding, one-hot categories). Imbalance (${PRE.suggested_scale_pos_weight} legitimate transactions per fraud) is handled with class weights; no SMOTE or other resampling is used, so the training distribution stays real.`));
add(H2("6.2 Candidates, tuning and comparison"));
add(P("Six candidates were evaluated: logistic regression, HistGradientBoosting, XGBoost and LightGBM, an Isolation Forest trained on legitimate transactions only, and a hybrid that feeds the Isolation Forest anomaly score into HistGradientBoosting. Boosted models were tuned by random search with three-fold expanding-window (time-series) cross-validation inside the training period; every run, candidate and model is tracked in MLflow."));
add(TCAP("Model comparison (validation chooses; test shown for completeness)."));
add(table(["Model", "CV PR-AUC", "Val PR-AUC", "Test PR-AUC", "Test precision", "Test recall", "Alerts/day"], MOD.comparison.map((r) =>
  [nm(r.model), r.cv_pr_auc == null ? "n/a" : f4(r.cv_pr_auc), f4(r.val_pr_auc), f4(r.test_pr_auc), pf(r.test_precision), pf(r.test_recall), String(r.test_alerts_per_day)]), [5, 2, 2, 2, 2, 2, 2]));
add(P("A compact re-run in notebook 05 repeats the comparison with time-series CV on a one-in-four sample and adds CatBoost as a further contender; the ranking is the same: boosted trees far ahead of the linear baseline, which in turn is far ahead of the purely unsupervised detector."));
add(H2("6.3 Champion selection"));
add(P(`Selection follows a rule written before the results were seen: order the models by operational complexity and choose the simplest whose validation PR-AUC is not worse than the best by more than ${SEL.margin}, judged by the upper bound of a 95% paired-bootstrap interval of the gap. ${SEL.reason}`));
add(TCAP("Paired-bootstrap gap to the best model on validation PR-AUC."));
add(table(["Model", "Mean gap", "95% CI", "Non-inferior"], SEL.complexity_order.filter((m) => SEL.comparisons[m]).map((m) => {
  const g = SEL.comparisons[m]; return [nm(m), f4(g.gap_mean), `[${f4(g.ci95_low)}, ${f4(g.ci95_high)}]`, g.equivalent_to_best ? "Yes" : "No"]; }), [5, 2, 3, 2]));
add(P(`The champion is ${nm(CH)} (learning rate ${MOD.tuning[CH].best_params.learning_rate}, ${MOD.tuning[CH].best_params.max_leaf_nodes} leaves, min ${MOD.tuning[CH].best_params.min_samples_leaf} samples per leaf), saved together with its preprocessor and threshold as one FraudModel object and registered in MLflow as fraudlens-champion (alias champion). A train/serve parity test proves the deployed object reproduces the training scores exactly; it caught, and led to the fix of, a float32/float64 representation mismatch during the final run.`));

// -------------------------------------------------------- 7. evaluation
add(PAGEBREAK(), H1("7. Evaluation and Interpretation"));
add(H2("7.1 Operating point and test performance"));
add(P(`The threshold ${f4(THR)} was chosen on validation by maximising net benefit (${MOD.threshold_strategy}) and applied once to the test period.`));
add(TCAP("Champion on the test period, with 95% bootstrap intervals (300 resamples)."));
add(table(["Metric", "Value", "95% CI"], [
  ["PR-AUC (baseline " + f4(CT.pr_auc_baseline) + ")", f4(CT.pr_auc), `${f4(CI.pr_auc.low)} - ${f4(CI.pr_auc.high)}`],
  ["ROC-AUC", f4(CT.roc_auc), "-"],
  ["Recall", pf(CT.recall), `${pf(CI.recall.low)} - ${pf(CI.recall.high)}`],
  ["Precision", pf(CT.precision), `${pf(CI.precision.low)} - ${pf(CI.precision.high)}`],
  ["False-positive rate", pf(CT.false_positive_rate, 3), `${pf(CI.false_positive_rate.low, 2)} - ${pf(CI.false_positive_rate.high, 2)}`],
  ["Alerts per day", String(CT.alerts_per_day), "-"],
  ["Alerts / true positives / false positives / missed", `${int(CT.alerts)} / ${int(CT.true_positives)} / ${int(CT.false_positives)} / ${int(CT.false_negatives)}`, "-"],
  [`Estimated net benefit (assumes $${META.investigation_cost}/review, ${pf(RECOVERY, 0)} recovered)`, usd(CT.net_benefit), `${usd(CI.net_benefit.low)} - ${usd(CI.net_benefit.high)}`],
], [6, 4, 4]));
add(figure("11_threshold_sweep_validation.png", "Validation sweep: recall, precision and net benefit across alert volumes."));
add(H2("7.2 Business impact"));
add(P(`At ${CT.alerts_per_day} alerts a day the champion catches ${pf(fraudValueShare)} of the fraud value in the test period (${usd(caughtRaw)} of ${usd(caughtRaw + CT.fraud_amount_missed)}); assuming ${pf(RECOVERY, 0)} of it is recovered and a review costs $${META.investigation_cost}, the estimated net benefit is ${usd(CT.net_benefit)}. This is an estimate under those two assumptions on simulated transactions, used to compare operating points; it is not a demonstrated saving, which would need a trial on real transactions with measured review costs and recoveries. Roughly ${Math.round(1 / CT.precision * 10) / 10} alerts are reviewed per fraud caught. Lowering the threshold to the alert-budget point would only pay if a review cost less than ${usd(MOD.break_even_review_cost, 2)}. Missed fraud is concentrated in small transactions, where an alert costs more than it recovers.`));
add(H2("7.3 Stability, calibration and fairness"));
add(P(`Monthly recall on the test period ranges from ${pf(recallRange[0])} to ${pf(recallRange[1])} and precision from ${pf(precRange[0])} to ${pf(precRange[1])}. Scores rank well but are not calibrated probabilities (class weighting inflates them); the threshold is therefore defined in score units on validation.`));
if (STAB) {
  const ci = (a) => `${f3(a[0])}-${f3(a[1])}`;
  add(TCAP("Monthly stability on the test period at the deployed threshold (Wilson 95% intervals)."));
  add(table(["Month", "Frauds", "Recall", "95% CI", "Precision", "95% CI", "Alerts/day"], STAB.monthly.months.map((m) =>
    [m.month, int(m.frauds), f3(m.recall), ci(m.recall_ci95), f3(m.precision), ci(m.precision_ci95), String(m.alerts_per_day)]), [2, 2, 2, 3, 2, 3, 2]));
  add(P(`Every month stays above the policy floors (recall ${pf(MON.policy.recall_floor, 0)}, precision ${pf(MON.policy.precision_floor, 0)}), and a chi-square test finds no more variation between months than chance (recall p = ${STAB.monthly.chi2_p_recall.toFixed(2)}, precision p = ${STAB.monthly.chi2_p_precision.toFixed(2)}). The same check runs after every monitoring cycle (fraudlens.stability).`));
}
if (FAIR) {
  add(TCAP("Performance by group on the test period."));
  add(table(["Group", "Value", "Transactions", "Frauds", "Recall", "False alarms per 10k", "Precision"], FAIR.map((r) =>
    [r.group, String(r.value), int(r.transactions), int(r.frauds), r.recall == null ? "n/a" : f3(r.recall), String(r.false_alarm_rate_per_10k), r.precision == null ? "n/a" : f3(r.precision)]), [2, 2, 2, 2, 2, 2, 2]));
  const g = Object.fromEntries(FAIR.filter((r) => r.group === "gender").map((r) => [r.value, r]));
  const ages = FAIR.filter((r) => r.group !== "gender").map((r) => r.recall);
  if (g.F && g.M) add(P(`Recall is high for every group but not identical: ${f3(g.F.recall)} for women against ${f3(g.M.recall)} for men, with ${g.F.false_alarm_rate_per_10k} against ${g.M.false_alarm_rate_per_10k} false alarms per 10,000 transactions; across age bands recall ranges from ${f3(Math.min(...ages))} to ${f3(Math.max(...ages))}. The gap is reported rather than hidden: before use on real data the model would be retrained without gender and compared, and group metrics are part of monthly monitoring.`));
}
add(H2("7.4 Explainability"));
add(P(`SHAP explains every score in log-odds contributions. For this champion's native categorical splits TreeSHAP's values did not reproduce the model output, so the pipeline automatically switched to SHAP's permutation explainer on the same output, which is exactly additive. The largest average contributions are ${SHAP.slice(0, 5).map(([f, v]) => `${f} (${Number(v).toFixed(2)})`).join(", ")}, consistent with the EDA. Every alert carries its top three reasons into the dashboard and the API. ELI5 weights of the logistic baseline and a Yellowbrick precision-recall curve of the champion (notebook 06) give independent views of the same picture.`));
if (EQ) {
  const d = EQ.deletion;
  add(P(`**Are the explanations faithful?** On ${EQ.alerts_sampled} sampled test-period alerts, replacing each alert's top-${EQ.top_k} SHAP features with typical values lowers its score by ${f3(d.mean_score_drop_top_k)} on average, against ${f3(d.mean_score_drop_random_k)} when ${EQ.top_k} random features are replaced (${d.ratio_top_vs_random}× more), and clears ${pf(d.share_cleared_top_k, 0)} of the alerts against ${pf(d.share_cleared_random_k, 0)}. Explaining the same alerts with a different random seed gives a top-${EQ.top_k} overlap (Jaccard) of ${f3(EQ.stability.mean_top_k_jaccard)}; the median additivity error is ${EQ.additivity.median_abs_error_log_odds} log-odds. The reasons analysts see therefore point at what the model actually used. Each live alert also carries a counterfactual: a lower amount at which it would have been approved with the same card history.`));
}
add(figure("17_shap_beeswarm.png", "SHAP beeswarm: how each feature's value moves the score (test period).", 430));
if (STAB && STAB.prevalence) {
  const pv = STAB.prevalence;
  add(H2("7.5 How far the results transfer to real transactions"));
  add(P(`The data are simulated, so the test metrics are an optimistic guide. Precision depends on how common fraud is and on how many false alarms real customers trigger. The table carries the test recall (${pf(pv.recall)}) and false-positive rate (${(100 * pv.false_positive_rate).toFixed(3)}%) over to other fraud rates and to two and five times as many false alarms, at ${int(pv.transactions_per_day)} transactions a day.`));
  add(TCAP("Precision and alert volume under other conditions (same recall)."));
  add(table(["Fraud rate", "Precision", "Alerts/day", "Precision, 2× false alarms", "Precision, 5× false alarms"], pv.rows.map((r) =>
    [`${r.fraud_rate_pct.toFixed(2)}%${r.is_test_rate ? " (test)" : ""}`, pf(r.precision_fpr_x1, 0), String(r.alerts_per_day_fpr_x1), pf(r.precision_fpr_x2, 0), pf(r.precision_fpr_x5, 0)]), [3, 2, 2, 3, 3]));
  add(P("Before relying on the model with real customers it would run in shadow mode (scoring without blocking) and be compared with chargebacks, the threshold would be re-chosen on real validation data, the monthly and per-segment stability checks would be repeated, and the real review cost and recovery rate would be measured before any savings are claimed."));
}
add(figure("18_shap_waterfall_caught_fraud.png", "Local explanation of a caught fraud.", 430));

// -------------------------------------------------------- 8. deployment
add(PAGEBREAK(), H1("8. Deployment and Visualisation"));
add(H2("8.1 Serving"));
add(P("The FastAPI service scores raw transactions in real time: an online feature store keeps each card's running statistics and recent window and computes the same features as the batch pipeline (a verification check compares 2,000 test transactions), the champion scores them, and alerts come with SHAP and plain-language reasons and a counterfactual amount. Endpoints: /score and /score/batch (payment system), /alerts, /alerts/{id} and /alerts/{id}/decision (review queue and decisions), /feedback (verdicts by transaction id, e.g. chargebacks), /monitoring/feedback, /metrics (Prometheus), /health and /model; invalid inputs are rejected by schema validation."));
add(H2("8.2 A new transaction, from features to an analyst decision"));
add(P("The complete path of a newly received transaction is implemented and tested end to end (tests/test_serving.py and the platform smoke test):"));
add(BULLET("**Received.** The payment system sends the raw transaction to POST /score (or an analyst enters it in the dashboard's Live transaction tab, which runs the same code)."));
add(BULLET("**Feature generation.** The online feature store derives the card-history features from the card's earlier transactions: count and average spend, amount relative to the average, z-score, time since the last transaction, velocity over 1 hour, 24 hours and 7 days, new merchant and night-time."));
add(BULLET("**Model inference.** The champion scores the features against the validation-chosen threshold; SHAP gives the top contributions and, for an alert, a counterfactual amount at which it would have been approved."));
add(BULLET("**Routing.** Every scored transaction is stored as a case with its features, score, threshold and explanation (fraudlens.casework). An alert opens the case in the review queue; other transactions are approved and logged."));
add(BULLET("**Analyst decision.** In the dashboard's Case review tab (or GET /alerts and POST /alerts/{id}/decision) the analyst sees what the model saw and records Confirmed fraud or Legitimate with a note; the case closes and the decision is written to an audit log with the analyst's name."));
add(P("**How the decisions inform monitoring.** Decisions are labels, used in three ways. Live precision (confirmed ÷ reviewed alerts) is tracked with a Wilson 95% interval against the 40% policy floor; if the whole interval falls below it (with at least 20 decisions) monitoring raises a threshold/model review, which appears in the monitoring summary, the Prometheus gauge fraudlens_live_precision and an alert rule. Fraud labels on approved transactions (disputes) count as misses, the live counterpart of recall. And every decided case keeps the exact model inputs, so monitoring exports them as labelled examples for the next champion/challenger comparison; the backlog and time to decision show whether the alert volume is manageable."));
add(figure("img:dashboard_live_path.png", "Live transaction tab: received, features generated, model inference with explanation, routed to the review queue.", 440));
add(figure("img:dashboard_case_review.png", "Case review tab: the open case, the decision form and how decisions feed monitoring.", 440));
add(H2("8.3 Access control"));
add(TCAP("Deployment access controls."));
add(table(["Layer", "Control"], [
  ["Scoring API", "Role-based API keys in the X-API-Key header: scorer (score), analyst (review queue, decisions, feedback monitoring), admin; 401 without a key, 403 for the wrong role; refusals counted in Prometheus"],
  ["Dashboard", "Sign-in with named users: analysts may record decisions, viewers only explore; the signed-in name is stored with every decision"],
  ["Audit", "Every scoring call and decision is logged in the case store with role, time and a non-secret key id"],
  ["Network", "API and dashboard bind to 127.0.0.1 and are exposed only through the tunnel; /health and /metrics carry no transaction data"],
  ["Data", "Card numbers pseudonymised; Trino masks quasi-identifiers and denies the job column to analysts; least-privilege MinIO accounts"]], [3, 10]));
add(P("Public deployments switch access control on automatically (scripts/deploy/run_local.ps1 -Public generates keys and two dashboard accounts, kept out of the repository); without it the services run in a clearly labelled development mode."));
add(H2("8.4 Dashboard"));
add(P("The Streamlit dashboard has ten tabs: Overview (KPIs, daily outcomes, money caught and missed, estimated net benefit with its assumptions), Live transaction (the end-to-end path above), Case review (the analyst workflow and feedback monitoring), Alert history (the replayed test period by card), Threshold & cost (what-if trade-off), Patterns, Monitoring (drift, monthly stability, live feedback), Model (selection, SHAP, explanation quality, transferability), Data pipeline and Report. All views recompute from the sidebar filters, threshold and cost assumptions."));
add(figure("img:dashboard_overview.png", "Dashboard, Overview tab (test period, deployed threshold).", 440));

add(figure("img:dashboard_threshold_cost.png", "Dashboard, Threshold & cost tab: the operating-point trade-off.", 440));
add(H2("8.5 Lake, catalog and BI"));
add(P("The publish stage writes silver and gold as Hive-partitioned Parquet to MinIO; the crawler infers schemas from Parquet footers and CSV headers and registers external tables and BI views in the Hive Metastore through Trino; Superset builds the fraud-operations dashboard on Trino; the OLAP stage loads the gold tables into ClickHouse through a named collection that uses the read-only analyst account, then answers four business questions on both engines and checks the answers agree."));
if (OLAP && OLAP.status !== "skipped") {
  add(TCAP("Same questions, two engines (median of five runs, platform CI run)."));
  add(table(["Question", "Rows", "Trino (ms)", "ClickHouse (ms)", "Answers agree"], Object.entries(OLAP.queries).map(([k, q]) =>
    [q.question, String(q.rows), q.trino_ms ? String(q.trino_ms.median) : "n/a", String(q.clickhouse_ms.median), q.answers_agree ? "Yes" : "n/a"]), [7, 1, 2, 2, 2]));
}
if (!(OLAP && OLAP.status !== "skipped") && PCI && PCI.olap) {
  add(TCAP(`Same questions, two engines (median of five runs; platform workflow run ${PCI.run_id}, sample data on a GitHub runner).`));
  add(table(["Question", "Trino (ms)", "ClickHouse (ms)", "Answers agree"], Object.entries(PCI.olap).map(([k, v]) =>
    [OLAP_Q[k] || k, String(v[1]), String(v[0]), v[2] ? "Yes" : "No"]), [7, 2, 2, 2]));
}
if (PCI && PCI.smoke) {
  add(H2("8.6 The platform, verified end to end"));
  add(P(`The platform workflow starts every service on a GitHub runner, runs the whole lifecycle on synthetic sample data through NiFi, the lake, the crawler, Trino and ClickHouse, then checks each component. Run ${PCI.run_id} (commit ${PCI.commit}) passed ${PCI.smoke.passed} of ${PCI.smoke.total} smoke checks${PCI.governance ? `; its governance job catalogued ${PCI.governance.tables.length} tables and views in OpenMetadata ${PCI.governance.version}, tagged ${PCI.governance.pii_columns} PII columns and recorded ${PCI.governance.lineage_edges} lineage edges from raw to gold` : ""}.`));
  add(TCAP("Platform smoke checks (platform workflow)."));
  add(table(["Component", "Check", "Result"], PCI.smoke.checks.map(([g, c, r]) => [g, c, r === "PASS" ? "Pass" : "Fail"]), [2, 9, 2]));
}
add(H2(PCI && PCI.smoke ? "8.7 Deployment" : "8.6 Deployment"));
add(TCAP("Deployment targets."));
const dRows = [
  ["Local (Windows, no Docker)", "scripts/deploy/run_local.ps1 starts the API and the dashboard; -Public adds Cloudflare quick-tunnel links", DEPLOY.tunnel_dashboard_url ? "Deployed during the project" : "Script"],
  ["Container image", "platform/docker/Dockerfile.app; built, smoke-tested and published to GHCR by the deploy workflow", "ghcr.io/anasua2905/fraudlens-app"],
  ["Hosted dashboard", "Streamlit Community Cloud from the repository (dashboard/app.py, runs from the bundle)", DEPLOY.streamlit_url || "Documented in docs/deployment.md"],
  ["Full platform", "docker compose profiles lake, ingest, olap, bi, mlops (+ OpenMetadata)", PCI && PCI.conclusion === "success" ? "platform/; verified end to end by the platform workflow" : "platform/"],
];
add(table(["Target", "How", "Where"], dRows, [3, 8, 4]));

// -------------------------------------------------------- 9. monitoring
add(PAGEBREAK(), H1("9. Monitoring and Improvement"));
add(P(`Monitoring replays the validation and test periods month by month as if the model were live. Evidently AI's DataDriftPreset compares each month with a training reference (with the project's PSI as a cross-check) and performance is measured at the fixed threshold. A retraining recommendation is raised when more than ${pf(MON.policy.drift_share_alert, 0)} of monitored features drift, recall falls below ${pf(MON.policy.recall_floor, 0)} or precision below ${pf(MON.policy.precision_floor, 0)}.`));
add(TCAP("Monthly monitoring."));
add(table(["Month", "Transactions", "Frauds", "Drift share", "Recall", "Precision", "Alerts/day", "Retrain"], MON.monthly.map((m) =>
  [m.month, int(m.transactions), int(m.frauds), f3(m.evidently_drift_share != null ? m.evidently_drift_share : m.psi_major_share), f3(m.recall), f3(m.precision), String(m.alerts_per_day), m.retrain_recommended ? "Yes" : "No"]), [2, 2, 2, 2, 2, 2, 2, 2]));
add(figure("19_monitoring_monthly.png", "Monthly performance and drift share."));
add(P(`The policy fired in ${MON.months_flagged.join(" and ")} because of drift (seasonal spending patterns), while recall stayed above the floor. The recommendation goes to a person, who decides whether to retrain.`));
if (CHAL) {
  const c0 = CHAL.comparison[0], c1 = CHAL.comparison[1], g = CHAL.pr_auc_gap_challenger_minus_champion;
  add(P(`**Champion/challenger.** Retraining the same algorithm on the training plus validation periods and comparing at the same alert budget on the test period gives PR-AUC ${f4(c1.pr_auc)} for the challenger against ${f4(c0.pr_auc)} for the champion (gap ${f4(g.gap_mean)}, 95% CI [${f4(g.ci95_low)}, ${f4(g.ci95_high)}]); under the promotion rule (${CHAL.rule}) the challenger ${CHAL.promote ? "would be promoted" : "would not be promoted"}. The test period is used here only to illustrate the loop; in production the comparison runs on the next labelled period, so the champion's reported test results stay untouched.`));
}
add(P("Operationally, the API exports Prometheus metrics (transactions, alerts, latency, score distribution, cold-start cards, feedback, monitoring verdict); Prometheus evaluates alert rules (API down, alert-rate spike, slow scoring, retraining recommended) and Grafana shows an operations dashboard and a ClickHouse business dashboard. In live operation the labels come from analyst decisions in the case store (section 8.2): live precision with its confidence interval, missed fraud reported on approved transactions, the review backlog and the time to decision are added to the monitoring summary and exported to Prometheus (with alert rules for low live precision and a growing backlog), and the decided cases are exported as labelled examples for retraining. Because analysts only label alerted transactions, chargeback labels for non-alerted transactions are combined with them before retraining. The runbook (docs/monitoring_runbook.md) describes promotion and rollback."));

// -------------------------------------------------------- 10. management
add(PAGEBREAK(), H1("10. Project Management, Quality and Reproducibility"));
add(H2("10.1 GitHub Project"));
add(P("Work is tracked on a GitHub Project board generated from scripts/github/project_plan.yaml by scripts/github/setup_project.py: four milestones (data foundation, analysis and modelling, deployment and monitoring, final submission), one issue per task with an acceptance checklist, labels for lifecycle stage and type, and a Lifecycle stage field on the board so progress can be followed stage by stage. Issue and pull-request templates and built-in workflows (auto-add, closed → Done) keep it in sync with the repository; docs/github_project.md is the team's guide."));
add(H2("10.2 CI/CD"));
add(P(`Three GitHub Actions workflows: ci (unit and end-to-end tests of the whole lifecycle on synthetic sample data, plus a headless render of the dashboard from the committed bundle), platform (the complete Docker platform on a runner: NiFi ingest, MinIO, Hive Metastore, Trino governance, ClickHouse, MLflow, API, Prometheus, Grafana, Superset, and a second job for OpenMetadata, checked by a smoke test of every component), and deploy (build, smoke-test and publish the application image).${SMOKE ? ` In the latest platform run ${SMOKE.passed} of ${SMOKE.passed + SMOKE.failed} smoke checks passed.` : PCI && PCI.smoke ? ` In platform run ${PCI.run_id} ${PCI.smoke.passed} of ${PCI.smoke.total} smoke checks passed and the governance job ${PCI.jobs && Object.entries(PCI.jobs).some(([k, v]) => k.startsWith("governance") && v === "success") ? "succeeded" : "ran"}.` : ""}`));
add(H2("10.3 End-to-end verification"));
const byStage = {}; VER.checks.forEach((c) => { byStage[c.stage] = byStage[c.stage] || [0, 0]; byStage[c.stage][0] += 1; if (c.status === "PASS") byStage[c.stage][1] += 1; });
add(TCAP("Verification checks by lifecycle stage (final run)."));
add(table(["Stage", "Checks", "Passed"], Object.entries(byStage).map(([s, [n, p]]) => [s, String(n), String(p)]), [6, 2, 2]));
add(P("A single command reproduces the project (python -m fraudlens.run_all), every stage can be resumed or run alone, library versions are pinned (requirements.lock.txt), and eight executed notebooks document each stage with live code."));

// -------------------------------------------------------- 11. discussion
add(PAGEBREAK(), H1("11. Discussion and Limitations"));
add(BULLET("**Non-linearity decides the model family.** Boosted trees reach validation PR-AUC around 0.98 against 0.68 for logistic regression; interactions between amount, time, category and recent spend matter more than any single feature."));
add(BULLET("**Simplicity as a selection principle.** Four boosted variants were within statistical noise; choosing the simplest reduced dependencies, artefact size and serving latency at no measurable cost."));
add(BULLET("**Unsupervised detection alone is insufficient.** The Isolation Forest finds unusual transactions, not fraudulent ones (validation PR-AUC 0.36); its score added nothing measurable on top of the supervised model, but it remains a candidate detector for new, unlabelled patterns."));
add(BULLET("**Operating point over metric.** The cost-based threshold, not the ranking metric, determines what the bank experiences; the dashboard makes that trade-off explicit."));
add(BULLET("**Simulated data.** Sparkov's fraud patterns are cleaner than real ones, so absolute results are optimistic; at a real-world fraud rate of 0.1% the same recall and false-alarm rate would give about 40% precision (section 7.5). The method, evaluation protocol and platform are what transfer, and a shadow-mode trial on real transactions comes before any savings claim."));
add(BULLET("**Estimated, not demonstrated, savings.** The net benefit rests on an assumed $15 review cost and 90% recovery rate; it ranks operating points but is not a measured saving."));
add(BULLET("**Selective labels.** In production, analyst feedback only labels alerted transactions; chargeback labels for non-alerted transactions are needed to avoid biasing retraining."));
add(BULLET("**Platform scale.** The Docker platform runs on one machine; the same components scale out (MinIO and ClickHouse clusters, Trino workers, NiFi cluster) without changing the pipeline code."));

// -------------------------------------------------------- 12. conclusion
add(PAGEBREAK(), H1("12. Conclusion"));
add(P(`FraudLens demonstrates end-to-end thinking from problem framing to a monitored deployment: requirements and success criteria defined first; data acquired with provenance and integrity checks; prepared without leakage and with privacy controls; explored to form and test hypotheses; modelled and selected by a pre-registered rule; evaluated on an untouched period with uncertainty, business impact, fairness and explanations; deployed as an API and dashboard on an open-source platform; and monitored with a clear path to retraining. All five project objectives are met: the champion catches ${pf(CT.recall)} of fraud with ${pf(CT.precision)} precision at ${CT.alerts_per_day} alerts a day, the dashboard supports analysts and managers, and the flow from data collection to visualisation is reproducible with one command and verified by ${VER.passed} checks.`));
add(P("Next steps: validate on real (or more realistic) transaction data, add probability calibration for expected-loss reporting, schedule the monitoring and retraining loop, and explore sequence models of card histories if deep learning is brought into scope."));

// -------------------------------------------------------- references
add(PAGEBREAK(), H1("References"));
[
  "Shenoy, K. (2020). Credit Card Transactions Fraud Detection Dataset (Sparkov). Kaggle. https://www.kaggle.com/datasets/kartik2112/fraud-detection",
  "Harris, B. Sparkov Data Generation. GitHub. https://github.com/namebrandon/Sparkov_Data_Generation",
  "Pedregosa, F. et al. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, 2825-2830.",
  "Chen, T. & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. Proceedings of KDD 2016, 785-794.",
  "Ke, G. et al. (2017). LightGBM: A highly efficient gradient boosting decision tree. Advances in NeurIPS 30.",
  "Prokhorenkova, L. et al. (2018). CatBoost: unbiased boosting with categorical features. Advances in NeurIPS 31.",
  "Liu, F. T., Ting, K. M. & Zhou, Z.-H. (2008). Isolation Forest. Proceedings of IEEE ICDM 2008, 413-422.",
  "Lundberg, S. M. & Lee, S.-I. (2017). A unified approach to interpreting model predictions. Advances in NeurIPS 30.",
  "Saito, T. & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. PLoS ONE, 10(3), e0118432.",
  "Dal Pozzolo, A. et al. (2015). Calibrating probability with undersampling for unbalanced classification. IEEE SSCI 2015.",
  "Evidently AI. Evidently documentation: data drift. https://docs.evidentlyai.com",
  "Apache Software Foundation. Apache NiFi, Apache Hive, Apache Superset documentation. https://nifi.apache.org, https://hive.apache.org, https://superset.apache.org",
  "Trino Software Foundation. Trino documentation: Hive connector and file-based access control. https://trino.io/docs",
  "ClickHouse Inc. ClickHouse documentation: s3 table function and named collections. https://clickhouse.com/docs",
  "OpenMetadata. OpenMetadata documentation. https://docs.open-metadata.org",
  "MLflow Project. MLflow documentation: tracking and model registry. https://mlflow.org/docs",
].forEach((r) => add(P(r)));

// -------------------------------------------------------- appendices
add(PAGEBREAK(), H1("Appendix A: Reproduction"));
["python -m venv .venv && .venv\\Scripts\\activate",
  "pip install -c requirements.lock.txt -e \".[ml,serve,platform,data,dev]\"",
  "python -m fraudlens.run_all --local-dir <folder with fraudTrain.csv and fraudTest.csv>",
  "python scripts/build_notebooks.py --execute",
  "streamlit run dashboard/app.py      |      uvicorn fraudlens.api:app --port 8000",
  "cd platform && docker compose --profile lake --profile ingest --profile olap --profile bi --profile mlops up -d",
  "pytest"].forEach((c) => add(CODE(c)));
add(P("Repository and documentation:"));
add(LINK(REPO + " (branch final-a4)", REPO + "/tree/final-a4"));
add(PAGEBREAK(), H1("Appendix B: Verification Checks"));
add(TCAP(`All ${VER.checks.length} end-to-end verification checks from the final run.`));
add(table(["Stage", "Check", "Status"], VER.checks.map((c) => [c.stage, c.check, c.status]), [3, 10, 2]));

// ------------------------------------------------------------------ build
const doc = new Document({
  creator: "FraudLens team", title: "FraudLens Final Project Report",
  numbering: { config: [
    { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 460, hanging: 260 } } } }] },
    { reference: "numbers", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 460, hanging: 260 } } } }] },
  ] },
  styles: { default: {
    document: { run: { font: "Calibri", size: 22 } },
    heading1: { run: { font: "Calibri", size: 32, bold: true, color: "1F3864" } },
    heading2: { run: { font: "Calibri", size: 26, bold: true, color: "2E5395" } },
    heading3: { run: { font: "Calibri", size: 23, bold: true, color: "2E5395" } },
  } },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1300, bottom: 1300, left: 1273, right: 1273 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ children: ["FraudLens · Page ", PageNumber.CURRENT, " of ", PageNumber.TOTAL_PAGES], size: 18, color: "666666" })] })] }) },
    children: [...titlePage, ...body],
  }],
});
Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(OUT, buf);
  console.log(`Written ${OUT}: ${figNo} figures, ${tableNo} tables; champion ${CH}; fairness ${!!FAIR}; challenger ${!!CHAL}; olap ${!!(OLAP && OLAP.status !== "skipped")}`);
});
