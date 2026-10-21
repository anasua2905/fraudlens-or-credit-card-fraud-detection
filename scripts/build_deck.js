/*
 * FraudLens final presentation (Assessment 4), built from the pipeline's result files.
 * Usage: node scripts/build_deck.js <reports_dir> <output.pptx>
 * Speaker notes on every slide carry the talk script.
 */
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");
const { applyTheme } = require(process.env.PPTX_SKILL_DIR
  ? path.join(process.env.PPTX_SKILL_DIR, "scripts", "apply_theme.js") : "/mnt/skills/public/pptx/scripts/apply_theme.js");

const DIR = process.argv[2] || "reports";
const OUT = process.argv[3] || "FraudLens_Final_Presentation.pptx";
const IMG = path.join(__dirname, "..", "docs", "images");
const J = (f) => JSON.parse(fs.readFileSync(path.join(DIR, f), "utf8"));
const JO = (f) => (fs.existsSync(path.join(DIR, f)) ? J(f) : null);
const MOD = J("model_results.json"), VER = J("verification_report.json"), EXP = J("explanations.json");
const MON = J("monitoring/summary.json"), RUN = J("pipeline_run.json"), EDA = J("eda_summary.json"), META = J("dashboard_meta.json");
const FAIR = JO("fairness.json"), CHAL = JO("monitoring/challenger.json"), PCI = JO("platform_ci.json");
const CH = MOD.champion, CT = MOD.results[CH].test, CI = MOD.champion_test_ci95;
const pf = (x, d = 1) => (100 * x).toFixed(d) + "%";
const usd = (n) => "$" + Math.round(n).toLocaleString("en-US");
const caughtShare = (CT.fraud_amount_caught / META.recovery_rate) / (CT.fraud_amount_caught / META.recovery_rate + CT.fraud_amount_missed);
const NAMES = { logistic: "Logistic", hist_gb: "HistGB", xgboost: "XGBoost", lightgbm: "LightGBM", isolation_forest: "Isolation F.", hybrid_gb: "Hybrid" };

const THEME = {
  name: "FraudLens",
  headFontFace: "Cambria", bodyFontFace: "Calibri",
  colors: { dk1: "1B2333", lt1: "FFFFFF", dk2: "14213D", lt2: "F2F4F8", accent1: "E4572E", accent2: "1F8A70",
            accent3: "2A78D6", accent4: "F2A541", accent5: "6C5CE7", accent6: "8A94A6", hlink: "2A78D6", folHlink: "6C5CE7" },
};
const HEX = THEME.colors;
const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";               // 10 x 5.625 in
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "FraudLens: final presentation";
pres.author = "FraudLens team";
const C = pres.SchemeColor;

// ---------------------------------------------------------------- layouts
pres.defineSlideMaster({ title: "DARK", background: { color: C.text2 },
  objects: [{ placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.7, w: 8.8, h: 1.8, fontSize: 38, bold: true, color: C.background1, fontFace: THEME.headFontFace, align: "left", valign: "bottom" }, text: "" } },
            { placeholder: { options: { name: "body", type: "body", x: 0.6, y: 2.75, w: 8.8, h: 2.0, fontSize: 17, color: "CADCFC", align: "left", valign: "top" }, text: "" } }] });
pres.defineSlideMaster({ title: "CONTENT", background: { color: C.background1 },
  margin: [0.4, 0.5, 0.5, 0.5],
  slideNumber: { x: 9.2, y: 5.25, w: 0.5, h: 0.25, fontSize: 9, color: C.accent6 },
  objects: [{ placeholder: { options: { name: "title", type: "title", x: 0.5, y: 0.28, w: 9.0, h: 0.7, fontSize: 28, bold: true, color: C.text2, fontFace: THEME.headFontFace, valign: "middle", align: "left" }, text: "" } },
            { text: { text: "FraudLens · PRT661 Assessment 4", options: { x: 0.5, y: 5.25, w: 4, h: 0.25, fontSize: 9, color: C.accent6 } } }] });

let n = 0;
const card = (s, x, y, w, h, fill = C.background2) => s.addShape(pres.shapes.ROUNDED_RECTANGLE,
  { x, y, w, h, fill: { color: fill }, line: { color: fill }, rectRadius: 0.08, objectName: `card-${++n}` });
const text = (s, t, o) => s.addText(t, { isTextBox: true, margin: 0, fontSize: 14, color: C.text1, valign: "top", ...o });
const stat = (s, x, y, w, value, label, color = C.accent1) => {
  card(s, x, y, w, 1.25);
  text(s, value, { x: x + 0.2, y: y + 0.12, w: w - 0.4, h: 0.6, fontSize: 30, bold: true, color, fontFace: THEME.headFontFace });
  text(s, label, { x: x + 0.2, y: y + 0.74, w: w - 0.4, h: 0.45, fontSize: 12, color: C.text1 });
};
const bullets = (s, items, o) => s.addText(items.map((t, i) => ({ text: t, options: { bullet: { indent: 14 }, breakLine: i < items.length - 1 } })),
  { isTextBox: true, margin: 0, fontSize: 14, color: C.text1, paraSpaceAfter: 6, valign: "top", ...o });
const chartText = { catAxisLabelColor: "4B5563", valAxisLabelColor: "4B5563", catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt",
  catAxisLabelFontSize: 11, valAxisLabelFontSize: 10, dataLabelFontFace: "+mn-lt", dataLabelFontSize: 10, legendFontFace: "+mn-lt", legendFontSize: 11 };
const img = (s, file, x, y, w, h) => s.addImage({ path: path.join(IMG, file), x, y, w, h, objectName: file });
function fitImage(file, maxW, maxH) {
  const b = fs.readFileSync(path.join(IMG, file)); const W = b.readUInt32BE(16), H = b.readUInt32BE(20);
  const r = Math.min(maxW / W, maxH / H); return [W * r, H * r];
}

// ================================================================ 1 title
pres.addSection({ title: "Introduction" });
let s = pres.addSlide({ masterName: "DARK", sectionTitle: "Introduction" });
s.addText("FraudLens", { placeholder: "title" });
s.addText([{ text: "An anomaly detection framework for fraudulent credit card transactions", options: { breakLine: true } },
  { text: "The complete data science lifecycle on an open-source data platform", options: { fontSize: 14, color: "8FA6CF" } }], { placeholder: "body" });
s.addNotes("Welcome. FraudLens is our end-to-end anomaly detection framework for card fraud. In the next ten minutes we walk through all eight stages of the data science lifecycle, show the open-source platform it runs on, demonstrate the deployed system, and finish with monitoring and how the project was managed.");

// ================================================================ 2 problem + result
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Introduction" });
s.addText("Catch fraud without flooding the review queue", { placeholder: "title" });
text(s, "Every alert costs an analyst $15; every missed fraud costs its amount. Four success criteria were set before modelling; all four are met on the untouched test period (Jun-Dec 2020).",
  { x: 0.5, y: 1.1, w: 9, h: 0.6, fontSize: 15 });
const crit = VER.success_criteria;
const lab = { min_test_recall: "recall · target ≥ 90%", min_test_precision: "precision · target ≥ 50%", max_alerts_per_day: "alerts/day · target ≤ 20", min_pr_auc_lift: "PR-AUC lift · target ≥ 50×" };
const val = { min_test_recall: pf(CT.recall), min_test_precision: pf(CT.precision), max_alerts_per_day: String(CT.alerts_per_day), min_pr_auc_lift: Math.round(CT.pr_auc_lift) + "×" };
crit.forEach((c, i) => stat(s, 0.5 + i * 2.3, 1.95, 2.1, val[c.criterion], lab[c.criterion], i % 2 ? C.accent2 : C.accent1));
text(s, [{ text: `≈ ${usd(CT.net_benefit)} estimated net benefit`, options: { bold: true, color: C.text2 } },
  { text: ` on the test period (${pf(caughtShare)} of fraud value caught), assuming $${META.investigation_cost} per review and ${pf(META.recovery_rate, 0)} recovery: an estimate on simulated data, not demonstrated savings. All ${VER.passed} verification checks pass.` }],
  { x: 0.5, y: 3.55, w: 9, h: 0.6, fontSize: 15 });
text(s, "Data: Sparkov simulated card transactions (Kaggle, CC0), 1.85 M transactions, 2019-2020, 0.52% fraud.", { x: 0.5, y: 4.5, w: 9, h: 0.4, fontSize: 12, color: C.accent6 });
s.addNotes(`The business problem is a trade-off: catch fraud, but keep the analysts' queue small and trustworthy. We wrote four success criteria into the configuration before any model was trained. On the six months of test data the champion catches ${pf(CT.recall)} of fraud with ${pf(CT.precision)} precision at ${CT.alerts_per_day} alerts a day. In money terms that is an estimated ${usd(CT.net_benefit)} net benefit, but note the word estimated: it assumes a review costs fifteen dollars and that ninety percent of caught fraud value is recovered, on simulated data. It ranks operating points; real savings would have to be measured in a trial. Accuracy is not on the slide on purpose: never alerting would be 99.6% accurate.`);

// ================================================================ 3 lifecycle
pres.addSection({ title: "Lifecycle" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Lifecycle" });
s.addText("Eight lifecycle stages, end to end", { placeholder: "title" });
const stages = [["1", "Problem", "criteria, cost model, hypotheses"], ["2", "Acquisition", "Kaggle API, NiFi, checksums"], ["3", "Preparation", "contract, privacy, leakage-safe"], ["4", "EDA", "Seaborn, Plotly, hypotheses"],
  ["5", "Modelling", "6 models, time-series CV"], ["6", "Evaluation", "CIs, cost, SHAP, fairness"], ["7", "Deployment", "FastAPI, Streamlit, Docker"], ["8", "Monitoring", "Evidently, Prometheus, retrain"]];
stages.forEach(([k, t, d], i) => {
  const x = 0.5 + (i % 4) * 2.3, y = 1.2 + Math.floor(i / 4) * 1.75;
  card(s, x, y, 2.1, 1.5);
  s.addShape(pres.shapes.OVAL, { x: x + 0.18, y: y + 0.18, w: 0.5, h: 0.5, fill: { color: i < 4 ? C.accent3 : C.accent2 }, line: { color: i < 4 ? C.accent3 : C.accent2 }, objectName: `step-${k}` });
  text(s, k, { x: x + 0.18, y: y + 0.18, w: 0.5, h: 0.5, fontSize: 16, bold: true, color: C.background1, align: "center", valign: "middle" });
  text(s, t, { x: x + 0.8, y: y + 0.22, w: 1.25, h: 0.45, fontSize: 15, bold: true, color: C.text2, valign: "middle" });
  text(s, d, { x: x + 0.18, y: y + 0.8, w: 1.8, h: 0.6, fontSize: 12 });
});
text(s, "python -m fraudlens.run_all reproduces every stage; 8 executed notebooks document them.", { x: 0.5, y: 4.75, w: 9, h: 0.35, fontSize: 12, color: C.accent6 });
s.addNotes("The project follows the eight-stage lifecycle. Each stage has a module in the package and an executed notebook. One command reproduces everything from the raw files, and a verification stage checks that the stages agree with each other.");

// ================================================================ 4 architecture
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Lifecycle" });
s.addText("An open-source version of the cloud data pipeline", { placeholder: "title" });
let [w, h] = fitImage("architecture.png", 9.0, 3.9);
img(s, "architecture.png", 0.5 + (9 - w) / 2, 1.1, w, h);
s.addNotes("Every role in a modern cloud data pipeline has an open-source component here. MinIO is the S3-compatible lake with raw, silver and gold buckets. NiFi ingests the files; a Python crawler infers schemas and registers tables in the Hive Metastore. Governance is enforced by Trino access rules and least-privilege MinIO accounts and documented in OpenMetadata. Trino and ClickHouse serve analytics; Superset, Grafana and our Streamlit dashboard visualise. MLflow, FastAPI, Evidently and Prometheus cover machine-learning operations.");

// ================================================================ 5 data
pres.addSection({ title: "Data" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Data" });
s.addText("Provenance in, leakage out", { placeholder: "title" });
const sp = RUN.split.splits;
card(s, 0.5, 1.15, 4.3, 3.7);
text(s, "Acquisition and preparation", { x: 0.75, y: 1.3, w: 3.8, h: 0.4, fontSize: 16, bold: true, color: C.text2 });
bullets(s, ["Kaggle API or local copy; SHA-256 manifest re-verified every run", "NiFi lands files in a versioned raw bucket with a write-only account",
  "12-check data contract; no duplicates or invalid rows found", "Names and streets dropped, card numbers salted-hashed",
  "Features use only each card's earlier transactions (proved by brute-force check)"], { x: 0.75, y: 1.8, w: 3.85, h: 2.9, fontSize: 13 });
text(s, "Chronological split", { x: 5.2, y: 1.2, w: 4.3, h: 0.4, fontSize: 16, bold: true, color: C.text2 });
s.addChart(pres.charts.BAR, [{ name: "Transactions", labels: ["Train", "Validation", "Test"], values: [sp.train.rows, sp.val.rows, sp.test.rows] }],
  { x: 5.1, y: 1.6, w: 4.4, h: 2.4, barDir: "bar", chartColors: [HEX.accent3], showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "#,##0",
    valAxisHidden: true, valGridLine: { style: "none" }, catGridLine: { style: "none" }, showLegend: false, ...chartText });
text(s, `Train ${sp.train.start.slice(0, 10)} → ${sp.train.end.slice(0, 10)} · validation 6 weeks · test = Kaggle's later file. Fraud rate ${sp.train.fraud_rate_pct.toFixed(2)}% → ${sp.test.fraud_rate_pct.toFixed(2)}%.`,
  { x: 5.2, y: 4.1, w: 4.3, h: 0.7, fontSize: 12, color: C.text1 });
s.addNotes("Acquisition records checksums, sizes and a column profile, and on the platform NiFi writes the files into an immutable raw bucket. Preparation validates a data contract, removes personal identifiers and builds behavioural features from each card's history, using only earlier transactions; we proved that with a brute-force recomputation. The split is chronological, because a fraud model is always deployed forward in time.");

// ================================================================ 6 EDA
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Data" });
s.addText("What the data says about fraud", { placeholder: "title" });
const night = EDA.behaviour.night["night (22-04)"], tenX = EDA.behaviour.amount_vs_card_mean["10x+"];
stat(s, 0.5, 1.2, 2.1, "$" + Math.round(EDA.amount_by_class.fraud.p50), `median fraud amount vs $${Math.round(EDA.amount_by_class.legit.p50)} legitimate`);
stat(s, 2.8, 1.2, 2.1, pf(night.share_of_frauds_pct / 100, 0), "of fraud happens 22:00-03:59", C.accent3);
stat(s, 5.1, 1.2, 2.1, pf(tenX.fraud_rate_pct / 100, 0), "fraud rate when amount > 10× the card's usual", C.accent2);
stat(s, 7.4, 1.2, 2.1, String(EDA.card_level.frauds_per_fraud_card_median), `frauds per compromised card, within ~${Math.round(EDA.card_level.fraud_span_hours_median)} h`, C.accent4);
[w, h] = [4.3, 4.3 * 950 / 2300];
s.addImage({ path: path.join(DIR, "figures", "03_fraud_by_category.png"), x: 0.5, y: 2.65, ...(() => { const b = fs.readFileSync(path.join(DIR, "figures", "03_fraud_by_category.png")); const W = b.readUInt32BE(16), H = b.readUInt32BE(20); const r = Math.min(4.3 / W, 2.45 / H); return { w: W * r, h: H * r }; })(), objectName: "eda-category" });
text(s, "Hypotheses confirmed", { x: 5.1, y: 2.75, w: 4.4, h: 0.4, fontSize: 16, bold: true, color: C.text2 });
bullets(s, ["H1 large amounts relative to the card: strongly", "H2 night-time: strongly", "H3 bursts on a compromised card: yes; 24 h spend is the top single feature",
  "H4 online and grocery-POS categories: 2-3× base rate", "H5 demographics: weak, kept but checked for fairness"], { x: 5.1, y: 3.2, w: 4.4, h: 1.9, fontSize: 13 });
s.addNotes("EDA on the training period confirmed the hypotheses from stage one. Fraud is large compared with the card's own history, concentrated at night, arrives in bursts on compromised cards, and is two to three times more common in online shopping and grocery point-of-sale. Demographics carry little signal, so we kept them but check fairness later.");

// ================================================================ 7 modelling
pres.addSection({ title: "Modelling" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Modelling" });
s.addText("Six candidates, one pre-registered selection rule", { placeholder: "title" });
const comp = [...MOD.comparison].sort((a, b) => b.val_pr_auc - a.val_pr_auc);
s.addChart(pres.charts.BAR, [{ name: "Validation PR-AUC", labels: comp.map((r) => NAMES[r.model]), values: comp.map((r) => +r.val_pr_auc.toFixed(3)) }],
  { x: 0.4, y: 1.1, w: 5.2, h: 3.7, barDir: "bar", chartColors: comp.map((r) => (r.model === CH ? HEX.accent1 : HEX.accent6)), showValue: true, dataLabelPosition: "outEnd",
    dataLabelFormatCode: "0.000", valAxisMinVal: 0, valAxisMaxVal: 1.1, valAxisHidden: true, valGridLine: { style: "none" }, catGridLine: { style: "none" },
    showLegend: false, showTitle: true, title: "Validation PR-AUC (fraud rate 0.006)", titleFontSize: 12, titleColor: "4B5563", titleFontFace: "+mn-lt", ...chartText });
card(s, 5.9, 1.15, 3.6, 3.65);
text(s, "Simplest non-inferior model wins", { x: 6.1, y: 1.3, w: 3.25, h: 0.6, fontSize: 15, bold: true, color: C.text2 });
bullets(s, ["Time-series CV tuning inside train; class weights, no SMOTE", "Four boosted models within noise of each other",
  `Rule: pick the simplest whose PR-AUC gap to the best has a 95% CI upper bound < ${MOD.selection.margin}`,
  `${NAMES[CH]} (scikit-learn): gap CI [${MOD.selection.comparisons[CH].ci95_low.toFixed(4)}, ${MOD.selection.comparisons[CH].ci95_high.toFixed(4)}]`,
  "Tracked and registered in MLflow"], { x: 6.1, y: 1.95, w: 3.25, h: 2.8, fontSize: 12 });
s.addNotes(`We compared logistic regression, three gradient-boosting libraries, an unsupervised Isolation Forest and a hybrid. The boosted models are all around 0.98 PR-AUC on validation, logistic regression 0.68, and the anomaly detector alone only 0.36. Rather than pick the top number, we registered a rule in advance: choose the simplest model that is statistically non-inferior to the best. That is ${NAMES[CH]}, which needs only scikit-learn and is the smallest and fastest to serve.`);

// ================================================================ 8 evaluation
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Modelling" });
s.addText("On unseen months: high recall, tight intervals", { placeholder: "title" });
stat(s, 0.5, 1.15, 2.1, pf(CT.recall), `recall (95% CI ${pf(CI.recall.low)}-${pf(CI.recall.high)})`, C.accent2);
stat(s, 0.5, 2.55, 2.1, pf(CT.precision), `precision (95% CI ${pf(CI.precision.low)}-${pf(CI.precision.high)})`, C.accent1);
stat(s, 0.5, 3.95, 2.1, String(CT.alerts_per_day), `alerts per day at threshold ${META.threshold.toFixed(3)}`, C.accent3);
const mt = MOD.champion_monthly_test;
s.addChart(pres.charts.LINE, [{ name: "Recall", labels: mt.map((m) => m.period), values: mt.map((m) => m.recall) },
  { name: "Precision", labels: mt.map((m) => m.period), values: mt.map((m) => m.precision) }],
  { x: 2.9, y: 1.1, w: 6.6, h: 3.3, chartColors: [HEX.accent2, HEX.accent1], lineSize: 2.5, lineDataSymbolSize: 7, valAxisMinVal: 0.5, valAxisMaxVal: 1,
    valAxisLabelFormatCode: "0%", valGridLine: { color: "E5E7EB", size: 0.5 }, catGridLine: { style: "none" }, showLegend: true, legendPos: "b",
    showTitle: true, title: "Monthly performance on the test period", titleFontSize: 12, titleColor: "4B5563", titleFontFace: "+mn-lt", ...chartText });
text(s, `Threshold chosen on validation by maximising net benefit, applied once to test. ${pf(caughtShare)} of fraud value caught; missed fraud is mostly low-value.`,
  { x: 2.9, y: 4.45, w: 6.6, h: 0.6, fontSize: 12 });
s.addNotes(`The threshold was chosen on validation to maximise net benefit and then applied once to the test period. Recall is ${pf(CT.recall)} and precision ${pf(CT.precision)}, with bootstrap intervals of about plus or minus one percentage point, and the monthly lines are stable from June to December. The fraud it misses is mostly small transactions, where an alert would cost more than it recovers.`);

// ================================================================ 9 explain + fairness
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Modelling" });
s.addText("Every alert explains itself", { placeholder: "title" });
const sh = Object.entries(EXP.mean_abs_shap).slice(0, 8).reverse();
s.addChart(pres.charts.BAR, [{ name: "mean |SHAP|", labels: sh.map(([f]) => f), values: sh.map(([, v]) => +v.toFixed(2)) }],
  { x: 0.4, y: 1.1, w: 5.0, h: 3.8, barDir: "bar", chartColors: [HEX.accent3], showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.00",
    valAxisHidden: true, valGridLine: { style: "none" }, catGridLine: { style: "none" }, showLegend: false, showTitle: true,
    title: "Average impact on the fraud log-odds (SHAP)", titleFontSize: 12, titleColor: "4B5563", titleFontFace: "+mn-lt", ...chartText });
card(s, 5.7, 1.15, 3.8, 1.75);
text(s, "Reasons an analyst can act on", { x: 5.9, y: 1.28, w: 3.4, h: 0.4, fontSize: 15, bold: true, color: C.text2 });
text(s, "Top-3 SHAP contributions on every alert, plus plain rules: \"Amount is 7× this card's usual spend; 5 transactions in 24 h; 3 min since the last one\".",
  { x: 5.9, y: 1.7, w: 3.4, h: 1.15, fontSize: 12 });
card(s, 5.7, 3.1, 3.8, 1.8);
text(s, "Fairness, reported honestly", { x: 5.9, y: 3.23, w: 3.4, h: 0.4, fontSize: 15, bold: true, color: C.text2 });
const fg = FAIR ? Object.fromEntries(FAIR.filter((r) => r.group === "gender").map((r) => [r.value, r])) : null;
text(s, fg ? `Recall ${fg.F.recall.toFixed(3)} for women vs ${fg.M.recall.toFixed(3)} for men; false alarms ${fg.F.false_alarm_rate_per_10k} vs ${fg.M.false_alarm_rate_per_10k} per 10k. Next step on real data: retrain without gender and compare.` : "Group metrics by gender and age band in notebook 06.",
  { x: 5.9, y: 3.65, w: 3.4, h: 1.2, fontSize: 12 });
s.addNotes("Explanations come from SHAP. Amount, category, the last 24 hours of spend, night-time and velocity drive the score, which matches the EDA. Every alert carries its top three reasons, so the analyst sees why. We also report performance by group: recall is high for everyone, but slightly lower for women than for men, so before any real deployment we would retrain without gender and compare.");

// ================================================================ 10 deployment
pres.addSection({ title: "Deployment" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Deployment" });
s.addText("Deployed: scoring API and analyst dashboard", { placeholder: "title" });
[w, h] = fitImage("dashboard_live_scoring.png", 6.1, 3.85);
img(s, "dashboard_live_scoring.png", 0.5, 1.15, w, h);
bullets(s, ["FastAPI /score: online feature store computes card history in real time; SHAP reasons and a counterfactual",
  "Review queue: /alerts and /alerts/{id}/decision; the dashboard's Case review tab",
  "Access control: role-based API keys (scorer, analyst, admin); dashboard sign-in (analyst, viewer); audit log",
  "One 40 MB bundle; container image ghcr.io/…/fraudlens-app; public links with access control on"], { x: 0.6 + w + 0.2, y: 1.2, w: 9.5 - (0.6 + w + 0.2), h: 3.6, fontSize: 12 });
s.addNotes("Deployment has two faces. The FastAPI service scores a raw transaction in real time: the online feature store keeps every card's running history, the model scores, and SHAP gives the reasons. The Streamlit dashboard is for analysts and managers. Here you see live scoring: a simulated card-testing burst of six rising amounts three minutes apart is flagged at 0.94 with readable reasons. Both run from one 40-megabyte bundle and are packaged as a container image. [Switch to the live demo here.]");

// ================================================================ 10b live path
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Deployment" });
s.addText("A new transaction, from features to an analyst decision", { placeholder: "title" });
const steps = [["1 Received", "POST /score or the dashboard's Live transaction tab"],
  ["2 Features", "online feature store: card history, velocity 1 h / 24 h / 7 d"],
  ["3 Inference", "score vs threshold, SHAP reasons, counterfactual amount"],
  ["4 Routed", "alert → case opened in the review queue; else approved"],
  ["5 Decision", "analyst: fraud / legitimate + note; audit log"]];
steps.forEach(([t, d], i) => {
  const x = 0.4 + i * 1.86;
  card(s, x, 1.15, 1.72, 1.55);
  text(s, t, { x: x + 0.12, y: 1.25, w: 1.5, h: 0.35, fontSize: 14, bold: true, color: i === 4 ? C.accent1 : C.text2 });
  text(s, d, { x: x + 0.12, y: 1.62, w: 1.5, h: 1.0, fontSize: 11 });
});
card(s, 0.4, 2.95, 9.2, 1.95);
text(s, "How decisions inform monitoring", { x: 0.6, y: 3.05, w: 8.8, h: 0.35, fontSize: 15, bold: true, color: C.accent3 });
bullets(s, ["Live precision (confirmed ÷ reviewed alerts) with a 95% interval vs the 40% floor → threshold/model review, Prometheus alert",
  "Fraud reported on approved transactions = live misses; backlog and time to decision = workload",
  "Every decided case keeps its exact model inputs → labelled examples for the next champion/challenger"],
  { x: 0.6, y: 3.45, w: 8.8, h: 1.4, fontSize: 12.5 });
s.addNotes("This answers the main question from the last review: what happens to a transaction we have never seen. It arrives at the scoring API, or here in the dashboard, which runs the same code. The online feature store computes its card-history features from the card's earlier transactions. The champion scores it, SHAP explains the score, and for an alert we also show the amount at which it would have been approved. An alert opens a case in the review queue; the analyst sees exactly what the model saw and records fraud or legitimate, with their name in the audit log. Those decisions are labels: live precision is tracked against the policy floor and alerts in Prometheus if it drops, disputes on approved transactions count as misses, and every decided case becomes a training example for the next model. Access is controlled throughout: API keys by role and a dashboard sign-in where only analysts can decide.");

// ================================================================ 11 monitoring
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Deployment" });
s.addText("Monitoring closes the loop", { placeholder: "title" });
const mm = MON.monthly;
s.addChart(pres.charts.BAR, [{ name: "Drift share", labels: mm.map((m) => m.month), values: mm.map((m) => +(m.evidently_drift_share ?? m.psi_major_share).toFixed(2)) }],
  { x: 0.4, y: 1.1, w: 5.4, h: 3.2, chartColors: mm.map((m) => (m.retrain_recommended ? HEX.accent1 : HEX.accent3)), showValue: true, dataLabelPosition: "outEnd",
    dataLabelFormatCode: "0%", valAxisMinVal: 0, valAxisMaxVal: 0.6, valAxisHidden: true, valGridLine: { style: "none" }, catGridLine: { style: "none" },
    showLegend: false, showTitle: true, title: `Share of monitored features drifting (Evidently); retrain trigger > ${pf(MON.policy.drift_share_alert, 0)}`,
    titleFontSize: 12, titleColor: "4B5563", titleFontFace: "+mn-lt", ...chartText });
text(s, `Flagged ${MON.months_flagged.join(" and ")} (seasonal spending); recall stayed ≥ ${pf(Math.min(...mm.map((m) => m.recall)))}.`, { x: 0.5, y: 4.4, w: 5.3, h: 0.5, fontSize: 12 });
card(s, 6.1, 1.15, 3.4, 3.7);
text(s, "Improvement loop", { x: 6.3, y: 1.3, w: 3.0, h: 0.4, fontSize: 15, bold: true, color: C.text2 });
const chal = CHAL ? CHAL.comparison : null;
bullets(s, ["Evidently drift + PSI and performance each month", "Prometheus alert rules; Grafana operations and business dashboards",
  "Analyst feedback via /feedback", chal ? `Challenger retrained on train + val: PR-AUC ${chal[1].pr_auc.toFixed(4)} vs ${chal[0].pr_auc.toFixed(4)} → promote via MLflow alias` : "Champion/challenger retraining test"],
  { x: 6.3, y: 1.75, w: 3.05, h: 3.0, fontSize: 12 });
s.addNotes("Monitoring replays the scored months as if the model were live. Evidently measures drift against the training data; performance is measured at the fixed threshold. The policy flagged August and December, driven by seasonal spending, while recall stayed above the floor, so a person reviews the recommendation. A challenger retrained on more recent data scores slightly higher and would be promoted through the MLflow registry alias. In operation, Prometheus and Grafana watch the API and analyst feedback provides new labels.");

// ================================================================ 12 management
pres.addSection({ title: "Close" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Close" });
s.addText("Managed and verified like a product", { placeholder: "title" });
const cols = [["GitHub Project", ["Board generated from project_plan.yaml", "4 milestones, one issue per task with acceptance checklist", "Lifecycle-stage field tracks progress stage by stage"]],
  ["CI/CD", ["ci: full lifecycle on sample data + dashboard render", PCI && PCI.smoke ? `platform: Docker stack end to end, ${PCI.smoke.passed}/${PCI.smoke.total} smoke checks pass` : "platform: Docker stack end to end, smoke tests", "deploy: build, test, publish the image"]],
  ["Quality", [`${VER.passed}/${VER.passed + VER.failed} verification checks pass`, "Train/serve parity test (caught a real bug)", "Pinned versions, one-command rebuild"]]];
cols.forEach(([t, items], i) => {
  const x = 0.5 + i * 3.05; card(s, x, 1.2, 2.85, 3.5);
  text(s, t, { x: x + 0.2, y: 1.35, w: 2.45, h: 0.4, fontSize: 16, bold: true, color: i === 0 ? C.accent3 : i === 1 ? C.accent2 : C.accent1 });
  bullets(s, items, { x: x + 0.2, y: 1.85, w: 2.5, h: 2.7, fontSize: 13 });
});
s.addNotes(`The project is run like a product. The GitHub Project board is generated from a plan file, with milestones and an issue per task, and a lifecycle-stage field so progress is visible stage by stage. Three workflows test the lifecycle, the whole Docker platform, and the deployable image.${PCI && PCI.smoke ? ` On the runner every platform service starts, the sample lifecycle flows from NiFi through the lake, Trino and ClickHouse, and ${PCI.smoke.passed} of ${PCI.smoke.total} smoke checks pass${PCI.governance ? `; OpenMetadata catalogues ${PCI.governance.tables.length} tables with ${PCI.governance.pii_columns} PII-tagged columns and lineage` : ""}.` : ""} Fifty-six verification checks pass, including a train-to-serve parity test that caught a real precision bug during the final run.`);

// ================================================================ 13 conclusion
s = pres.addSlide({ masterName: "DARK", sectionTitle: "Close" });
s.addText("From data collection to a monitored deployment", { placeholder: "title" });
s.addText([{ text: `${pf(CT.recall)} of fraud caught at ${CT.alerts_per_day} alerts/day, ${pf(CT.precision)} precision, every alert explained`, options: { breakLine: true, bullet: true } },
  { text: "Open-source platform: MinIO, NiFi, Hive Metastore, Trino, ClickHouse, Superset, Grafana, MLflow", options: { breakLine: true, bullet: true } },
  { text: "Next: real transaction data, calibrated probabilities, scheduled retraining", options: { bullet: true } }], { placeholder: "body" });
s.addNotes("To conclude: FraudLens meets all five objectives, from data collection to a monitored deployment, on open-source tools. The limitation to keep in mind is that the data are simulated, so the numbers validate the method rather than predict production accuracy. Next steps are real data, calibrated probabilities and a scheduled retraining loop. Thank you; we are happy to take questions.");

(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("Written", OUT);
})();
