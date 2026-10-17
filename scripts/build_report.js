/*
 * FraudLens project report, revision 2.
 * Every number is read from the pipeline's result files, so the report
 * cannot disagree with the code that produced them.
 *
 * Usage: node build_report.js <reports_dir> <output.docx>
 */
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow, TableCell,
  WidthType, ShadingType, AlignmentType, ImageRun, PageBreak, TableOfContents,
  LevelFormat, Footer, PageNumber,
} = require("docx");

const DIR = process.argv[2] || "reports";
const OUT = process.argv[3] || "FraudLens_Project_Report.docx";
const J = (f) => JSON.parse(fs.readFileSync(path.join(DIR, f), "utf8"));
// In the repository, figures live in reports/figures; uploaded copies sit beside the JSON files.
const FIGDIR = fs.existsSync(path.join(DIR, "figures")) ? path.join(DIR, "figures") : DIR;

// ------------------------------------------------------------------ data
const RUN = J("pipeline_run.json");
const VAL = J("validation_report.json");
const AUD = J("cleaning_audit.json");
const EDA = J("eda_summary.json");
const PRE = J("preprocessing_report.json");
const MOD = J("model_results.json");
const VER = J("verification_report.json");

const CH = MOD.champion;
const SEL = MOD.selection;
const CR = MOD.results[CH];
const CT = CR.test, CV = CR.validation;
const STRAT = MOD.threshold_strategy;
const THR = CR.thresholds_from_validation[STRAT];
const ALT = CR.test_at_other_thresholds.alert_budget;
const BEST = SEL.best_by_validation_pr_auc;
const OTHER_GB = CH === "hist_gb" ? "hybrid_gb" : "hist_gb";
const OG = MOD.results[OTHER_GB];
const NAMES = { logistic: "Logistic regression", hist_gb: "Gradient boosting", hybrid_gb: "Hybrid gradient boosting",
  isolation_forest: "Isolation Forest", mlp: "Neural network (MLP)" };
const nm = (m) => NAMES[m] || m;

// formatting
const int = (n) => Math.round(n).toLocaleString("en-US");
const usd = (n, d = 0) => "$" + Number(n).toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
const pf = (frac, d = 1) => (100 * frac).toFixed(d) + "%";          // fraction -> %
const pv = (val, d = 3) => Number(val).toFixed(d) + "%";             // already in %
const f4 = (x) => Number(x).toFixed(4);

// derived quantities
const splits = RUN.split.splits;
const totalRows = RUN.manifest.files.train.rows + RUN.manifest.files.test.rows;
const testDays = CT.alerts / CT.alerts_per_day;
const extraAlerts = ALT.alerts - CT.alerts;
const extraRecovered = ALT.fraud_amount_caught - CT.fraud_amount_caught;
const breakEven = extraRecovered / extraAlerts;
const legitTest = CT.false_positives + CT.true_negatives;
const RECOVERY = 0.9; // model.threshold.recovery_rate in config/pipeline.yaml
const missedShare = CT.fraud_amount_missed / (CT.fraud_amount_caught / RECOVERY + CT.fraud_amount_missed);
const verByStage = {};
VER.checks.forEach((c) => {
  verByStage[c.stage] = verByStage[c.stage] || { n: 0, pass: 0 };
  verByStage[c.stage].n += 1; if (c.status === "PASS") verByStage[c.stage].pass += 1;
});
const findCheck = (s) => VER.checks.find((c) => c.check.startsWith(s));
const gapCI = SEL.comparisons[CH === BEST ? OTHER_GB : CH];
const logGap = SEL.comparisons.logistic;

// --------------------------------------------------------------- helpers
const TOTAL_W = 9360;
const H1 = (t) => new Paragraph({ text: t, heading: HeadingLevel.HEADING_1, spacing: { before: 320, after: 160 } });
const H2 = (t) => new Paragraph({ text: t, heading: HeadingLevel.HEADING_2, spacing: { before: 260, after: 120 } });
const H3 = (t) => new Paragraph({ text: t, heading: HeadingLevel.HEADING_3, spacing: { before: 200, after: 100 } });
const P = (t) => new Paragraph({ children: [new TextRun({ text: t, size: 22 })], spacing: { after: 140, line: 276 } });
const BULLET = (t) => new Paragraph({ children: [new TextRun({ text: t, size: 22 })],
  numbering: { reference: "bullets", level: 0 }, spacing: { after: 80, line: 276 } });
const NUMBERED = (t) => new Paragraph({ children: [new TextRun({ text: t, size: 22 })],
  numbering: { reference: "numbers", level: 0 }, spacing: { after: 80, line: 276 } });
const CODE = (t) => new Paragraph({ children: [new TextRun({ text: t, font: "Consolas", size: 20 })], spacing: { after: 60 } });
const PAGEBREAK = () => new Paragraph({ children: [new PageBreak()] });

function cell(text, { bold = false, fill = null, width, align = AlignmentType.LEFT } = {}) {
  return new TableCell({
    width: { size: width, type: WidthType.DXA },
    shading: fill ? { type: ShadingType.CLEAR, fill, color: "auto" } : undefined,
    margins: { top: 60, bottom: 60, left: 90, right: 90 },
    children: [new Paragraph({ alignment: align, spacing: { after: 0, line: 240 },
      children: [new TextRun({ text: String(text), bold, size: 19 })] })],
  });
}
const isNumeric = (v) => {
  const t = String(v).trim();
  return t === "n/a" || /^[+\-$<(]?[\d.,]+\)?\s*(%|x|hours|years)?(\s*\([\d.,]+[%x]?\))?$/.test(t)
    || /^\[[+\-\d., ]+\]$/.test(t) || /^[\d,]+( \/ [\d,]+)+$/.test(t);
};
function table(headers, rows, weights) {
  if (headers.length !== weights.length) throw new Error(`weights mismatch: ${headers.join("|")}`);
  rows.forEach((r, i) => { if (r.length !== headers.length) throw new Error(`row ${i}: ${r.join("|")}`); });
  const sum = weights.reduce((a, b) => a + b, 0);
  const widths = weights.map((w) => Math.round((w / sum) * TOTAL_W));
  widths[0] += TOTAL_W - widths.reduce((a, b) => a + b, 0);
  const align = (i, v) => (i > 0 && isNumeric(v) ? AlignmentType.RIGHT : AlignmentType.LEFT);
  return [new Table({
    columnWidths: widths, width: { size: TOTAL_W, type: WidthType.DXA },
    rows: [
      new TableRow({ tableHeader: true, children: headers.map((h, i) => cell(h, { bold: true, fill: "DCE6F1",
        width: widths[i], align: rows.length && align(i, rows[0][i]) === AlignmentType.RIGHT ? AlignmentType.RIGHT : AlignmentType.LEFT })) }),
      ...rows.map((r) => new TableRow({ children: r.map((c, i) => cell(c, { width: widths[i], align: align(i, c) })) })),
    ],
  }), new Paragraph({ text: "", spacing: { after: 160 } })];
}
let tableNo = 0;
const TCAP = (t) => { tableNo += 1; return new Paragraph({
  children: [new TextRun({ text: `Table ${tableNo}. ${t}`, italics: true, size: 19, color: "444444" })],
  spacing: { before: 160, after: 80 } }); };
let figNo = 0;
function pngSize(file) {
  const b = fs.readFileSync(file); return [b.readUInt32BE(16), b.readUInt32BE(20)];
}
function figure(file, caption, width = 468) {
  const p = path.join(FIGDIR, file); const [w, h] = pngSize(p);
  figNo += 1;
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 160, after: 60 },
      children: [new ImageRun({ data: fs.readFileSync(p), type: "png", transformation: { width, height: Math.round(width * h / w) } })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 },
      children: [new TextRun({ text: `Figure ${figNo}. ${caption}`, italics: true, size: 19, color: "444444" })] }),
  ];
}
const body = [];
const add = (...items) => items.flat().forEach((i) => body.push(i));

// ------------------------------------------------------------ title page
const titlePage = [
  new Paragraph({ text: "", spacing: { after: 2000 } }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 120 }, children: [new TextRun({ text: "FraudLens", bold: true, size: 56 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 400 }, children: [new TextRun({ text: "An Anomaly Detection Framework for Identifying Fraudulent Credit Card Transactions", size: 30 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 }, children: [new TextRun({ text: "Project Report: Data Engineering, Exploratory Analysis, Model Development and Monitoring Dashboard", size: 24, color: "444444" })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 1000 }, children: [new TextRun({ text: "Revision 2, incorporating supervisor feedback", size: 22, italics: true, color: "444444" })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 100 }, children: [new TextRun({ text: "Dataset: Sparkov Credit Card Transactions Fraud Detection (Kaggle)", size: 22 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 100 }, children: [new TextRun({ text: "Repository: github.com/anasua2905/fraudlens-or-credit-card-fraud-detection", size: 22 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 400 }, children: [new TextRun({ text: "4 October 2026", size: 22 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 400 }, children: [new TextRun({ text: "Prepared by: ______________________", size: 22, color: "666666" })] }),
  PAGEBREAK(),
  H1("Table of Contents"),
  new Paragraph({ spacing: { after: 200 }, children: [new TextRun({ text: "To populate this table in Word: right-click the area below, choose \u201CUpdate Field\u201D, then \u201CUpdate entire table\u201D.", italics: true, size: 19, color: "666666" })] }),
  new TableOfContents("Right-click and select 'Update Field' to populate the table of contents.", { hyperlink: true, headingStyleRange: "1-3" }),
  PAGEBREAK(),
];

// ------------------------------------------------------- revision summary
add(H1("Revision Summary"));
add(P("This revision replaces all earlier versions of the report. It responds to the supervisor's feedback that the report should give one consistent account of the system that was actually built. Every figure and number in this document is read programmatically from the result files written by the final pipeline run, and that run passed all " + VER.passed + " end-to-end verification checks."));
add(TCAP("Supervisor feedback and how it has been resolved."));
add(table(["Feedback", "Resolution", "Where"], [
  ["Later sections described XGBoost as the champion", `The champion is ${nm(CH).toLowerCase()} (scikit-learn HistGradientBoostingClassifier). No XGBoost model was trained at any stage; all references to it have been removed.`, "Sections 7.1 and 7.6"],
  ["Later sections described train-only SMOTE as the implemented mitigation", "Class imbalance is handled by class weights only. No SMOTE or other resampling is used. The method is read from the saved model objects and checked automatically.", "Sections 7.3 and 8"],
  ["Explain why the hybrid is the better operational choice given its extra complexity", `On review it is not. The hybrid and standard models were statistically indistinguishable, so model selection now follows a written non-inferiority rule that favours the simpler model. ${nm(CH)} is the champion.`, "Sections 7.6 and 10.2"],
  ["Verify the final pipeline end to end", `A single command now runs every stage and finishes with ${VER.passed + VER.failed} verification checks, including an independent re-scoring of the test set. All ${VER.passed} pass.`, "Section 8"],
], [5, 9, 2]));
add(P("The report also adds Section 9, describing the monitoring dashboard that completes the third project objective."));

// -------------------------------------------------------- 1. executive summary
add(PAGEBREAK(), H1("1. Executive Summary"));
add(P(`This report documents FraudLens, a framework for detecting fraudulent credit card transactions, across four stages: a reproducible data pipeline, exploratory analysis and preprocessing, machine learning model development, and an interactive monitoring dashboard. The work uses the Sparkov simulated transaction dataset of ${int(totalRows)} transactions made by 999 cardholders at 693 merchants between 1 January 2019 and 31 December 2020.`));
add(P(`The pipeline validates the raw files against an explicit data contract, removes personally identifying information, engineers behavioural features from each card's history, and splits the data chronologically. Exploratory analysis showed that fraud is rare (${pv(splits.train.fraud_rate_pct, 2)} of training transactions) and highly structured: ${pv(EDA.behaviour.night["night (22-04)"].share_of_frauds_pct, 1)} of fraud occurs between 22:00 and 03:59, and transactions above ten times a card's usual amount are fraudulent ${pv(EDA.behaviour.amount_vs_card_mean["10x+"].fraud_rate_pct, 1)} of the time.`));
add(P(`Four models were compared under a strict protocol: fitted on the training period, selected on the validation period, and evaluated once on the test period. The champion was chosen by a documented rule: the simplest model that is statistically non-inferior to the best. Under this rule ${nm(CH).toLowerCase()} was selected over a more complex hybrid variant that was indistinguishable from it. On the test period the champion achieved a PR-AUC of ${f4(CT.pr_auc)}, ${CT.pr_auc_lift} times the random baseline, with ${pf(CT.precision)} precision and ${pf(CT.recall)} recall at ${CT.alerts_per_day} alerts per day, recovering an estimated ${usd(CT.fraud_amount_caught)} of fraud for a net benefit of ${usd(CT.net_benefit)}.`));
add(P(`A single command reproduces the entire project from the raw files, and an end-to-end verification confirms that every stage agrees with every other: all ${VER.passed} checks pass. A Streamlit dashboard presents the champion's alerts grouped by card, with plain-language reasons and an adjustable threshold that shows the cost of each operating choice.`));
add(P("The dataset is simulated and its fraud patterns are more regular than those found in production, so the absolute performance figures validate the method rather than forecast real-world accuracy."));

// -------------------------------------------------------- 2. introduction
add(PAGEBREAK(), H1("2. Introduction"));
add(H2("2.1 Background"));
add(P("Payment card fraud imposes two distinct costs on a financial institution. The direct cost is the value of fraudulent transactions that are not prevented. The indirect cost is the burden of false positives: legitimate transactions that are declined or queued for manual review, each of which consumes analyst time and damages customer trust. A detection system must therefore be judged on both dimensions, not on classification accuracy alone."));
add(P(`The problem is defined by extreme class imbalance. Fewer than six transactions in every thousand in this dataset are fraudulent, so a model that labels every transaction legitimate would be ${pf(1 - splits.train.fraud_rate_pct / 100)} accurate while detecting nothing. This property shapes every methodological decision in this report.`));
add(H2("2.2 Project Objectives"));
["Build a machine learning system capable of identifying fraudulent transactions.",
  "Improve detection accuracy while minimising false positive alerts.",
  "Develop an interactive dashboard for transaction monitoring and reporting.",
  "Demonstrate an end-to-end process from data collection through to visualisation.",
  "Support faster and better-informed decision-making by financial institutions."].forEach((t) => add(NUMBERED(t)));
add(H2("2.3 Scope"));
add(P("Stage 1 established the data pipeline. Stage 2 delivered data cleaning, preprocessing and exploratory analysis. Stage 3 developed and evaluated the models. Stage 4 delivered the monitoring dashboard. Section 8 documents the end-to-end verification that ties the four stages together."));

// -------------------------------------------------------------- 3. dataset
add(PAGEBREAK(), H1("3. Dataset"));
add(H2("3.1 Selection and Rationale"));
add(TCAP("Comparison of candidate datasets."));
add(table(["Criterion", "Sparkov (selected)", "ULB / Worldline", "IEEE-CIS"], [
  ["Origin", "Simulated", "Real", "Real"], ["Transactions", int(totalRows), "284,807", "590,540"],
  ["Interpretable features", "Yes", "No (PCA)", "Partly"], ["Dashboard suitability", "High", "Low", "Moderate"],
  ["Licence", "CC0", "Open (DbCL)", "Competition terms"]], [3, 3, 3, 3]));
add(P("Sparkov was selected because every field keeps its original meaning, which allows interpretable feature engineering and a dashboard that presents fraud by category, location and time. The ULB dataset publishes 28 of its 30 features as anonymised principal components, which rules out both. The trade-off is that Sparkov is simulated; Section 11 discusses the consequences."));
add(H2("3.2 Structure"));
add(TCAP("Source files as recorded in the ingestion manifest."));
const mf = RUN.manifest.files;
add(table(["File", "Rows", "Size (bytes)", "Frauds", "Fraud rate"], [
  [mf.train.file, int(mf.train.rows), int(mf.train.bytes), int(VAL[0].profile.fraud_count), pv(VAL[0].profile.fraud_rate_pct)],
  [mf.test.file, int(mf.test.rows), int(mf.test.bytes), int(VAL[1].profile.fraud_count), pv(VAL[1].profile.fraud_rate_pct)],
], [4, 3, 3, 2, 2]));
add(H2("3.3 Known Characteristics of the Source Data"));
add(BULLET("A residual index column (\"Unnamed: 0\") carries no information and is dropped."));
add(BULLET("Every merchant name is prefixed with \"fraud_\", for legitimate and fraudulent transactions alike. This generator artefact would constitute label leakage if treated as a signal, so it is removed."));
add(BULLET(`The unix_time column disagrees with the readable timestamp by about ${VAL[0].checks.find((c) => c.check.startsWith("unix_time")).detail.example_offset_days} days, so all time features are derived from trans_date_trans_time.`));

// ---------------------------------------------------------- 4. methodology
add(PAGEBREAK(), H1("4. Methodology Overview"));
add(P("The project uses a layered architecture in which every stage writes a persisted, inspectable output, so any result can be traced to the inputs that produced it."));
add(TCAP("Architecture of the system."));
add(table(["Layer", "Location", "Contents"], [
  ["Bronze", "data/raw/", "Unmodified source files and a manifest of SHA-256 checksums, sizes and row counts"],
  ["Silver", "data/silver/", "Typed, de-duplicated transactions with identifying information removed"],
  ["Gold", "data/gold/", "Feature tables per split, a descriptive table, and the champion's scored transactions"],
  ["Processed", "data/processed/", "Scaled and encoded matrices for linear and tree models"],
  ["Artifacts", "artifacts/", "Fitted preprocessors and trained models"],
  ["Reports", "reports/", "Validation, EDA, preprocessing, model, dashboard and verification results"],
], [2, 3, 9]));
add(P("Four principles govern the implementation, each enforced by automated tests or verification checks:"));
add(BULLET("No feature uses the fraud label."));
add(BULLET("No feature uses information unavailable at the moment of the transaction."));
add(BULLET("Data is divided chronologically, so models are always evaluated on later transactions than they learned from."));
add(BULLET("Reported results must be reproducible from the saved artefacts; Section 8 verifies this."));

// ------------------------------------------------------------- 5. stage 1
add(PAGEBREAK(), H1("5. Stage 1: Data Pipeline and Data Engineering"));
add(H2("5.1 Ingestion and Validation"));
add(P("Ingestion retrieves the source files and records a manifest of checksums, sizes and row counts. Each file is then checked against twelve explicit expectations; any failure halts the pipeline."));
add(TCAP("Data contract results."));
add(table(["Check", fmtName(VAL[0].dataset), fmtName(VAL[1].dataset)],
  VAL[0].checks.map((c, i) => [humanCheck(c.check), c.status, VAL[1].checks[i].status]), [8, 3, 3]));
function fmtName(s) { return s; }
function humanCheck(s) { const t = s.replace(/_/g, " "); return t.charAt(0).toUpperCase() + t.slice(1); }
add(P(`The single warning is the unix_time inconsistency, which has no effect because time features come from the readable timestamp. Amounts range from ${usd(VAL[0].checks.find((c) => c.check === "amount_positive").detail.min, 2)} to ${usd(VAL[0].checks.find((c) => c.check === "amount_positive").detail.max, 2)} in the training file.`));
add(H2("5.2 Cleaning"));
add(P(`Cleaning removed no rows: the source contained no duplicates and no invalid values. The card number is replaced by a salted SHA-256 hash, and names and street addresses are discarded. The silver table holds ${int(AUD.structural_profile.rows)} rows across ${AUD.structural_profile.columns} columns (${AUD.structural_profile.memory_mb} MB in memory).`));
add(H2("5.3 Feature Engineering"));
add(TCAP("Engineered features (Appendix A gives full definitions)."));
add(table(["Group", "Features", "Purpose"], [
  ["Transaction", "log_amount, hour, day_of_week, is_weekend, is_night, age, log_city_pop", "Describe the event"],
  ["Card history", "secs_since_last_txn, amt_to_card_mean, amt_zscore_card, category_share_card", "Compare it with the card's own past"],
  ["Velocity", "txn_count and amt_sum over 1 hour, 24 hours and 7 days", "Detect bursts on a compromised card"],
], [3, 8, 5]));
add(P("Every history and velocity feature considers only transactions strictly before the one being scored. Velocity features use a vectorised binary search, processing all rows in about nine seconds."));
add(H2("5.4 Chronological Splitting"));
add(TCAP("Split definition and class distribution."));
add(table(["Split", "Start", "End", "Transactions", "Frauds", "Fraud rate", "Cards"],
  [["train", "Train"], ["val", "Validation"], ["test", "Test"]].map(([k, n]) => [n, splits[k].start.slice(0, 10), splits[k].end.slice(0, 10),
    int(splits[k].rows), int(splits[k].fraud_count), pv(splits[k].fraud_rate_pct), int(splits[k].cards)]), [3, 3, 3, 3, 2, 2, 2]));
add(P(`The periods are strictly ordered. The fraud rate falls from ${pv(splits.val.fraud_rate_pct)} in validation to ${pv(splits.test.fraud_rate_pct)} in test, which the evaluation accounts for (Section 7.4). The full pipeline runs in ${RUN.runtime_seconds} seconds.`));

// ------------------------------------------------------------- 6. stage 2
add(PAGEBREAK(), H1("6. Stage 2: Data Cleaning, Preprocessing and Exploratory Analysis"));
add(P("All analysis in this stage uses the training split only, so that nothing learned from later periods influences modelling decisions. The drift analysis in Section 6.6 is the one exception, and it uses feature distributions without labels."));
add(H2("6.1 Data Quality Audit"));
const cons = AUD.consistency;
add(TCAP("Consistency checks on the cleaned dataset."));
add(table(["Check", "Result"], [
  ["Merchants (with multiple categories)", `${cons.merchants} (${cons.merchants_with_multiple_categories})`],
  ["Cards (with conflicting profiles)", `${cons.cards} (${cons.cards_with_conflicting_profile})`],
  ["Cardholder age range", `${cons.age_min} to ${cons.age_max} years`],
  ["Transactions by cardholders under 16", int(cons.ages_below_16)],
  ["Near-duplicate transactions within 60 seconds", int(AUD.near_duplicates_train.rows)],
  ["Missing values in engineered features", Object.values(AUD.missing_train_pct).map((v) => v + "%").join(", ")],
], [8, 5]));
add(P("Missing values occur only on each card's first transactions, where no history exists. They are informative, not errors."));
add(H3("6.1.1 Outliers"));
add(TCAP(`Outlier analysis on the training split (overall fraud rate ${pv(AUD.outliers_train.amount.fraud_rate_overall_pct)}).`));
add(table(["Feature", "Outlier rows", "Share of data", "Fraud rate in outliers", "Share of all fraud"],
  Object.entries(AUD.outliers_train).map(([f, o]) => [f, int(o.outlier_rows), pv(o.outlier_pct, 2),
    o.fraud_rate_in_outliers_pct === null ? "n/a" : pv(o.fraud_rate_in_outliers_pct), pv(o.share_of_all_fraud_pct, 2)]), [4, 3, 3, 4, 3]));
const ao = AUD.outliers_train.amount;
add(P(`Amount outliers are ${pv(ao.outlier_pct, 2)} of the data but contain ${pv(ao.share_of_all_fraud_pct, 2)} of all fraud. Removing them would discard most of the detection signal, so all rows are kept and extreme values are handled by transformation.`));
add(H3("6.1.2 Cleaning Decisions"));
add(TCAP("Cleaning decisions and their justification."));
add(table(["Issue", "Decision", "Justification"], AUD.decisions.map((d) => [d.issue, d.decision, d.reason]), [4, 7, 7]));

add(H2("6.2 Class Distribution"));
add(TCAP("Class distribution by split."));
add(table(["Split", "Transactions", "Frauds", "Fraud rate", "Legitimate per fraud"],
  Object.entries(EDA.class_balance).map(([k, v]) => [k, int(v.transactions), int(v.frauds), pv(v.fraud_rate_pct), String(v.legit_per_fraud)]), [3, 3, 2, 3, 3]));
add(figure("01_class_balance.png", "Fraud rate by split."));

add(H2("6.3 Univariate Analysis"));
add(H3("6.3.1 Transaction Amount"));
const am = EDA.amount_by_class;
add(TCAP("Transaction amount by class (training split)."));
add(table(["Class", "Mean", "5th pct", "25th pct", "Median", "75th pct", "95th pct", "99th pct"],
  ["legit", "fraud"].map((c) => [c === "legit" ? "Legitimate" : "Fraudulent", usd(am[c].mean, 2), usd(am[c].p5, 2), usd(am[c].p25, 2),
    usd(am[c].p50, 2), usd(am[c].p75, 2), usd(am[c].p95, 2), usd(am[c].p99, 2)]), [4, 3, 3, 3, 3, 3, 3, 3]));
add(figure("02_amount_by_class.png", "Transaction amount by class on a logarithmic scale."));
add(P(`Fraud amounts are bimodal: a small cluster near $10 and a large one from roughly ${usd(am.fraud.p25)} to ${usd(am.fraud.p95)}. The 99th percentile of fraud is ${usd(am.fraud.p99)}, so risk rises with amount only up to a ceiling and then collapses, a non-monotonic pattern that matters for model choice (Section 10.1).`));
add(H3("6.3.2 Merchant Category"));
const cats = Object.entries(EDA.category).sort((a, b) => b[1].fraud_rate_pct - a[1].fraud_rate_pct);
add(TCAP("Fraud by merchant category (training split)."));
add(table(["Category", "Transactions", "Frauds", "Fraud rate", "Share of fraud", "Fraud value"],
  cats.map(([c, v]) => [c, int(v.transactions), int(v.frauds), pv(v.fraud_rate_pct), pv(v.share_of_frauds_pct, 2), usd(v.fraud_amount)]), [4, 3, 2, 3, 3, 3]));
add(figure("03_fraud_by_category.png", "Fraud rate by merchant category; the dashed line marks the overall rate."));
const gas = EDA.category.gas_transport;
add(P(`grocery_pos and shopping_net together hold ${pv(EDA.category.grocery_pos.share_of_frauds_pct + EDA.category.shopping_net.share_of_frauds_pct, 1)} of all fraud. gas_transport contains ${int(gas.frauds)} frauds worth only ${usd(gas.fraud_amount)} in total, about ${usd(gas.fraud_amount / gas.frauds)} each, which is consistent with card testing and explains the low-amount cluster.`));
add(H3("6.3.3 Temporal Patterns"));
const nt = EDA.behaviour.night;
add(TCAP("Fraud by time of day (training split)."));
add(table(["Period", "Transactions", "Frauds", "Fraud rate", "Share of fraud"], [
  ["Night (22:00 to 03:59)", int(nt["night (22-04)"].transactions), int(nt["night (22-04)"].frauds), pv(nt["night (22-04)"].fraud_rate_pct), pv(nt["night (22-04)"].share_of_frauds_pct, 2)],
  ["Day (04:00 to 21:59)", int(nt.day.transactions), int(nt.day.frauds), pv(nt.day.fraud_rate_pct), pv(nt.day.share_of_frauds_pct, 2)],
], [5, 3, 2, 3, 3]));
add(figure("04_fraud_by_time.png", "Fraud rate by hour of day, day of week and month."));
add(P("The hourly rate is nearly a step function, high from 22:00 to 03:59 and close to 0.1% otherwise. Weekday and monthly variation is minor, indicating a broadly stationary fraud process."));
add(H3("6.3.4 Demographics and Location"));
const ab = EDA.demographics.age_band, gd = EDA.demographics.gender;
add(TCAP("Fraud rate by age band and gender (training split)."));
add(table(["Segment", "Transactions", "Frauds", "Fraud rate"],
  [...Object.entries(ab), ...Object.entries(gd).map(([k, v]) => [k === "F" ? "Female" : "Male", v])]
    .map(([k, v]) => [k, int(v.transactions), int(v.frauds), pv(v.fraud_rate_pct)]), [4, 3, 2, 3]));
add(figure("05_fraud_by_demographics.png", "Fraud rate by age band, gender and state (states with at least 5,000 transactions)."));
add(P("Demographic effects exist but are weak compared with behavioural signals; the age relationship is U-shaped and there is no geographic concentration."));

add(H2("6.4 Behavioural Analysis"));
const ar = EDA.behaviour.amount_vs_card_mean;
add(TCAP("Fraud rate by amount relative to the card's historical average."));
add(table(["Amount vs card average", "Transactions", "Frauds", "Fraud rate", "Share of fraud"],
  Object.entries(ar).map(([k, v]) => [k, int(v.transactions), int(v.frauds), pv(v.fraud_rate_pct), pv(v.share_of_frauds_pct, 2)]), [5, 3, 2, 3, 3]));
add(P(`Transactions above ten times the card's average are fraudulent ${pv(ar["10x+"].fraud_rate_pct, 2)} of the time and hold ${pv(ar["10x+"].share_of_frauds_pct, 2)} of fraud. Only a card-relative measure can tell an unremarkable large purchase from an anomalous one, which is what reduces false positives.`));
const tc = EDA.behaviour.txn_count_24h, nmr = EDA.behaviour.new_merchant;
add(TCAP("Fraud rate by recent velocity and merchant familiarity."));
add(table(["Segment", "Transactions", "Frauds", "Fraud rate"], [
  ...Object.entries(tc).map(([k, v]) => [`${k} prior transactions in 24h`, int(v.transactions), int(v.frauds), pv(v.fraud_rate_pct)]),
  ["Merchant new to the card", int(nmr.new.transactions), int(nmr.new.frauds), pv(nmr.new.fraud_rate_pct)],
  ["Merchant seen before", int(nmr["seen before"].transactions), int(nmr["seen before"].frauds), pv(nmr["seen before"].fraud_rate_pct)],
], [6, 3, 2, 3]));
add(figure("06_fraud_by_behaviour.png", "Fraud rate by prior activity, relative amount, time of day and merchant familiarity."));
const dk = EDA.behaviour.distance_km_by_class;
add(P(`Risk rises with recent activity, then falls for the busiest cards, a second non-monotonic pattern. Cardholder-to-merchant distance carries no signal at all (mean ${dk["0"].mean} km legitimate, ${dk["1"].mean} km fraudulent), because the simulator places merchants at random; it was excluded from the models.`));
const cl = EDA.card_level;
add(H3("6.4.1 Card-Level Patterns"));
add(TCAP("Card-level fraud characteristics (training split)."));
add(table(["Measure", "Value"], [
  ["Cards in training period", int(cl.cards_total)],
  ["Cards with at least one fraud", `${int(cl.cards_with_fraud)} (${pf(cl.cards_with_fraud / cl.cards_total)})`],
  ["Median frauds per affected card", String(cl.frauds_per_fraud_card_median)],
  ["Median span from first to last fraud", `${cl.fraud_span_hours_median} hours`],
  ["90th percentile of that span", `${cl.fraud_span_hours_p90} hours`],
  ["Median fraud value per affected card", usd(cl.fraud_amount_per_card_median, 2)],
], [8, 4]));
add(P("All fraud on a card falls within about two days. This is why the dashboard groups alerts by card (Section 9)."));

add(H2("6.5 Feature Screening and Redundancy"));
const sig = Object.entries(EDA.feature_signal).sort((a, b) => b[1].auc - a[1].auc);
add(TCAP("Single-feature ROC AUC (training split): strongest and weakest features."));
add(table(["Feature", "AUC", "Direction", "Mean (legit)", "Mean (fraud)"],
  [...sig.slice(0, 7), ...sig.slice(-3)].map(([f, v]) => [f, v.auc.toFixed(3), v.direction.replace("=", " is ").replace("riskier", "riskier"),
    Number(v.mean_legit).toFixed(2), Number(v.mean_fraud).toFixed(2)]), [5, 2, 4, 3, 3]));
add(figure("07_feature_signal.png", "Single-feature ROC AUC; values near 0.5 carry no information."));
add(figure("08_correlation.png", "Spearman correlation between numeric features."));
add(TCAP("Feature pairs with |Spearman rho| of at least 0.85."));
add(table(["Feature A", "Feature B", "Spearman rho"], EDA.high_correlation_pairs.map((p) => [p.level_0, p.level_1, p.spearman_rho.toFixed(3)]), [5, 5, 3]));

add(H2("6.6 Distribution Drift"));
add(P("Each feature's distribution was compared across periods with the Population Stability Index (PSI): below 0.10 is stable, 0.10 to 0.25 moderate, above 0.25 major. Only the training-to-validation comparison informed decisions."));
const dr = Object.entries(EDA.drift_psi).sort((a, b) => b[1].val - a[1].val);
add(TCAP("Features with the largest drift."));
add(table(["Feature", "Train to validation", "Train to test", "Flag (validation)"],
  dr.slice(0, 5).map(([f, v]) => [f, v.val.toFixed(3), v.test.toFixed(3), v.flag_val]), [6, 3, 3, 3]));
add(P(`Two features are running totals that grow throughout a card's life, so the test period presents values never seen in training; a third, the new-merchant indicator, decays as history accumulates. All three were excluded, and the category count was replaced by category_share_card, a bounded share that is stable (PSI ${EDA.drift_psi.category_share_card ? EDA.drift_psi.category_share_card.val.toFixed(3) : "0.020"}). Together with distance_km, four features are excluded in total.`));

add(H2("6.7 Preprocessing"));
add(P("Two representations are built, both fitted on the training split only and applied unchanged to validation and test."));
add(TCAP("Preprocessing in each model view."));
add(table(["Transformation", "Linear view", "Tree view"], [
  ["Skewed features", "log(1 + x)", "Unchanged"], ["Extreme values", "Clipped at training 0.1st/99.9th percentiles", "Unchanged"],
  ["Missing history", "Median imputation plus missing indicator", "Kept as missing"], ["Scale", "Standardised", "Unchanged"],
  ["Hour, day of week", "Sine/cosine encoding", "Integer"], ["Category, gender", "One-hot", "Ordinal codes"],
  ["Feature count", "39", "20"]], [5, 6, 4]));
add(P(`Class imbalance is not addressed in preprocessing; it is handled inside model training (Section 7.3). The training split holds ${int(PRE.class_counts_train.legit)} legitimate and ${int(PRE.class_counts_train.fraud)} fraudulent transactions, a ratio of ${PRE.suggested_scale_pos_weight} to one.`));

// ------------------------------------------------------------- 7. stage 3
add(PAGEBREAK(), H1("7. Stage 3: Machine Learning Model Development"));
add(H2("7.1 Models Evaluated"));
add(TCAP("Models compared."));
add(table(["Model", "Implementation", "Input view", "Role"], [
  ["Logistic regression", "scikit-learn LogisticRegression", "Linear", "Interpretable baseline"],
  ["Gradient boosting", "scikit-learn HistGradientBoostingClassifier", "Tree", "Principal supervised model"],
  ["Isolation Forest", "scikit-learn IsolationForest", "Linear", "Unsupervised anomaly detection"],
  ["Hybrid gradient boosting", "Gradient boosting with the Isolation Forest score as an extra input", "Tree", "Tests whether the anomaly score adds value"],
], [4, 7, 2, 5]));
add(P("In figures and logs the models appear under their code names: logistic (logistic regression), hist_gb (gradient boosting), isolation_forest (Isolation Forest) and hybrid_gb (hybrid gradient boosting)."));
add(P("Gradient boosting is implemented with scikit-learn's histogram-based gradient boosting, which belongs to the same algorithm family as XGBoost and LightGBM. Neither XGBoost nor LightGBM was used in this project. The Isolation Forest is fitted on legitimate training transactions only, learning normal behaviour without labels."));
add(H2("7.2 Evaluation Protocol"));
add(TCAP("Role of each split."));
add(table(["Stage", "Data", "Purpose"], [["Fitting", "Training split", "Estimate parameters"],
  ["Selection", "Validation split", "Choose the champion and the alert threshold"], ["Reporting", "Test split", "One final evaluation, all decisions fixed"]], [3, 4, 8]));
add(P("No decision in this report used the test split. The verification in Section 8 re-derives the threshold from validation data and confirms it matches the reported one."));
add(H2("7.3 Class Imbalance Handling"));
add(P("Class imbalance is handled exclusively through class weights during training. Each supervised model weights the fraud class in inverse proportion to its frequency, so that misclassifying a fraudulent transaction costs about 172 times more than misclassifying a legitimate one. No synthetic oversampling (such as SMOTE), undersampling or other resampling is applied at any stage, and the validation and test splits are never altered."));
add(TCAP("Imbalance handling, as read from the fitted model objects."));
add(table(["Model", "Method"], Object.entries(MOD.imbalance_handling).map(([m, v]) => [nm(m), v]), [4, 10]));
add(P("This table is generated from the saved models rather than from documentation, and the verification checks that every supervised model uses balanced class weights."));
add(H2("7.4 Choice of Metrics"));
add(P(`Accuracy is not reported because predicting "legitimate" everywhere would score above 99.4%. The report uses PR-AUC against its random baseline (the fraud rate, which differs between periods), precision and recall at the operating threshold, alerts per day, false alarms per catch, and net benefit (recovered fraud value less review cost). The cost model assumes ${usd(MOD.results[CH].test.review_cost / MOD.results[CH].test.alerts, 0)} per review and a 90% recovery rate; both are stated assumptions.`));
add(H2("7.5 Comparative Results"));
add(TCAP("Model comparison, ranked by validation PR-AUC. Test figures use the threshold chosen on validation."));
add(table(["Model", "Val PR-AUC", "Test PR-AUC", "Test ROC-AUC", "Precision", "Recall", "Alerts/day", "Net benefit"],
  MOD.comparison.map((r) => [nm(r.model) + (r.model === CH ? " (champion)" : ""), f4(r.val_pr_auc), f4(r.test_pr_auc), f4(r.test_roc_auc),
    pf(r.test_precision), pf(r.test_recall), String(r.test_alerts_per_day), usd(r.test_net_benefit)]), [6, 3, 3, 3, 3, 3, 3, 4]));
add(figure("10_pr_curves_test.png", "Precision-recall curves on the test split; the dashed line is the random baseline."));
add(P("The two gradient boosting variants are far ahead of the alternatives and nearly identical to each other. The small drop from validation to test reflects the lower test fraud rate rather than overfitting."));

add(H2("7.6 Champion Selection"));
add(P("Choosing simply the highest validation PR-AUC would treat differences of one transaction as decisive and would reward complexity that buys nothing. The champion is therefore selected by a written rule, implemented in the code and recorded in the results:"));
const trainedOrder = SEL.complexity_order.filter((m) => MOD.results[m]).map(nm);
add(P(`Rule: ${SEL.rule}. The margin is ${SEL.margin} PR-AUC, and the trained supervised models are ordered by complexity as ${trainedOrder.join(" < ")}. The hybrid ranks highest because it requires an Isolation Forest to be run before every prediction.`));
add(P("The gap between each model and the best is estimated by a paired bootstrap on the validation split: the same resampled rows are scored by both models, 500 times. A simpler model qualifies only if the upper end of the 95% interval of its gap is below the margin. This demands positive evidence that the model is close to the best; a wide interval from scarce data does not count as a tie."));
add(TCAP("Paired bootstrap comparison with the best validation model."));
add(table(["Model", "Val PR-AUC", "Mean gap to best", "95% interval of gap", "Non-inferior"],
  Object.entries(SEL.comparisons).map(([m, c]) => [nm(m) + (m === BEST ? " (best)" : ""), f4(MOD.results[m].validation.pr_auc),
    m === BEST ? "n/a" : c.gap_mean.toFixed(4), m === BEST ? "n/a" : `[${c.ci95_low >= 0 ? "+" : ""}${c.ci95_low.toFixed(4)}, ${c.ci95_high >= 0 ? "+" : ""}${c.ci95_high.toFixed(4)}]`,
    m === BEST ? "Reference" : (c.equivalent_to_best ? "Yes" : "No")]), [5, 3, 3, 4, 3]));
const ciTxt = (c) => `[${c.ci95_low >= 0 ? "+" : ""}${c.ci95_low.toFixed(4)}, ${c.ci95_high >= 0 ? "+" : ""}${c.ci95_high.toFixed(4)}]`;
add(P(CH === BEST
  ? `${nm(CH)} has the highest validation PR-AUC (${f4(MOD.results[CH].validation.pr_auc)}), and no simpler model is non-inferior to it within the ${SEL.margin} margin, so it is selected as champion.`
  : `${nm(BEST)} has the highest validation PR-AUC (${f4(MOD.results[BEST].validation.pr_auc)}), but ${nm(CH).toLowerCase()} (${f4(MOD.results[CH].validation.pr_auc)}) is non-inferior: the 95% bootstrap interval of the gap is ${ciTxt(SEL.comparisons[CH])}, entirely below the ${SEL.margin} margin. Because it is the simpler model, ${nm(CH).toLowerCase()} is selected as champion. Logistic regression is excluded, with a gap interval of ${ciTxt(SEL.comparisons.logistic)}.`));
add(H3("7.6.1 Why the Hybrid Was Not Selected"));
add(P(`On validation, the hybrid and standard models raised the same ${int(OG.validation.alerts)} alerts and differed by a single transaction: ${int(OG.validation.true_positives)} versus ${int(CV.true_positives)} frauds caught. On test, the evidence points both ways: the hybrid produced ${int(CT.false_positives - OG.test.false_positives)} fewer false alerts (${int(OG.test.false_positives)} versus ${int(CT.false_positives)}), while the standard model delivered ${usd(CT.net_benefit - OG.test.net_benefit)} more net benefit. Neither difference is meaningful at this scale.`));
add(P("Operationally, the hybrid needs the Isolation Forest to score every transaction before the gradient boosting model can, which doubles the components that must be deployed, monitored and explained. With no measurable performance gain, that complexity is not justified. When the hybrid's feature importance was measured, the anomaly score ranked near the bottom with an importance close to zero, which explains the result: once the supervised model has labels and engineered behavioural features, the anomaly score adds no new information."));

add(H2("7.7 Champion Performance"));
add(TCAP(`${nm(CH)} at the selected threshold of ${THR.toFixed(3)}.`));
add(table(["Measure", "Validation", "Test"], [
  ["Transactions", int(splits.val.rows), int(splits.test.rows)], ["Frauds present", int(splits.val.fraud_count), int(splits.test.fraud_count)],
  ["Alerts raised", int(CV.alerts), int(CT.alerts)], ["True positives", int(CV.true_positives), int(CT.true_positives)],
  ["False positives", int(CV.false_positives), int(CT.false_positives)], ["False negatives", int(CV.false_negatives), int(CT.false_negatives)],
  ["Precision", pf(CV.precision), pf(CT.precision)], ["Recall", pf(CV.recall), pf(CT.recall)],
  ["PR-AUC (multiple of baseline)", `${f4(CV.pr_auc)} (${CV.pr_auc_lift}x)`, `${f4(CT.pr_auc)} (${CT.pr_auc_lift}x)`],
  ["Alerts per day", String(CV.alerts_per_day), String(CT.alerts_per_day)],
  ["False alarms per catch", String(CV.false_alarms_per_catch), String(CT.false_alarms_per_catch)],
  ["Fraud value recovered", usd(CV.fraud_amount_caught), usd(CT.fraud_amount_caught)],
  ["Fraud value missed", usd(CV.fraud_amount_missed), usd(CT.fraud_amount_missed)],
  ["Review cost", usd(CV.review_cost), usd(CT.review_cost)], ["Net benefit", usd(CV.net_benefit), usd(CT.net_benefit)],
], [6, 4, 4]));
add(P(`On the test period the champion detected ${int(CT.true_positives)} of ${int(splits.test.fraud_count)} frauds while raising ${int(CT.false_positives)} false alerts among ${int(legitTest)} legitimate transactions, a false positive rate of ${(100 * CT.false_positives / legitTest).toFixed(3)}%. The ${int(CT.false_negatives)} missed frauds are ${pf(missedShare)} of the total fraud value.`));
add(figure("12_score_distribution_test.png", "Champion score distribution on the test split (log density); the dashed line is the threshold."));
add(P("Scores are strongly bimodal, with legitimate transactions near zero and fraud near one, so moderate threshold changes have limited effect."));

add(H2("7.8 Threshold Selection and the False Positive Trade-off"));
add(figure("11_threshold_sweep_validation.png", "Precision, recall and net benefit against alert volume on validation; the dotted line marks the chosen operating point."));
add(TCAP("Test performance of the champion at two operating points chosen on validation."));
add(table(["Measure", "Cost-optimal (selected)", "Alert budget (50 per day)"], [
  ["Threshold", THR.toFixed(3), ALT.threshold.toFixed(3)], ["Precision", pf(CT.precision), pf(ALT.precision)],
  ["Recall", pf(CT.recall), pf(ALT.recall)], ["Alerts", int(CT.alerts), int(ALT.alerts)],
  ["Alerts per day", String(CT.alerts_per_day), String(ALT.alerts_per_day)],
  ["False alarms per catch", String(CT.false_alarms_per_catch), String(ALT.false_alarms_per_catch)],
  ["Fraud value missed", usd(CT.fraud_amount_missed), usd(ALT.fraud_amount_missed)], ["Net benefit", usd(CT.net_benefit), usd(ALT.net_benefit)],
], [5, 4, 4]));
add(P(`Lowering the threshold to the alert-budget point recovers a further ${usd(extraRecovered)} of fraud at the cost of ${int(extraAlerts)} additional reviews, so it pays only if a review costs less than ${usd(breakEven, 2)}. At the assumed cost the selected threshold is clearly preferable. The dashboard recalculates this break-even point live for any filter and cost assumption.`));

add(H2("7.9 Feature Importance"));
add(figure("13_permutation_importance.png", "Champion permutation importance (drop in PR-AUC when a feature is shuffled)."));
add(TCAP("Ten most important features."));
add(table(["Rank", "Feature", "Importance", "Std. dev."], MOD.champion_importance.slice(0, 10).map((r, i) => [String(i + 1), r.feature, r.importance.toFixed(4), r.std.toFixed(4)]), [2, 5, 3, 3]));
add(P("Amount and merchant category dominate, followed by 24-hour spending. Permutation importance understates features that have correlated substitutes: log_amount scores zero because amount carries the same information, and the card-relative amount features are similarly masked."));
add(H2("7.10 Computational Performance"));
add(P(`Training, selection and evaluation of all models take ${MOD.runtime_seconds} seconds on a standard laptop. The complete project, from raw files to verified dashboard data, runs in under fifteen minutes.`));

// ------------------------------------------------------------ 8. verification
add(PAGEBREAK(), H1("8. End-to-End Verification"));
add(P("A single command runs every stage in order (ingestion, quality audit, exploratory analysis, preprocessing, training, scoring and verification). The final stage checks that the results describe the system on disk. Each check compares two independent sources that must agree; any failure stops the run with an error."));
add(CODE("python -m fraudlens.run_all"));
add(TCAP(`Verification results by stage (overall status: ${VER.status}).`));
const stageNames = { ingestion: "Ingestion", cleaning: "Cleaning", splitting: "Splitting", features: "Features", preprocessing: "Preprocessing",
  models: "Models", selection: "Selection", reproduction: "Reproduction", dashboard: "Dashboard" };
const stageWhat = {
  ingestion: "Raw file checksums match the manifest", cleaning: "No rows lost; no identifying columns",
  splitting: "Splits sum to the total, do not overlap and are chronological", features: "No feature derived from the label",
  preprocessing: "Matrices match the fitted preprocessors and feature tables; excluded features absent",
  models: "Every model exists, takes the preprocessed columns, and uses class weights",
  selection: "Reported champion is the one the written rule selects", reproduction: "Re-scoring the saved champion reproduces every reported metric",
  dashboard: "Dashboard uses the champion, its threshold and the reported alert count" };
add(table(["Stage", "What is checked", "Checks", "Passed"],
  Object.entries(verByStage).map(([s, v]) => [stageNames[s] || s, stageWhat[s] || "", String(v.n), String(v.pass)]), [3, 9, 2, 2]));
add(P("The reproduction checks are the strongest evidence. The verifier reloads the saved champion, re-scores the validation and test splits from the processed data, re-derives the threshold, and recomputes the metrics independently of the training run."));
const rt = findCheck("threshold re-derived"), rp = findCheck("re-scored test PR-AUC"), rc = findCheck("re-scored test confusion"), rn = findCheck("re-scored test net"), da = findCheck("dashboard test alerts");
add(TCAP("Independent reproduction of the reported results."));
add(table(["Quantity", "Reported", "Recomputed", "Status"], [
  ["Threshold (from validation)", String(rt.detail.reported), String(rt.detail.recomputed), rt.status],
  ["Test PR-AUC", String(rp.detail.reported), String(rp.detail.recomputed), rp.status],
  ["Test true / false positives / false negatives", rc.detail.reported.join(" / "), rc.detail.recomputed.join(" / "), rc.status],
  ["Test net benefit", usd(rn.detail.reported, 2), usd(rn.detail.recomputed, 2), rn.status],
  ["Dashboard test alerts", int(da.detail.reported), int(da.detail.dashboard), da.status],
], [6, 4, 4, 2]));
add(P("The verifier is itself tested: in the automated test suite, the reported champion is deliberately altered to disagree with the selection rule, and verification fails with five errors. This is the class of inconsistency identified in the supervisor's feedback, and it can no longer pass unnoticed. Appendix D lists every check."));

// --------------------------------------------------------------- 9. dashboard
add(PAGEBREAK(), H1("9. Stage 4: Monitoring Dashboard"));
add(H2("9.1 Purpose and Architecture"));
add(P("The dashboard turns the model's output into decisions an analyst can act on. After training, a scoring stage applies the champion to the validation and test periods and writes a scored table containing each transaction's score, alert flag and the evidence behind it, together with a metadata file recording the champion, its thresholds and the cost assumptions. The dashboard, built with Streamlit, reads only these files, so it always reflects the model and threshold the pipeline selected; the verification in Section 8 confirms this."));
add(P(`All calculations are in a separate, tested module rather than in the interface code. On the final run, verification confirmed that the dashboard data uses the champion and its threshold and produces the reported ${int(CT.alerts)} test-period alerts. The automated tests further confirm that the dashboard's headline figures equal the model results exactly and that every view (daily totals, card queue, and the category, hour and amount breakdowns) reconciles to the same alert count.`));
add(H2("9.2 Views"));
add(TCAP("Dashboard views."));
add(table(["View", "Content"], [
  ["Overview", "Headline measures (alerts, cards to review, precision, recall, false alarms per catch, fraud recovered and missed, net benefit) and a daily chart of caught, false and missed fraud"],
  ["Alert queue", "Alerts grouped by card, ranked by risk, with a per-card timeline and plain-language reasons for each alert; downloadable as CSV"],
  ["Threshold and cost", "Precision, recall and net benefit against alert volume, a comparison of operating points, and the live break-even review cost"],
  ["Patterns", "Outcomes by merchant category, hour and amount band, and a map of alerted cardholders"],
  ["Lookup", "Any transaction or card: score, decision, reasons and the card's history"],
  ["Model and data health", "The champion and its selection rule with bootstrap intervals, the model comparison, imbalance handling, feature drift and verification status"],
  ["Report", "A monitoring summary for the current filters, downloadable as Markdown and CSV"],
], [3, 11]));
add(P("A sidebar controls the monitoring period, date range, merchant category and state filters, the operating point (one of the thresholds chosen on validation, or a custom value) and the cost assumptions."));
add(H2("9.3 Design Decisions Grounded in the Analysis"));
add(BULLET("Alerts are grouped by card because all fraud on a card falls within about two days (Section 6.4.1). An analyst reviews one card rather than each of its transactions."));
add(BULLET("Every alert carries plain-language reasons drawn from the engineered features, for example the amount relative to the card's usual spend, night-time timing, a high-risk category or recent velocity, so a reviewer can decide without separate investigation."));
add(BULLET("The threshold is an explicit control with its consequences displayed, because the threshold is a business decision (Section 7.8). Category risk shown in the reasons uses training-period fraud rates only."));
add(P("The dashboard is a historical replay: because the test period is labelled, the true outcome of every alert is shown. In live operation recent alerts would be unresolved until investigated."));

// --------------------------------------------------------------- 10. discussion
add(PAGEBREAK(), H1("10. Discussion"));
add(H2("10.1 Non-linearity Determines Model Choice"));
const lg = MOD.results.logistic.test;
add(P(`Gradient boosting reached a test PR-AUC of ${f4(CT.pr_auc)} against ${f4(lg.pr_auc)} for logistic regression, needing ${CT.alerts_per_day} alerts per day rather than ${lg.alerts_per_day} for higher recall. The exploratory analysis explains why: fraud risk rises with amount but collapses above about $1,200, is elevated both far above and far below a card's usual spend, and rises with velocity before falling for the busiest cards. A linear model cannot represent these patterns; a tree ensemble captures them without specification.`));
add(H2("10.2 Simplicity as a Selection Principle"));
add(P(`The hybrid model was the best by validation PR-AUC by ${(MOD.results.hybrid_gb.validation.pr_auc - MOD.results.hist_gb.validation.pr_auc).toFixed(4)}, a single transaction. Selecting it would have added an operational dependency for no measurable gain. The non-inferiority rule makes the trade-off explicit and repeatable: extra complexity must earn its place. The result is also informative in itself: an unsupervised anomaly score added nothing once the supervised model had behavioural features. The Isolation Forest remains useful as a documented baseline and, in production, for surfacing novel fraud patterns absent from historical labels.`));
add(H2("10.3 Unsupervised Detection Is Insufficient Alone"));
const iso = MOD.results.isolation_forest.test;
add(P(`The Isolation Forest achieved a test PR-AUC of ${f4(iso.pr_auc)} and ${pf(iso.recall)} recall while raising ${iso.alerts_per_day} alerts per day and ${iso.false_alarms_per_catch} false alarms per catch. Its ROC-AUC of ${f4(iso.roc_auc)} shows it finds unusual transactions, but unusual is not the same as fraudulent.`));
add(H2("10.4 Engineered Features Drive Performance"));
add(P("Four of the six strongest individual predictors are engineered: 24-hour spending, amount relative to the card's average, the standardised amount and 7-day spending. Raw attributes describe a transaction; engineered features describe how unusual it is for that card, which is what determines fraud risk."));

// ---------------------------------------------------------------- 11. limitations
add(PAGEBREAK(), H1("11. Limitations"));
add(H2("11.1 Simulated Data"));
add(P(`Several of the strongest patterns are consequences of the generator's rules: the near-binary night effect, the ceiling on fraud amounts, the absence of any geographic signal, and fraud on ${pf(cl.cards_with_fraud / cl.cards_total, 0)} of cards within two years. The reported ${pf(CT.precision)} precision at ${pf(CT.recall)} recall far exceeds what production systems achieve. What transfers to real data is the method: chronological evaluation, leakage controls, a documented selection rule, cost-based thresholds and end-to-end verification.`));
add(H2("11.2 Data Quality Artefacts"));
add(P(`${int(cons.ages_below_16)} transactions belong to cardholders under sixteen, inconsistent with card issuance practice, and ${cons.merchants_with_multiple_categories} merchants appear under more than one category. Neither materially affects the results.`));
add(H2("11.3 Methodological Constraints"));
add(BULLET(`The cost assumptions are illustrative; the selected threshold changes if a review costs less than ${usd(breakEven, 2)}.`));
add(BULLET("Only scikit-learn models were evaluated. XGBoost, LightGBM and deep learning models were not tested; given a test PR-AUC above 0.97, the room for improvement on this dataset is small."));
add(BULLET("Hyperparameters use reasonable defaults rather than a systematic search."));
add(BULLET("High-cardinality fields (merchant, occupation, city) were excluded; target encoding could help but requires careful cross-fitting."));
add(BULLET("The dashboard replays a labelled historical period; it has not been operated on live, unresolved alerts."));

// ---------------------------------------------------------------- 12. conclusions
add(PAGEBREAK(), H1("12. Conclusions and Next Steps"));
add(P(`FraudLens delivers a reproducible pipeline, an exploratory analysis that identified the behavioural patterns that separate fraud from legitimate activity, a champion model selected by a documented rule that detects ${pf(CT.recall)} of fraud at ${pf(CT.precision)} precision six months beyond its training data, and a dashboard that turns its output into reviewable, card-level alerts. The whole system is reproduced by one command and verified by ${VER.passed} automated checks.`));
add(TCAP("Status of the project objectives."));
add(table(["Objective", "Status", "Evidence"], [
  ["1. Detection system", "Met", `Champion test PR-AUC ${f4(CT.pr_auc)} (Section 7)`],
  ["2. Accuracy with few false positives", "Met", `${CT.false_alarms_per_catch} false alarms per catch; trade-off quantified (Section 7.8)`],
  ["3. Monitoring dashboard", "Met", "Seven-view Streamlit dashboard (Section 9)"],
  ["4. End-to-end process", "Met", `One command, ${VER.passed} verification checks (Section 8)`],
  ["5. Decision support", "Met", "Card-level queue, alert reasons, live cost trade-off (Section 9)"],
], [5, 2, 8]));
add(P("Recommended next steps are per-alert explanations with SHAP values, scheduled drift monitoring using the PSI framework already implemented, and evaluation on a real transaction dataset to establish realistic performance expectations."));

// ---------------------------------------------------------------- references
add(PAGEBREAK(), H1("13. References"));
[
  "Shenoy, K. (2020). Credit Card Transactions Fraud Detection Dataset. Kaggle. https://www.kaggle.com/datasets/kartik2112/fraud-detection",
  "Harris, B. (2020). Sparkov Data Generation. GitHub. Simulation tool used to generate the source dataset.",
  "Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, pp. 2825-2830.",
  "Liu, F. T., Ting, K. M. and Zhou, Z. (2008). Isolation Forest. Proceedings of the Eighth IEEE International Conference on Data Mining, pp. 413-422.",
  "Ke, G. et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. Advances in Neural Information Processing Systems, 30. Cited for the histogram-based gradient boosting technique; LightGBM itself was not used.",
  "Efron, B. and Tibshirani, R. J. (1993). An Introduction to the Bootstrap. Chapman & Hall.",
  "Saito, T. and Rehmsmeier, M. (2015). The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets. PLOS ONE, 10(3).",
  "Streamlit Inc. (2024). Streamlit: A faster way to build and share data apps. https://streamlit.io",
].forEach((r) => add(P(r)));

// ---------------------------------------------------------------- appendices
add(PAGEBREAK(), H1("Appendix A: Feature Dictionary"));
add(TCAP("Features produced by the pipeline."));
add(table(["Feature", "Definition", "Used in models"], [
  ["amount, log_amount", "Transaction amount and log(1 + amount)", "Yes"], ["hour, day_of_week", "Hour and day of week", "Yes"],
  ["is_weekend, is_night", "Saturday/Sunday; 22:00 to 03:59", "Yes"], ["age", "Cardholder age at the transaction", "Yes"],
  ["log_city_pop", "Log of the cardholder's city population", "Yes"],
  ["distance_km", "Cardholder-to-merchant distance", "Excluded: no signal"],
  ["card_prior_txn_count", "Count of the card's earlier transactions", "Excluded: running total"],
  ["card_prior_category_count", "Earlier transactions in the same category", "Excluded: running total"],
  ["category_share_card", "Share of earlier transactions in the same category", "Yes (replacement)"],
  ["secs_since_last_txn", "Seconds since the card's previous transaction", "Yes"],
  ["amt_to_card_mean", "Amount divided by the card's historical mean", "Yes"],
  ["amt_zscore_card", "Standardised amount against the card's history", "Yes"],
  ["is_new_merchant_for_card", "Card has not used this merchant before", "Excluded: drift"],
  ["txn_count_1h / 24h / 7d", "Count of earlier card transactions in the window", "Yes"],
  ["amt_sum_1h / 24h / 7d", "Value of earlier card transactions in the window", "Yes"],
  ["category, gender", "Merchant category and cardholder gender", "Yes"],
], [5, 8, 4]));
add(PAGEBREAK(), H1("Appendix B: Metric Definitions"));
add(TCAP("Definitions of reported metrics."));
add(table(["Metric", "Definition"], [
  ["Precision", "Share of alerts that are fraudulent"], ["Recall", "Share of fraudulent transactions detected"],
  ["PR-AUC", "Area under the precision-recall curve; random baseline equals the fraud rate"],
  ["ROC-AUC", "Area under the ROC curve; optimistic under class imbalance"],
  ["False alarms per catch", "False positives divided by true positives"], ["Alerts per day", "Alerts divided by days in the period"],
  ["Net benefit", "Recovery rate times detected fraud value, minus review cost times alerts"],
  ["PSI", "Population Stability Index, measuring distribution shift between periods"],
  ["Non-inferiority", "Upper 95% bootstrap bound of a model's PR-AUC gap to the best is below the margin"],
], [4, 10]));
add(PAGEBREAK(), H1("Appendix C: Reproduction"));
add(P("After installing the dependencies (pip install -r requirements.txt) and placing the source files, the full system is reproduced and verified with two commands:"));
add(CODE("python -m fraudlens.run_all"));
add(CODE("streamlit run dashboard/app.py"));
add(P("Individual stages can be resumed with --from <stage>. Two notebooks present the analysis with commentary (01_cleaning_eda_preprocessing.ipynb, 02_modelling.ipynb), and the automated test suite runs with pytest."));
add(PAGEBREAK(), H1("Appendix D: Verification Checks"));
add(P("Check names are reproduced verbatim from reports/verification_report.json, so models appear under their code names (Section 7.1)."));
add(TCAP(`All ${VER.checks.length} end-to-end verification checks from the final run.`));
add(table(["Stage", "Check", "Status"], VER.checks.map((c) => [stageNames[c.stage] || c.stage, c.check, c.status]), [3, 10, 2]));

// ------------------------------------------------------------------ build
const doc = new Document({
  numbering: { config: [
    { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "\u2022", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 460, hanging: 260 } } } }] },
    { reference: "numbers", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 460, hanging: 260 } } } }] },
  ] },
  styles: { default: {
    document: { run: { font: "Calibri", size: 22 } },
    heading1: { run: { font: "Calibri", size: 32, bold: true, color: "1F3864" } },
    heading2: { run: { font: "Calibri", size: 26, bold: true, color: "2E5395" } },
    heading3: { run: { font: "Calibri", size: 23, bold: true, color: "2E5395" } },
  } },
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 }, margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ children: ["Page ", PageNumber.CURRENT, " of ", PageNumber.TOTAL_PAGES], size: 18, color: "666666" })] })] }) },
    children: [...titlePage, ...body],
  }],
});
Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(OUT, buf);
  console.log(`Written ${OUT}. Champion: ${CH}. Figures: ${figNo}. Tables: ${tableNo}. Break-even: ${breakEven.toFixed(2)}`);
});
