import json, pandas as pd
from docx import Document
from docx.shared import Pt, Inches
from common import *
sm = pd.read_csv(OUT/"sentiment_model_comparison.csv"); zn = pd.read_csv(OUT/"zone_model_comparison_nowcast.csv"); zf = pd.read_csv(OUT/"zone_model_comparison.csv")
eda = json.load(open(OUT/"eda_results.json")); k = eda["kde"]; boot = json.load(open(OUT/"bootstrap_rag_vs_tfidf.json")); imp = pd.read_csv(OUT/"zone_feature_importance_nowcast.csv").head(8)
py = pd.read_csv(OUT/"zone_model_per_year_nowcast.csv")
doc = Document(); st = doc.styles["Normal"]; st.font.name = "Calibri"; st.font.size = Pt(11)
def P(t, b=False): p = doc.add_paragraph(); r = p.add_run(t); r.bold = b; return p
def B(t): doc.add_paragraph(t, style="List Bullet")
def T(df, cols, names, fmt="{:.3f}"):
    t = doc.add_table(rows=1, cols=len(cols)); t.style = "Light Grid Accent 1"
    for i, n in enumerate(names): t.rows[0].cells[i].text = n
    for _, r in df.astype(object).iterrows():
        c = t.add_row().cells
        for i, col in enumerate(cols): c[i].text = fmt.format(r[col]) if isinstance(r[col], float) else str(r[col])
    doc.add_paragraph()
doc.add_heading("A Retrieval-Augmented Framework for Xenophobic Sentiment Classification and Spatio-Temporal Resource Allocation in South Africa", 0)
P("Implementation report: results on SA_XenoDataset_model_ready.csv (15,551 tweets, Jan 2017 to Jul 2022)", True)
doc.add_heading("1. Summary", 1)
for t in [
 f"All four lifecycle stages in the proposal were implemented end to end: cleaning checks, a dense vector index with lexicon, incident and tweet documents, time-series and virality features, KDE and lead-lag EDA, a RAG-style classifier benchmarked against baselines, and a Green/Amber/Red priority-zone model with an interactive dashboard.",
 f"Spatial finding: negative-tweet density and recorded attacks overlap strongly. {k['attack_mass_in_top10pct_neg_density']*100:.0f}% of recorded attack mass lies in the densest 10% of negative-tweet area (a random-town null gives {k['null_attack_mass_top10_mean']*100:.0f}%, permutation p = {k['perm_p_top10']:.3f}). Much of this is shared concentration in Johannesburg, Pretoria and Boksburg: across towns, negative-tweet share has almost no rank correlation with attacks (rho = {k['town_spearman_negshare_vs_attacks']:.2f}) while negative-tweet count does (rho = {k['town_spearman_negcount_vs_attacks']:.2f}).",
 "Temporal finding: the strongest relationship is simultaneous (lag 0). A lead of about 72 hours is only marginally above the permutation band (r = 0.065 vs 0.046), so the data do not support quoting a reliable operational lead time.",
 "Sentiment classification: every model is weak on the three-class task (best macro-F1 0.44). The retrieval-augmented hybrid (0.436) did not beat TF-IDF + logistic regression (0.442); the paired bootstrap difference is -0.007 (95% CI -0.013 to +0.001).",
 f"Zone model: classifying the current week's zone benefits clearly from sentiment, RAG and virality signals (alert PR-AUC {zn[zn.model.str.startswith('RF | B')].pr_auc_alert.iloc[0]:.2f} vs {zn[zn.model.str.startswith('RF | A')].pr_auc_alert.iloc[0]:.2f} for volume only; base rate 0.06). Forecasting next week's zone is much weaker: ranking is slightly better than persistence (alert PR-AUC 0.14 vs 0.09) but alert precision and lift at a 10% budget are lower than the simple persistence rule (lift 2.4 vs 3.5)."]: B(t)
doc.add_heading("2. Data and cleaning verification", 1)
P("The supplied file is already model-ready. Checks run in s01_features.py: no missing values in sentiment, location or attack fields; three sentiment classes only; coordinates inside South Africa. Imputation flags show how little spatial imputation remained: 7 rows with imputed coordinates, 1 province, 53 districts, 1 attack value.")
P("Counts differ from the proposal text. The proposal quotes Negative 9,080 / Neutral 4,339 / Positive 2,417 (15,836). This file holds 8,993 / 4,180 / 2,378 (15,551), so 285 records were removed in preparation (probably duplicates or empty text). The proposal should be updated to the final counts.")
P("Distribution notes: 89% of tweets are from Gauteng, 82% urban, 97% English; 348 tweets (2.2%) carry Number of Attacks > 0, forming 208 town-week incidents totalling 455 attacks. Attacks cluster in Feb 2017, Aug-Sep 2019, Nov 2021 and Feb-May 2022 (Operation Dudula period).")
doc.add_heading("3. Methods", 1)
doc.add_heading("3.1 Features (Structuring and Wrangling)", 2)
for t in ["Virality: log(1 + likes + 2 x retweets + 1.5 x replies), engagement and retweet ratio.",
 "Sentiment velocity at national, province and town scope: rolling 6-hour and 24-hour negative counts using only past data, plus velocity = negatives in the last 6 h relative to the average 6-hour rate over the preceding 18 h. A unit-conversion bug in the first version of the rolling windows was found by brute-force comparison and fixed.",
 "Spatial features: town centroid (median coordinates), distance to Johannesburg CBD, urban flag, province.",
 "Weekly town panel: 42 towns with at least 30 tweets, 3,608 town-weeks."]: B(t)
doc.add_heading("3.2 Vector index and RAG classifier", 2)
P("Embeddings are TF-IDF followed by 256-dimensional LSA, L2-normalised, searched by exact cosine similarity. The index holds 27 lexicon entries (derogatory terms, threat, expulsion and mobilisation language, grievance and scapegoating themes, local glossary), 208 incident records built from the attack data, and all tweets as labelled exemplars. For every test fold the retrieval corpus contains only training-fold tweets, and identical texts are grouped so duplicates never straddle train and test.")
P("The classifier uses retrieved context: similarity-weighted neighbour label distribution (k = 15), lexicon matches, nearest lexicon-entry similarity, and a lexicon hostility score, fed to a gradient-boosting meta-classifier. The hybrid variant adds out-of-fold TF-IDF logistic-regression probabilities. A hostility-severity score (0-3) is the lexicon score smoothed with retrieved neighbours' scores. It has no ground truth and has not been validated.")
doc.add_heading("3.3 Resource-priority zone model", 2)
P("Unit: town-week. Zone from the maximum recorded attack count: Green = 0, Amber = 1, Red = 2 or more. Two tasks: classify the same week (nowcast) and forecast the following week. Evaluation is walk-forward by year (train on earlier years, test 2019, 2020, 2021, 2022). Because attacks are rare, argmax classification collapses to all-Green, so zones are assigned by an alert budget: the top 3% of risk scores per year are Red and the next 7% Amber. Sentiment inputs are out-of-fold, so no label leakage reaches the zone model.")
doc.add_heading("4. Results", 1)
doc.add_heading("4.1 Sentiment benchmark (5-fold grouped CV, pooled out-of-fold)", 2)
T(sm, ["model", "accuracy", "macro_f1", "f1_Negative", "f1_Neutral", "f1_Positive"], ["Model", "Accuracy", "Macro-F1", "F1 Neg", "F1 Neu", "F1 Pos"])
P("* The gradient-boosting and dense-MLP rows are stand-ins for XGBoost and DistilBERT, which could not be installed (see section 5). The dense MLP reaches the highest accuracy by predicting almost only Negative, which is why its macro-F1 is low.")
doc.add_heading("4.2 Spatial KDE", 2)
doc.add_picture(str(FIG/"kde_gauteng.png"), width=Inches(6.5))
P(f"Fixed 4.5 km kernel over Gauteng ({k['gauteng_box_negative_tweets']:,} negative tweets, {k['gauteng_box_incidents']} incident-weeks). Density correlation rho = {k['spearman_kde_corr']:.2f}; Jaccard overlap of top-10% hotspots {k['jaccard_hotspots_top10']:.2f}.")
doc.add_heading("4.3 Lead-lag", 2)
doc.add_picture(str(FIG/"lead_lag.png"), width=Inches(6.5))
rows = [[r["name"], r["peak_lag_h"], round(r["peak_cc"], 3), r["best_lead_h"], round(r["best_lead_cc"], 3), round(r["band95"], 3)] for r in eda["leadlag"]]
t = doc.add_table(rows=1, cols=6); t.style = "Light Grid Accent 1"
for i, n in enumerate(["Series", "Peak lag (h)", "Peak r", "Best lead (h)", "r at lead", "95% band"]): t.rows[0].cells[i].text = n
for r in rows:
    c = t.add_row().cells
    for i, v in enumerate(r): c[i].text = str(v)
doc.add_paragraph()
P("Method: de-trended log counts (7-day centred rolling mean removed), cross-correlation over +/-5 days, 95% band from 200 circular-shift permutations. The attack field is attached to tweets by date and place, so the attack series reflects when tweets about incidents were posted rather than the minute an incident began; this limits what any lead-lag analysis can show.")
doc.add_heading("4.4 Zone model", 2)
P("Same-week classification (2019-2022 pooled, alert budget 10%):", True)
T(zn, ["model", "pr_auc_alert", "lift", "alert_recall", "recall_red", "macro_f1"], ["Model", "Alert PR-AUC", "Lift", "Alert recall", "Red recall", "Macro-F1"])
P("Next-week forecast:", True)
T(zf, ["model", "pr_auc_alert", "lift", "alert_recall", "recall_red", "macro_f1"], ["Model", "Alert PR-AUC", "Lift", "Alert recall", "Red recall", "Macro-F1"])
P("Per-year results for the best same-week model:", True)
T(py, ["year", "rows", "amber", "red", "alert_recall", "alert_precision"], ["Year", "Town-weeks", "Amber", "Red", "Alert recall", "Alert precision"])
P("Top permutation-importance features (2022, same-week model):", True)
T(imp, ["feature", "importance"], ["Feature", "Importance"], "{:.4f}")
doc.add_heading("5. Deviations from the proposal and limitations", 1)
for t in [
 "No LLM in the evaluated pipeline. The sandbox had no model access. The retrieval-conditioned classifier stands in for the LLM step; rag_llm.py builds the full RAG prompt (lexicon, similar labelled posts, recent local incidents) and accepts any LLM callable, but it was not run and no metrics are claimed for it.",
 "XGBoost, FAISS, ChromaDB and DistilBERT could not be installed (PyPI and Hugging Face were blocked). Substitutes: scikit-learn gradient boosting, exact cosine search over an LSA index, and a dense MLP. train_distilbert.py is provided to run on Colab with the same folds; it is untested.",
 "Spatial imputation was already done in the supplied file; no imputation was re-run.",
 "Sentiment labels: the ceiling of about 0.44 macro-F1 suggests noisy or hard-to-separate labels, especially Neutral vs Positive. Check how labels were assigned before drawing conclusions about RAG.",
 "Attacks are rare (166 non-Green of 3,608 town-weeks) and observed only through tweets, so a town with no tweets in a week counts as Green. This biases the next-week target.",
 "The lexicon is a seed list drawn from the data; it needs review by a South African domain expert, and the severity score needs human validation.",
 "Zone thresholds (Amber = 1, Red = 2+) and the 3%/7% alert budget are analyst choices to be agreed with emergency-management stakeholders."]: B(t)
doc.add_heading("6. Ethical considerations and policy guidance", 1)
for t in ["Treat outputs as triage signals for human review, never as grounds for action against individuals or communities. The tweets come from users, not perpetrators, and negative sentiment about foreigners is not itself a threat.",
 "Geographic aggregation at town level is deliberate; do not publish user-level locations. Remove user handles from any shared tweet text.",
 "Hostile discourse about migrants is common (58% of tweets are labelled Negative), so volume alone over-alerts. Use change against a town's own baseline plus mobilisation signals.",
 "Do not rely on a multi-day lead time based on these results. Same-day monitoring and pre-staging in the few persistently high-risk towns is better supported.",
 "Check performance by language and region before deployment; 97% of tweets are English and code-switched isiZulu/Sesotho text is likely misread."]: B(t)
doc.add_heading("7. Reproducing the results", 1)
P("Run in order from xeno_rag/src: s01_features.py, s03_sentiment_models.py (about 10 minutes), s04_eda.py, HORIZON=1 and HORIZON=0 s05_zone_model.py, s06_dashboard_data.py. Fixed seed 42 throughout.")
doc.save(OUT/"Xeno_RAG_Implementation_Report.docx"); print("saved")
