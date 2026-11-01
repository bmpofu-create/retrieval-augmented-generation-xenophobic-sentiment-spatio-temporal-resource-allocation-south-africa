"""Stage 4a - sentiment classification: baselines vs retrieval-augmented (RAG) classifier.
5-fold StratifiedGroupKFold (identical texts grouped). All metrics are out-of-fold (OOF).
The retrieval corpus for a test tweet contains ONLY training-fold tweets (no leakage)."""
import sys, json, time, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
from common import *
from lexicon import lexicon_matrix, severity_from_matrix, ENTRY_KEYS
from rag_index import Embedder, VectorIndex, lexicon_docs, incident_docs

K = 15
df = add_groups_and_folds(load_raw())
y = df["Sentiment"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
texts = df["text_ml"].tolist(); n = len(df)
LEX = lexicon_matrix(texts); sev_own, sev_raw = severity_from_matrix(LEX)
lex_df, inc_df = lexicon_docs(), incident_docs(df)
print("n", n, "folds", df.fold.value_counts().sort_index().tolist(), flush=True)

def retr_feats(index, Q, tweet_mask, tweet_y_all, grp_q=None):
    I, S = index.search_vec(Q, k=K, mask=tweet_mask, exclude_groups=grp_q)
    lab = np.where(I >= 0, tweet_y_all[np.clip(I, 0, None)], -1)
    W = np.maximum(S, 0) ** 3
    wd = np.stack([(W * (lab == c)).sum(1) for c in range(3)], 1); wd = wd / (wd.sum(1, keepdims=True) + 1e-9)
    t5 = np.stack([((lab[:, :5] == c)).mean(1) for c in range(3)], 1)
    li, ls = index.search_vec(Q, k=3, mask=(index.meta["doc_type"] == "lexicon").to_numpy())
    return I, S, np.column_stack([wd, t5, S[:, 0], S[:, :5].mean(1), ls[:, 0], ls.mean(1)]), wd

def metrics(yt, yp):
    p, r, f, _ = precision_recall_fscore_support(yt, yp, labels=[0, 1, 2], zero_division=0)
    pm, rm, fm, _ = precision_recall_fscore_support(yt, yp, average="macro", zero_division=0)
    _, _, fw, _ = precision_recall_fscore_support(yt, yp, average="weighted", zero_division=0)
    return dict(accuracy=accuracy_score(yt, yp), macro_precision=pm, macro_recall=rm, macro_f1=fm, weighted_f1=fw,
                **{f"f1_{c}": f[i] for i, c in enumerate(CLASSES)})

MODELS = ["Majority class", "TF-IDF + Logistic Regression", "TF-IDF(LSA) + GradientBoosting [XGBoost stand-in]",
          "LSA dense MLP [DistilBERT stand-in]", "RAG: retrieval kNN vote", "RAG: retrieval+lexicon meta-classifier", "RAG hybrid: retrieval+lexicon+TF-IDF stack"]
OOF = {m: np.zeros(n, int) for m in MODELS}; PROB = {m: np.zeros((n, 3)) for m in MODELS}
sev_final = np.zeros(n); nn_rows = []; fold_metrics = []

for k in range(5):
    t0 = time.time(); tr = np.where(df.fold != k)[0]; te = np.where(df.fold == k)[0]
    emb = Embedder().fit(texts_tr := [texts[i] for i in tr] + lex_df["text"].tolist() + inc_df["text"].tolist())
    ix = VectorIndex(emb)
    lex_meta = lex_df.assign(grp=-1)[["doc_type", "key", "grp"]]
    tw_meta = pd.DataFrame({"doc_type": "tweet", "key": "", "grp": df["grp"].to_numpy()[tr]})
    ix.add(lex_df["text"].tolist(), lex_meta); ix.add([texts[i] for i in tr], tw_meta)
    # map index doc position -> global tweet row
    pos2row = np.full(len(ix.meta), -1); pos2row[len(lex_df):] = tr
    tmask = (ix.meta["doc_type"] == "tweet").to_numpy()
    y_pos = np.full(len(ix.meta), -1); y_pos[len(lex_df):] = y[tr]
    Qtr, Qte = ix.V[len(lex_df):], emb.transform([texts[i] for i in te])
    _, _, Ftr, _ = retr_feats(ix, Qtr, tmask, y_pos, df["grp"].to_numpy()[tr])
    Ite, Ste, Fte, wd_te = retr_feats(ix, Qte, tmask, y_pos)
    Xtr_r = np.column_stack([Ftr, LEX[tr], sev_raw[tr]]); Xte_r = np.column_stack([Fte, LEX[te], sev_raw[te]])
    # --- RAG meta-classifier
    m = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.06, max_leaf_nodes=15, l2_regularization=1.0,
                                       class_weight="balanced", random_state=SEED).fit(Xtr_r, y[tr])
    P = m.predict_proba(Xte_r); name = MODELS[5]; PROB[name][te] = P; OOF[name][te] = P.argmax(1)
    name = MODELS[4]; PROB[name][te] = wd_te; OOF[name][te] = wd_te.argmax(1)
    # severity: own lexicon score smoothed with retrieved neighbours' scores
    nb = np.where(Ite[:, :5] >= 0, sev_own[pos2row[np.clip(Ite[:, :5], 0, None)]], 0).mean(1)
    sev_final[te] = 0.7 * sev_own[te] + 0.3 * nb
    # --- baselines
    tf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=150_000)
    tc = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True, max_features=150_000)
    Xa = sp.hstack([tf.fit_transform([texts[i] for i in tr]), tc.fit_transform([texts[i] for i in tr])]).tocsr()
    Xb = sp.hstack([tf.transform([texts[i] for i in te]), tc.transform([texts[i] for i in te])]).tocsr()
    lr = LogisticRegression(C=4.0, max_iter=400, class_weight="balanced").fit(Xa, y[tr])
    P = lr.predict_proba(Xb); P_lr = P; name = MODELS[1]; PROB[name][te] = P; OOF[name][te] = P.argmax(1)
    inner = np.zeros((len(tr), 3))
    for itr, ite in StratifiedGroupKFold(3, shuffle=True, random_state=SEED).split(tr, y[tr], df["grp"].to_numpy()[tr]):
        inner[ite] = LogisticRegression(C=4.0, max_iter=300, class_weight="balanced").fit(Xa[itr], y[tr][itr]).predict_proba(Xa[ite])
    hy = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=8, l2_regularization=2.0,
                                        class_weight="balanced", random_state=SEED).fit(np.column_stack([Xtr_r, inner]), y[tr])
    P = hy.predict_proba(np.column_stack([Xte_r, P_lr])); name = MODELS[6]; PROB[name][te] = P; OOF[name][te] = P.argmax(1)
    Ltr, Lte = np.column_stack([Qtr, LEX[tr]]), np.column_stack([Qte, LEX[te]])
    gb = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.08, max_leaf_nodes=15, class_weight="balanced",
                                        l2_regularization=1.0, random_state=SEED).fit(Ltr, y[tr])
    P = gb.predict_proba(Lte); name = MODELS[2]; PROB[name][te] = P; OOF[name][te] = P.argmax(1)
    mlp = MLPClassifier((256, 64), alpha=1e-3, early_stopping=True, max_iter=80, random_state=SEED).fit(Qtr, y[tr])
    P = mlp.predict_proba(Qte); name = MODELS[3]; PROB[name][te] = P; OOF[name][te] = P.argmax(1)
    dm = DummyClassifier(strategy="most_frequent").fit(Qtr, y[tr]); OOF[MODELS[0]][te] = dm.predict(Qte)
    PROB[MODELS[0]][te] = dm.predict_proba(Qte)
    for mn in MODELS: fold_metrics.append(dict(model=mn, fold=k, **metrics(y[te], OOF[mn][te])))
    print(f"fold {k} done {time.time()-t0:.0f}s | RAG meta F1 {fold_metrics[-1]['macro_f1']:.3f}", flush=True)

fm = pd.DataFrame(fold_metrics); fm.to_csv(OUT / "sentiment_fold_metrics.csv", index=False)
rows = []
for mn in MODELS:
    pooled = metrics(y, OOF[mn]); f = fm[fm.model == mn]
    rows.append(dict(model=mn, **pooled, macro_f1_fold_sd=f.macro_f1.std(), accuracy_fold_sd=f.accuracy.std()))
res = pd.DataFrame(rows).round(4); res.to_csv(OUT / "sentiment_model_comparison.csv", index=False)
print(res.to_string(), flush=True)
best = res.sort_values("macro_f1").iloc[-1]["model"]
cm = {mn: confusion_matrix(y, OOF[mn], labels=[0, 1, 2]).tolist() for mn in MODELS}
json.dump(cm, open(OUT / "sentiment_confusions.json", "w"))
# paired bootstrap: RAG meta vs TF-IDF LR (macro-F1 difference)
rng = np.random.default_rng(SEED); a, b = OOF[MODELS[6]], OOF[MODELS[1]]; diffs = []
for _ in range(400):
    s = rng.integers(0, n, n)
    diffs.append(precision_recall_fscore_support(y[s], a[s], average="macro", zero_division=0)[2] -
                 precision_recall_fscore_support(y[s], b[s], average="macro", zero_division=0)[2])
json.dump({"hybrid_minus_tfidf_lr_macroF1_mean": float(np.mean(diffs)), "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]},
          open(OUT / "bootstrap_rag_vs_tfidf.json", "w"))
# per-tweet OOF outputs consumed by the zone model / dashboard
rag = PROB[MODELS[6]]
out = pd.DataFrame({"raw_id": df.raw_id, "y_true": df.Sentiment, "rag_pred": [CLASSES[i] for i in OOF[MODELS[6]]],
                    "tfidf_lr_pred": [CLASSES[i] for i in OOF[MODELS[1]]],
                    "rag_p_neg": rag[:, 0], "rag_p_neu": rag[:, 1], "rag_p_pos": rag[:, 2],
                    "lr_p_neg": PROB[MODELS[1]][:, 0], "lex_severity_raw": sev_raw, "severity_score": sev_final})
out["hostility"] = out.rag_p_neg * out.severity_score
out["severity_level"] = pd.cut(out.severity_score, [-1, 0.0001, 0.2, 0.45, 2], labels=[0, 1, 2, 3]).astype(int)
out.to_csv(OUT / "sentiment_oof_predictions.csv.gz", index=False)
print("severity level dist (all):", out.severity_level.value_counts().sort_index().to_dict(), flush=True)

# ---- fit deployable artefacts on ALL data (index for retrieval / LLM prompting)
emb = Embedder().fit(texts + lex_df["text"].tolist() + inc_df["text"].tolist()); ix = VectorIndex(emb)
ix.add(lex_df["text"].tolist(), lex_df.assign(grp=-1)[["doc_type", "key", "category", "weight", "text", "grp"]])
ix.add(inc_df["text"].tolist(), inc_df.assign(grp=-1))
ix.add(df["CleanedTweet"].tolist(), pd.DataFrame({"doc_type": "tweet", "key": "", "grp": df["grp"], "text": df["CleanedTweet"],
                                                   "label": df["Sentiment"], "town": df["Tweet Origin"], "ts": df["dt"]}))
ix.save(OUT / "rag_vector_index.pkl"); print("index docs:", ix.meta.doc_type.value_counts().to_dict(), "faiss:", __import__("rag_index").HAVE_FAISS)
