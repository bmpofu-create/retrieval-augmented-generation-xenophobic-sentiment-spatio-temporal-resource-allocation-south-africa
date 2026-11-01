"""Binary hostile / not-hostile sentiment. Same TF-IDF logistic regression as the
published winner, trained only on texts whose binary label does not conflict.

Out-of-fold P(hostile) is what the zone model consumes. Lexicon features are
reported as a second classifier and are not fed into that probability, so
mobilisation can be scored on its own later.
"""
import json
import time
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score, average_precision_score, precision_recall_fscore_support,
)
from common import *
from labels import with_binary_labels
from lexicon import lexicon_matrix

df = with_binary_labels(load_raw())
df = df[df.text_ml.str.strip().ne("") & ~df.binary_conflict].reset_index(drop=True)
df["grp"] = df["text_ml"].factorize()[0]
df["fold"] = -1
y = df["hostile"].to_numpy()
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
for k, (_, te) in enumerate(sgkf.split(df, y, df["grp"])):
    df.loc[df.index[te], "fold"] = k
texts = df["text_ml"].tolist()
LEX = lexicon_matrix(texts)
print("binary rows", len(df), "hostile rate", round(float(y.mean()), 3),
      "folds", df.fold.value_counts().sort_index().tolist(), flush=True)

names = ["Majority class", "TF-IDF + logistic regression", "TF-IDF + lexicon + logistic regression"]
oof = {m: np.zeros(len(df), dtype=int) for m in names}
prob = {m: np.zeros(len(df)) for m in names}

def score(yt, yp, pr):
    p, r, f, _ = precision_recall_fscore_support(yt, yp, labels=[0, 1], zero_division=0)
    _, _, fm, _ = precision_recall_fscore_support(yt, yp, average="macro", zero_division=0)
    return dict(accuracy=accuracy_score(yt, yp), macro_f1=fm, f1_nonhostile=f[0], f1_hostile=f[1],
                precision_hostile=p[1], recall_hostile=r[1],
                pr_auc_hostile=average_precision_score(yt, pr))

fold_rows = []
for k in range(5):
    t0 = time.time()
    tr = np.where(df.fold.to_numpy() != k)[0]
    te = np.where(df.fold.to_numpy() == k)[0]
    tf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=150_000)
    tc = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True, max_features=150_000)
    Xtr = sp.hstack([tf.fit_transform([texts[i] for i in tr]), tc.fit_transform([texts[i] for i in tr])]).tocsr()
    Xte = sp.hstack([tf.transform([texts[i] for i in te]), tc.transform([texts[i] for i in te])]).tocsr()
    lr = LogisticRegression(C=4.0, max_iter=400, class_weight="balanced").fit(Xtr, y[tr])
    P = lr.predict_proba(Xte)
    prob[names[1]][te] = P[:, list(lr.classes_).index(1)]
    oof[names[1]][te] = (prob[names[1]][te] >= 0.5).astype(int)
    both_tr = sp.hstack([Xtr, LEX[tr]]).tocsr()
    both_te = sp.hstack([Xte, LEX[te]]).tocsr()
    lr2 = LogisticRegression(C=4.0, max_iter=400, class_weight="balanced").fit(both_tr, y[tr])
    P2 = lr2.predict_proba(both_te)
    prob[names[2]][te] = P2[:, list(lr2.classes_).index(1)]
    oof[names[2]][te] = (prob[names[2]][te] >= 0.5).astype(int)
    dum = DummyClassifier(strategy="most_frequent").fit(Xtr, y[tr])
    oof[names[0]][te] = dum.predict(Xte)
    prob[names[0]][te] = dum.predict_proba(Xte)[:, list(dum.classes_).index(1)]
    for mn in names:
        fold_rows.append(dict(model=mn, fold=k, **score(y[te], oof[mn][te], prob[mn][te])))
    print(f"fold {k} {time.time() - t0:.0f}s  tfidf macro-F1 {fold_rows[-2]['macro_f1']:.3f}", flush=True)

fm = pd.DataFrame(fold_rows)
fm.to_csv(OUT / "binary_sentiment_fold_metrics.csv", index=False)
rows = []
for mn in names:
    pooled = score(y, oof[mn], prob[mn])
    sub = fm[fm.model == mn]
    rows.append(dict(model=mn, **pooled, macro_f1_fold_sd=float(sub.macro_f1.std())))
res = pd.DataFrame(rows).round(4)
res.to_csv(OUT / "binary_sentiment_comparison.csv", index=False)
print(res.to_string(index=False), flush=True)

rng = np.random.default_rng(SEED)
diffs = []
for _ in range(400):
    s = rng.integers(0, len(df), len(df))
    diffs.append(score(y[s], oof[names[2]][s], prob[names[2]][s])["macro_f1"]
                 - score(y[s], oof[names[1]][s], prob[names[1]][s])["macro_f1"])
boot = {"lexicon_minus_tfidf_macroF1_mean": float(np.mean(diffs)),
        "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]}
json.dump(boot, open(OUT / "bootstrap_binary_lexicon_vs_tfidf.json", "w"))
print("lexicon minus tfidf macro-F1", round(boot["lexicon_minus_tfidf_macroF1_mean"], 4), boot["ci95"], flush=True)

out = pd.DataFrame({
    "raw_id": df.raw_id,
    "hostile": y,
    "p_hostile": prob[names[1]],
    "pred": oof[names[1]],
    "p_hostile_lex": prob[names[2]],
})
out.to_csv(OUT / "binary_oof_predictions.csv.gz", index=False)
print("wrote", len(out), "oof rows", flush=True)
