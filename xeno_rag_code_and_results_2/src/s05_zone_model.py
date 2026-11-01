"""Stage 4b - Resource Priority Zone model (Green / Amber / Red) per town-week, 1 week ahead.
Walk-forward by calendar year (train on all earlier years; test 2019, 2020, 2021, 2022)."""
import json, os, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import precision_recall_fscore_support, balanced_accuracy_score, average_precision_score, confusion_matrix
from sklearn.inspection import permutation_importance
from common import *

f = pd.read_csv(OUT / "features_tweet.csv.gz", parse_dates=["dt"])
s = pd.read_csv(OUT / "sentiment_oof_predictions.csv.gz")
d = f.merge(s[["raw_id", "rag_p_neg", "severity_score", "severity_level", "hostility"]], on="raw_id")
H = int(os.environ.get("HORIZON", "1"))  # 1 = forecast next week, 0 = nowcast same week
SUF = "" if H == 1 else "_nowcast"
MIN_TWEETS = 30
towns = d["Tweet Origin"].value_counts(); keep = towns[towns >= MIN_TWEETS].index
d = d[d["Tweet Origin"].isin(keep)].copy()
d["week"] = d["dt"].dt.to_period("W").dt.start_time
d["hi_sev"] = (d.severity_level >= 2).astype(int)
g = d.groupby(["Tweet Origin", "week"])
W = g.agg(n=("raw_id", "size"), neg=("is_neg", "sum"), p_neg_mean=("rag_p_neg", "mean"), rag_neg_expected=("rag_p_neg", "sum"),
          hostility_sum=("hostility", "sum"), hostility_mean=("hostility", "mean"), hi_sev=("hi_sev", "sum"), sev_mean=("severity_score", "mean"),
          likes=("Likes", "sum"), retweets=("Retweets", "sum"), replies=("No of replies", "sum"), viral_sum=("virality", "sum"), viral_max=("virality", "max"),
          vel_max=("neg_velocity_town", "max"), neg6_max=("neg_6h_town", "max"), neg24_max=("neg_24h_town", "max"),
          lat=("cent_lat", "first"), lon=("cent_lon", "first"), dist_jhb=("dist_jhb_km", "first"), urban=("urban", "mean"),
          prov=("Province", "first"), att=("Number of Attacks", "max")).reset_index()
W["neg_share"] = W.neg / W.n
W["zone_now"] = np.select([W.att >= 2, W.att == 1], [2, 1], 0)
# province-week and national-week pressure
pw = d.groupby(["Province", "week"]).agg(prov_neg=("is_neg", "sum"), prov_att=("att_flag", "sum")).reset_index().rename(columns={"Province": "prov"})
nw = d.groupby("week").agg(nat_neg=("is_neg", "sum")).reset_index()
W = W.merge(pw, on=["prov", "week"]).merge(nw, on="week")
W["prov_code"] = W.prov.astype("category").cat.codes
# lags within town (previous observed calendar week)
W = W.sort_values(["Tweet Origin", "week"]).reset_index(drop=True)
def shifted(col, k=1):
    t = W[["Tweet Origin", "week", col]].copy(); t["week"] += pd.Timedelta(weeks=k)
    return W[["Tweet Origin", "week"]].merge(t, on=["Tweet Origin", "week"], how="left")[col].to_numpy()
W["zone_prev1"] = np.nan_to_num(shifted("zone_now", 1)); W["zone_prev2"] = np.nan_to_num(shifted("zone_now", 2))
W["neg_prev1"] = np.nan_to_num(shifted("neg", 1)); W["neg_delta"] = W.neg - W.neg_prev1
W["share_prev1"] = np.nan_to_num(shifted("neg_share", 1)); W["share_delta"] = W.neg_share - W.share_prev1
W["host_prev1"] = np.nan_to_num(shifted("hostility_sum", 1)); W["host_delta"] = W.hostility_sum - W.host_prev1
# target: zone of the SAME town in week+1 (0 if the town has no recorded tweets that week)
nxt = W[["Tweet Origin", "week", "zone_now"]].copy(); nxt["week"] -= pd.Timedelta(weeks=1)
W["y"] = (W[["Tweet Origin", "week"]].merge(nxt, on=["Tweet Origin", "week"], how="left")["zone_now"].fillna(0).astype(int).to_numpy() if H == 1 else W.zone_now.to_numpy())
W["year"] = W.week.dt.year
W = W[W.week < W.week.max()] if H == 1 else W  # last week has no observable next week
print("town-weeks", len(W), "towns", W["Tweet Origin"].nunique(), "target dist", W.y.value_counts().sort_index().to_dict(), flush=True)
print(W.groupby("year").y.value_counts().unstack(fill_value=0))

VOL = ["n", "likes", "retweets", "replies"]
SENT = VOL + ["neg", "neg_share", "p_neg_mean", "rag_neg_expected", "hostility_sum", "hostility_mean", "hi_sev", "sev_mean", "viral_sum", "viral_max",
              "vel_max", "neg6_max", "neg24_max", "neg_delta", "share_delta", "host_delta", "prov_neg", "nat_neg", "lat", "lon", "dist_jhb", "urban", "prov_code"]
HIST = SENT + (["zone_now", "zone_prev1", "zone_prev2", "prov_att"] if H == 1 else ["zone_prev1", "zone_prev2"])
SETS = {"A: volume only": VOL, "B: sentiment+RAG+virality+spatial": SENT, "C: B + attack history": HIST}
ZN = ["Green", "Amber", "Red"]
def mk(kind):
    if kind == "RF": return RandomForestClassifier(400, min_samples_leaf=3, class_weight="balanced_subsample", n_jobs=2, random_state=SEED)
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=8, l2_regularization=2.0, class_weight="balanced", random_state=SEED)

TEST_YEARS = [2019, 2020, 2021, 2022]; rows = []; preds = {}
for kind in ("RF", "GB"):
    for sn, cols in SETS.items():
        pr = np.full((len(W), 3), np.nan)
        for ty in TEST_YEARS:
            tr = (W.year < ty).to_numpy(); te = (W.year == ty).to_numpy()
            if W.y[tr].nunique() < 3: tr_y = W.y[tr]
            m = mk(kind).fit(W.loc[tr, cols], W.y[tr]); P = np.zeros((te.sum(), 3)); P[:, m.classes_] = m.predict_proba(W.loc[te, cols]); pr[te] = P
        preds[(kind, sn)] = pr
# baselines
base = {("Persistence (this week's zone)" if H == 1 else "Last week's zone"): (W.zone_now if H == 1 else W.zone_prev1).to_numpy().astype(int), "Always Green": np.zeros(len(W), int)}
tm = W.year.isin(TEST_YEARS).to_numpy(); yt = W.y.to_numpy()
def budget_zones(score, ranked=True):
    z = np.zeros(len(W), int)
    for ty in TEST_YEARS:
        m = (W.year == ty).to_numpy(); sc = score[m]; r = pd.Series(sc).rank(method="first", ascending=False).to_numpy() / m.sum()
        z[m] = np.where(r <= 0.03, 2, np.where(r <= 0.10, 1, 0))
    return z
def ev(name, yp, score):
    p, r, f1, _ = precision_recall_fscore_support(yt[tm], yp[tm], labels=[0, 1, 2], zero_division=0)
    pm, rm, fm, _ = precision_recall_fscore_support(yt[tm], yp[tm], average="macro", zero_division=0)
    ap = average_precision_score((yt[tm] > 0).astype(int), score[tm]) if score is not None else np.nan
    # alert metrics (Amber+Red as positive)
    a_t = yt[tm] > 0; a_p = yp[tm] > 0
    return dict(model=name, macro_f1=fm, macro_recall=rm, balanced_acc=balanced_accuracy_score(yt[tm], yp[tm]), recall_amber=r[1], recall_red=r[2],
                precision_amber=p[1], precision_red=p[2], alert_recall=(a_t & a_p).sum() / max(a_t.sum(), 1), alert_precision=(a_t & a_p).sum() / max(a_p.sum(), 1),
                alert_rate=a_p[:].mean(), pr_auc_alert=ap, base_rate=a_t.mean(), lift=((a_t & a_p).sum() / max(a_p.sum(), 1)) / a_t.mean())
for k, v in base.items(): rows.append(ev(k, v, v.astype(float)))
for (kind, sn), pr in preds.items():
    sc = np.nan_to_num(pr[:, 1:].sum(1)) + 1e-6 * np.nan_to_num(pr[:, 2]); preds[(kind, sn)] = pr
    rows.append(ev(f"{kind} | {sn}", budget_zones(sc), sc))
res = pd.DataFrame(rows).round(3); res.to_csv(OUT / f"zone_model_comparison{SUF}.csv", index=False); print(res.to_string(), flush=True)
# per-year for the best full model
best = res[res.model.str.startswith(("RF", "GB"))].sort_values("macro_f1").iloc[-1].model; print("best:", best)
kind, sn = best.split(" | "); pr = preds[(kind, sn)]; yp = budget_zones(np.nan_to_num(pr[:, 1:].sum(1)) + 1e-6 * np.nan_to_num(pr[:, 2]))
per_year = []
for ty in TEST_YEARS:
    m = (W.year == ty).to_numpy(); a_t = yt[m] > 0; a_p = yp[m] > 0
    per_year.append(dict(year=ty, rows=int(m.sum()), amber=int((yt[m] == 1).sum()), red=int((yt[m] == 2).sum()), alert_recall=float((a_t & a_p).sum() / max(a_t.sum(), 1)), alert_precision=float((a_t & a_p).sum() / max(a_p.sum(), 1))))
pd.DataFrame(per_year).round(3).to_csv(OUT / f"zone_model_per_year{SUF}.csv", index=False)
cm = confusion_matrix(yt[tm], yp[tm], labels=[0, 1, 2]).tolist(); json.dump({"best": best, "confusion": cm}, open(OUT / f"zone_confusion{SUF}.json", "w"))
# permutation importance (last test year, model refit on <2022)
cols = SETS[sn]; tr = (W.year < 2022).to_numpy(); te = (W.year == 2022).to_numpy()
m = mk(kind).fit(W.loc[tr, cols], W.y[tr])
pi = permutation_importance(m, W.loc[te, cols], W.y[te], scoring="balanced_accuracy", n_repeats=15, random_state=SEED)
imp = pd.DataFrame({"feature": cols, "importance": pi.importances_mean, "sd": pi.importances_std}).sort_values("importance", ascending=False)
imp.to_csv(OUT / f"zone_feature_importance{SUF}.csv", index=False); print(imp.head(12).to_string(), flush=True)
# export predictions for dashboard
W["pred_zone"] = np.where(tm, yp, -1); W["p_amber"] = np.nan_to_num(pr[:, 1]); W["p_red"] = np.nan_to_num(pr[:, 2]); W["p_alert"] = W.p_amber + W.p_red
W.rename(columns={"Tweet Origin": "town"})[["town", "week", "prov", "lat", "lon", "n", "neg", "neg_share", "hostility_sum", "hi_sev", "zone_now", "y", "pred_zone", "p_amber", "p_red", "p_alert"]].to_csv(OUT / f"zone_predictions_town_week{SUF}.csv", index=False)
json.dump({"n_town_weeks": int(len(W)), "n_towns": int(W['Tweet Origin'].nunique()), "test_rows": int(tm.sum()), "test_dist": {ZN[i]: int((yt[tm] == i).sum()) for i in range(3)}}, open(OUT / f"zone_meta{SUF}.json", "w"))
