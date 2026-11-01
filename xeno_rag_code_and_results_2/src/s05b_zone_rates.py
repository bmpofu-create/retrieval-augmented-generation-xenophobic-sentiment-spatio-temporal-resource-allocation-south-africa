"""Same-week and next-week zone models scored against each town's own past.

Volume sums made the published model flag busy towns. This script replaces
those sums with past-only rates and z-scores, and keeps mobilisation,
threat, expulsion and derogatory lexicon hits as their own rates.

An optional incidents_independent.csv (date, town, attacks) replaces the
tweet-attached attack target when it is present. Without that file the
target is still Number of Attacks on the tweets.

Outputs do not overwrite the published zone_model_comparison*.csv files.
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score, precision_recall_fscore_support,
)
from sklearn.inspection import permutation_importance
from common import *
from lexicon import category_frame

MIN_TWEETS = 30
TEST_YEARS = [2019, 2020, 2021, 2022]
ZN = ["Green", "Amber", "Red"]
INCIDENTS = ROOT / "incidents_independent.csv"


def add_past(frame, group, cols):
    """Z-score, ratio and delta using strictly earlier rows inside each group."""
    frame = frame.sort_values([group, "week"]).reset_index(drop=True)
    pieces = []
    for _, sub in frame.groupby(group, sort=False):
        sub = sub.copy()
        for col in cols:
            past = sub[col].shift(1)
            mu = past.expanding(min_periods=2).mean()
            sd = past.expanding(min_periods=2).std()
            z = (sub[col] - mu) / sd.where(sd > 1e-8)
            ratio = sub[col] / mu.where(mu.abs() > 1e-8)
            sub[f"{col}_z"] = z.replace([np.inf, -np.inf], np.nan)
            sub[f"{col}_ratio"] = ratio.replace([np.inf, -np.inf], np.nan)
            sub[f"{col}_delta"] = sub[col] - past
        pieces.append(sub)
    out = pd.concat(pieces, ignore_index=True)
    zcols = [c for c in out.columns if c.endswith("_z") or c.endswith("_delta")]
    rcols = [c for c in out.columns if c.endswith("_ratio")]
    out[zcols] = out[zcols].fillna(0.0)
    out[rcols] = out[rcols].fillna(1.0)
    return out


def assert_past_only(frame, value_col, ratio_col, group):
    town = frame[group].value_counts().idxmax()
    sub = frame[frame[group] == town].sort_values("week")
    row = sub.iloc[8]
    past = sub.loc[sub.week < row.week, value_col]
    if len(past) < 2 or past.mean() == 0:
        return
    mu = float(row[value_col]) / float(row[ratio_col])
    if abs(mu - float(past.mean())) > 1e-4:
        raise AssertionError(f"baseline for {value_col} used a value outside earlier weeks: {mu} vs {past.mean()}")


def budget_zones(years, score):
    z = np.zeros(len(score), dtype=int)
    for ty in TEST_YEARS:
        m = years == ty
        if m.sum() == 0:
            continue
        rank = pd.Series(score[m]).rank(method="first", ascending=False).to_numpy() / m.sum()
        z[m] = np.where(rank <= 0.03, 2, np.where(rank <= 0.10, 1, 0))
    return z


def evaluate(name, years, yt, yp, score, mask):
    y_true, y_pred, sc = yt[mask], yp[mask], score[mask]
    p, r, _, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2], zero_division=0)
    _, rm, fm, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    alert_true = y_true > 0
    alert_pred = y_pred > 0
    precision = (alert_true & alert_pred).sum() / max(alert_pred.sum(), 1)
    base = alert_true.mean()
    return dict(
        model=name, macro_f1=fm, macro_recall=rm,
        balanced_acc=balanced_accuracy_score(y_true, y_pred),
        recall_amber=r[1], recall_red=r[2], precision_amber=p[1], precision_red=p[2],
        alert_recall=(alert_true & alert_pred).sum() / max(alert_true.sum(), 1),
        alert_precision=precision, alert_rate=alert_pred.mean(),
        pr_auc_alert=average_precision_score(alert_true.astype(int), sc),
        base_rate=base, lift=precision / base if base else np.nan,
    )


def fit_predict(kind, Xtr, ytr, Xte):
    if kind == "RF":
        model = RandomForestClassifier(
            400, min_samples_leaf=3, class_weight="balanced_subsample", n_jobs=1, random_state=SEED)
    else:
        model = HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.05, max_leaf_nodes=8, l2_regularization=2.0,
            class_weight="balanced", random_state=SEED)
    model.fit(Xtr, ytr)
    P = np.zeros((len(Xte), 3))
    P[:, model.classes_] = model.predict_proba(Xte)
    return model, P


def alert_scorer(est, X, y):
    P = np.zeros((len(X), 3))
    P[:, est.classes_] = est.predict_proba(X)
    return average_precision_score((np.asarray(y) > 0).astype(int), P[:, 1] + P[:, 2])


def external_attacks(weeks, towns):
    if not INCIDENTS.exists():
        return None, "not provided; zone target is still the tweet-attached Number of Attacks field"
    inc = pd.read_csv(INCIDENTS, parse_dates=["date"])
    need = {"date", "town", "attacks"}
    if not need.issubset(inc.columns):
        raise SystemExit(f"{INCIDENTS.name} must have columns date, town, attacks")
    inc["week"] = inc["date"].dt.to_period("W").dt.start_time
    g = inc.groupby(["town", "week"], as_index=False)["attacks"].sum()
    key = pd.DataFrame({"town": towns, "week": weeks})
    merged = key.merge(g, on=["town", "week"], how="left")
    return merged["attacks"].fillna(0).to_numpy(), f"loaded {INCIDENTS.name} ({len(g)} town-weeks)"


feat = pd.read_csv(OUT / "features_tweet.csv.gz", parse_dates=["dt"])
oof = pd.read_csv(OUT / "binary_oof_predictions.csv.gz")
feat["raw_id"] = feat["raw_id"].astype(int)
oof["raw_id"] = oof["raw_id"].astype(int)
keep_towns = feat["Tweet Origin"].value_counts()
keep_towns = keep_towns[keep_towns >= MIN_TWEETS].index
feat = feat[feat["Tweet Origin"].isin(keep_towns)].copy()
d = feat.merge(oof[["raw_id", "p_hostile"]], on="raw_id", how="inner")
lex = category_frame(d["text_ml"].fillna("").astype(str).tolist())
d = pd.concat([d.reset_index(drop=True), lex], axis=1)
for col, flag in [("mobilisation", "is_mob"), ("threat", "is_threat"), ("expulsion", "is_expel"), ("derogatory", "is_derog")]:
    d[flag] = (d[col] > 0).astype(int)
d["hostility"] = d["p_hostile"] * d["severity"]
d["hi_sev"] = (d["severity"] > 0.2).astype(int)

d["week"] = d["dt"].dt.to_period("W").dt.start_time
g = d.groupby(["Tweet Origin", "week"], sort=False)
W = g.agg(
    n=("raw_id", "size"),
    likes=("Likes", "sum"), retweets=("Retweets", "sum"), replies=("No of replies", "sum"),
    neg=("is_neg", "sum"),
    p_hostile_mean=("p_hostile", "mean"), p_hostile_sum=("p_hostile", "sum"),
    hostility_sum=("hostility", "sum"), hostility_mean=("hostility", "mean"), hi_sev=("hi_sev", "sum"),
    mob_rate=("is_mob", "mean"), threat_rate=("is_threat", "mean"),
    expel_rate=("is_expel", "mean"), derog_rate=("is_derog", "mean"),
    sev_mean=("severity", "mean"), viral_sum=("virality", "sum"), viral_max=("virality", "max"),
    vel_max=("neg_velocity_town", "max"), neg6_max=("neg_6h_town", "max"), neg24_max=("neg_24h_town", "max"),
    lat=("cent_lat", "first"), lon=("cent_lon", "first"),
    dist_jhb=("dist_jhb_km", "first"), urban=("urban", "mean"),
    prov=("Province", "first"), att_tweet=("Number of Attacks", "max"),
).reset_index()
W["neg_share"] = W["neg"] / W["n"]
W["viral_mean"] = W["viral_sum"] / W["n"]
W["log_n"] = np.log1p(W["n"])
ext, incident_note = external_attacks(W["week"], W["Tweet Origin"])
W["attacks"] = ext if ext is not None else W["att_tweet"].to_numpy()
W["zone_now"] = np.select([W.attacks >= 2, W.attacks == 1], [2, 1], 0)
W = W.rename(columns={"Tweet Origin": "town"})
W["prov_code"] = W.prov.astype("category").cat.codes
W = add_past(W, "town", ["n", "p_hostile_mean", "mob_rate", "threat_rate", "expel_rate", "derog_rate", "sev_mean", "viral_mean"])
W["has_baseline"] = (W.groupby("town").cumcount() >= 2).astype(int)
assert_past_only(W, "n", "n_ratio", "town")

pw = W.groupby(["prov", "week"], as_index=False)["p_hostile_sum"].sum().rename(columns={"p_hostile_sum": "prov_hostile"})
pw = add_past(pw, "prov", ["prov_hostile"])
nw = W.groupby("week", as_index=False)["p_hostile_sum"].sum().rename(columns={"p_hostile_sum": "nat_hostile"})
nw["prov"] = "national"
nw = add_past(nw, "prov", ["nat_hostile"])
W = W.merge(pw[["prov", "week", "prov_hostile_z"]], on=["prov", "week"], how="left")
W = W.merge(nw[["week", "nat_hostile_z"]], on="week", how="left")
W[["prov_hostile_z", "nat_hostile_z"]] = W[["prov_hostile_z", "nat_hostile_z"]].fillna(0.0)

# Previous observed zone for this town, used only when forecasting the next week.
W = W.sort_values(["town", "week"]).reset_index(drop=True)
prev = W[["town", "week", "zone_now"]].copy()
prev["week"] = prev["week"] + pd.Timedelta(weeks=1)
W["zone_prev"] = W[["town", "week"]].merge(prev, on=["town", "week"], how="left")["zone_now"].fillna(0).to_numpy()
nxt = W[["town", "week", "zone_now"]].copy()
nxt["week"] = nxt["week"] - pd.Timedelta(weeks=1)
W["zone_next"] = W[["town", "week"]].merge(nxt, on=["town", "week"], how="left")["zone_now"].to_numpy()

def earlier_week(col):
    t = W[["town", "week", col]].copy()
    t["week"] = t["week"] + pd.Timedelta(weeks=1)
    return W[["town", "week"]].merge(t, on=["town", "week"], how="left")[col].fillna(0).to_numpy()

W["neg_delta"] = W["neg"] - earlier_week("neg")
W["share_delta"] = W["neg_share"] - earlier_week("neg_share")
W["host_delta"] = W["hostility_sum"] - earlier_week("hostility_sum")
prov_neg = W.groupby(["prov", "week"], as_index=False)["neg"].sum().rename(columns={"neg": "prov_neg"})
nat_neg = W.groupby("week", as_index=False)["neg"].sum().rename(columns={"neg": "nat_neg"})
W = W.merge(prov_neg, on=["prov", "week"], how="left").merge(nat_neg, on="week", how="left")

VOL = ["n", "likes", "retweets", "replies"]
SUMS = VOL + ["neg", "neg_share", "p_hostile_mean", "p_hostile_sum", "hostility_sum", "hostility_mean",
              "hi_sev", "sev_mean", "viral_sum", "viral_max", "vel_max", "neg6_max", "neg24_max",
              "neg_delta", "share_delta", "host_delta", "prov_neg", "nat_neg", "lat", "lon", "dist_jhb", "urban", "prov_code"]
RATES = ["n_z", "n_ratio", "p_hostile_mean", "p_hostile_mean_z", "p_hostile_mean_delta",
         "mob_rate", "mob_rate_z", "mob_rate_delta", "threat_rate", "threat_rate_z",
         "expel_rate", "expel_rate_z", "derog_rate", "derog_rate_z", "sev_mean_z", "viral_mean_z",
         "prov_hostile_z", "nat_hostile_z", "dist_jhb", "urban", "prov_code", "has_baseline"]
RATES_VOL = RATES + ["log_n"]

published = {
    "nowcast": pd.read_csv(OUT / "zone_model_comparison_nowcast.csv"),
    "forecast": pd.read_csv(OUT / "zone_model_comparison.csv"),
}

def run_horizon(horizon):
    """horizon 0 = this week, 1 = next week. A town with no tweets next week counts as Green."""
    panel = W.copy()
    if horizon == 1:
        panel = panel[panel.week < panel.week.max()].copy()
        panel["y"] = panel.zone_next.fillna(0).astype(int)
        sets = {
            "A: volume only": VOL,
            "B: binary sentiment sums": SUMS,
            "R: within-town rates + mobilisation": RATES,
            "R+vol: rates plus log volume": RATES_VOL,
            "R+last: rates plus this week's zone": RATES + ["zone_now"],
        }
        baselines = {"Persistence (this week's zone)": panel.zone_now.to_numpy().astype(int)}
    else:
        panel["y"] = panel.zone_now.astype(int)
        sets = {
            "A: volume only": VOL,
            "B: binary sentiment sums": SUMS,
            "R: within-town rates + mobilisation": RATES,
            "R+vol: rates plus log volume": RATES_VOL,
        }
        baselines = {"Last week's zone": panel.zone_prev.to_numpy().astype(int)}
    panel["year"] = panel.week.dt.year
    years = panel.year.to_numpy()
    yt = panel.y.to_numpy()
    mask = np.isin(years, TEST_YEARS)
    rows = []
    fitted = {}
    for name, pred in baselines.items():
        rows.append(evaluate(name, years, yt, pred, pred.astype(float), mask))
    rows.append(evaluate("Always Green", years, yt, np.zeros(len(panel), int), np.zeros(len(panel)), mask))
    for kind in ("RF", "GB"):
        for sn, cols in sets.items():
            pr = np.full((len(panel), 3), np.nan)
            model_2022 = None
            for ty in TEST_YEARS:
                tr = years < ty
                te = years == ty
                ytr = panel.y.to_numpy()[tr]
                if tr.sum() == 0 or te.sum() == 0 or len(np.unique(ytr)) < 2:
                    continue
                model, P = fit_predict(kind, panel.loc[tr, cols], ytr, panel.loc[te, cols])
                pr[te] = P
                if ty == 2022:
                    model_2022 = (model, cols, tr, te)
            score = np.nan_to_num(pr[:, 1]) + np.nan_to_num(pr[:, 2]) + 1e-6 * np.nan_to_num(pr[:, 2])
            pred = budget_zones(years, score)
            label = f"{kind} | {sn}"
            rows.append(evaluate(label, years, yt, pred, score, mask))
            fitted[label] = dict(score=score, pred=pred, held=model_2022, pr=pr)
    res = pd.DataFrame(rows).round(3)
    return panel, res, fitted

print("town-weeks", len(W), "towns", W.town.nunique(), "zones", W.zone_now.value_counts().sort_index().to_dict(), flush=True)
print("incidents:", incident_note, flush=True)
now_panel, now_res, now_fit = run_horizon(0)
fc_panel, fc_res, fc_fit = run_horizon(1)
now_res.to_csv(OUT / "zone_rates_comparison_nowcast.csv", index=False)
fc_res.to_csv(OUT / "zone_rates_comparison_forecast.csv", index=False)
print("\nNOWCAST\n", now_res.to_string(index=False), flush=True)
print("\nFORECAST\n", fc_res.to_string(index=False), flush=True)

def rf_row(res, label):
    hit = res[res.model == label]
    if hit.empty:
        raise SystemExit(f"missing model row: {label}")
    return hit.iloc[0]

now_rates = rf_row(now_res, "RF | R: within-town rates + mobilisation")
now_rvol = rf_row(now_res, "RF | R+vol: rates plus log volume")
now_sums = rf_row(now_res, "RF | B: binary sentiment sums")
now_vol = rf_row(now_res, "RF | A: volume only")
ref_now = published["nowcast"]
ref_b = ref_now[ref_now.model == "RF | B: sentiment+RAG+virality+spatial"].iloc[0]
ref_a = ref_now[ref_now.model == "RF | A: volume only"].iloc[0]

best_name = now_rates.model
kind, sn = best_name.split(" | ")
held = now_fit[best_name]["held"]
per_year = []
yt = now_panel.y.to_numpy()
yp = now_fit[best_name]["pred"]
years = now_panel.year.to_numpy()
for ty in TEST_YEARS:
    m = years == ty
    alert_true = yt[m] > 0
    alert_pred = yp[m] > 0
    per_year.append(dict(
        year=ty, rows=int(m.sum()), amber=int((yt[m] == 1).sum()), red=int((yt[m] == 2).sum()),
        alert_recall=float((alert_true & alert_pred).sum() / max(alert_true.sum(), 1)),
        alert_precision=float((alert_true & alert_pred).sum() / max(alert_pred.sum(), 1)),
    ))
pd.DataFrame(per_year).round(3).to_csv(OUT / "zone_rates_per_year_nowcast.csv", index=False)

if held is not None:
    model, cols, tr, te = held
    pi = permutation_importance(
        model, now_panel.loc[te, cols], now_panel.y.to_numpy()[te],
        scoring=alert_scorer, n_repeats=8, random_state=SEED, n_jobs=1)
    imp = pd.DataFrame({"feature": cols, "importance": pi.importances_mean, "sd": pi.importances_std})
    imp = imp.sort_values("importance", ascending=False)
    imp.to_csv(OUT / "zone_rates_feature_importance_nowcast.csv", index=False)
    print(imp.head(12).to_string(index=False), flush=True)

pred = now_panel.copy()
pred["pred_zone"] = np.where(np.isin(pred.year, TEST_YEARS), yp, -1)
pred["p_alert"] = now_fit[best_name]["score"]
pred[["town", "week", "prov", "lat", "lon", "n", "n_z", "p_hostile_mean", "mob_rate",
      "zone_now", "y", "pred_zone", "p_alert"]].to_csv(OUT / "zone_rates_predictions_nowcast.csv", index=False)

ref_fc = published["forecast"]
fc_rates = rf_row(fc_res, "RF | R: within-town rates + mobilisation")
fc_last = rf_row(fc_res, "RF | R+last: rates plus this week's zone")
fc_sums = rf_row(fc_res, "RF | B: binary sentiment sums")
fc_persist = fc_res[fc_res.model.str.startswith("Persistence")].iloc[0]
ref_fc_b = ref_fc[ref_fc.model == "RF | B: sentiment+RAG+virality+spatial"].iloc[0]
ref_persist = ref_fc[ref_fc.model.str.startswith("Persistence")].iloc[0]

summary = {
    "incident_target": incident_note,
    "labels_rewritten": False,
    "town_weeks": int(len(W)),
    "towns": int(W.town.nunique()),
    "published_nowcast_volume_pr_auc": float(ref_a.pr_auc_alert),
    "published_nowcast_sentiment_pr_auc": float(ref_b.pr_auc_alert),
    "published_nowcast_sentiment_lift": float(ref_b.lift),
    "rerun_volume_pr_auc": float(now_vol.pr_auc_alert),
    "binary_sums_pr_auc": float(now_sums.pr_auc_alert),
    "binary_sums_model": str(now_sums.model),
    "rates_pr_auc": float(now_rates.pr_auc_alert),
    "rates_lift": float(now_rates.lift),
    "rates_model": str(now_rates.model),
    "rates_plus_volume_pr_auc": float(now_rvol.pr_auc_alert),
    "rates_plus_volume_model": str(now_rvol.model),
    "rates_beat_published_sentiment": bool(now_rates.pr_auc_alert > ref_b.pr_auc_alert + 0.01),
    "rates_plus_volume_beats_published": bool(now_rvol.pr_auc_alert > ref_b.pr_auc_alert + 0.01),
    "forecast_binary_sums_pr_auc": float(fc_sums.pr_auc_alert),
    "forecast_binary_sums_lift": float(fc_sums.lift),
    "forecast_rates_pr_auc": float(fc_rates.pr_auc_alert),
    "forecast_rates_lift": float(fc_rates.lift),
    "forecast_rates_model": str(fc_rates.model),
    "forecast_rates_plus_last_zone_pr_auc": float(fc_last.pr_auc_alert),
    "forecast_rates_plus_last_zone_lift": float(fc_last.lift),
    "forecast_rates_plus_last_zone_model": str(fc_last.model),
    "forecast_persistence_pr_auc": float(fc_persist.pr_auc_alert),
    "forecast_persistence_lift": float(fc_persist.lift),
    "published_forecast_sentiment_pr_auc": float(ref_fc_b.pr_auc_alert),
    "published_forecast_persistence_lift": float(ref_persist.lift),
}
json.dump(summary, open(OUT / "improvement_summary.json", "w"), indent=2)
print(json.dumps(summary, indent=2), flush=True)

def panel_chart():
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.4), sharey=False)
    specs = [
        (axes[0], "Same week", [
            ("Published volume", float(ref_a.pr_auc_alert), "#9AA4B2"),
            ("Published sentiment", float(ref_b.pr_auc_alert), "#9AA4B2"),
            ("Volume, this run", float(now_vol.pr_auc_alert), "#1F4E79"),
            ("Binary sums", float(now_sums.pr_auc_alert), "#1F4E79"),
            ("Town rates", float(now_rates.pr_auc_alert), "#C0504D"),
            ("Rates + volume", float(now_rvol.pr_auc_alert), "#C0504D"),
        ]),
        (axes[1], "Next week", [
            ("Published persistence", float(ref_persist.pr_auc_alert), "#9AA4B2"),
            ("Published sentiment", float(ref_fc_b.pr_auc_alert), "#9AA4B2"),
            ("Persistence, this run", float(fc_persist.pr_auc_alert), "#1F4E79"),
            ("Binary sums", float(fc_sums.pr_auc_alert), "#1F4E79"),
            ("Town rates", float(fc_rates.pr_auc_alert), "#C0504D"),
        ]),
    ]
    for ax, title, rows in specs:
        ys = np.arange(len(rows))
        ax.barh(ys, [v for _, v, _ in rows], color=[c for _, _, c in rows], height=0.72)
        ax.set_yticks(ys, [n for n, _, _ in rows])
        ax.set_xlabel("Alert PR-AUC")
        ax.set_title(title)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for y, (_, v, _) in zip(ys, rows):
            ax.text(v + 0.008, y, f"{v:.3f}", va="center", fontsize=8)
        ax.set_xlim(0, max(v for _, v, _ in rows) * 1.28)
    fig.tight_layout()
    fig.savefig(FIG / "zone_rates_comparison.png")
    plt.close()

panel_chart()
print("saved figure", FIG / "zone_rates_comparison.png", flush=True)
