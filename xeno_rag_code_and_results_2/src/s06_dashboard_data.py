"""Assemble dashboard JSON (embedded in outputs/dashboard.html)."""
import json, numpy as np, pandas as pd
from common import *
now = pd.read_csv(OUT / "zone_predictions_town_week_nowcast.csv", parse_dates=["week"])
fc = pd.read_csv(OUT / "zone_predictions_town_week.csv", parse_dates=["week"])
m = now.merge(fc[["town", "week", "y", "pred_zone", "p_alert"]].rename(columns={"y": "fc_actual", "pred_zone": "fc_pred", "p_alert": "fc_p"}), on=["town", "week"], how="left")
m = m[m.week >= "2019-01-01"].copy()
for c in ("fc_actual", "fc_pred"): m[c] = m[c].fillna(-1).astype(int)
m["fc_p"] = m["fc_p"].fillna(0)
weeks = sorted(m.week.unique()); wi = {w: i for i, w in enumerate(weeks)}
towns = m.groupby("town").agg(lat=("lat", "first"), lon=("lon", "first"), prov=("prov", "first"), n=("n", "sum"), neg=("neg", "sum")).reset_index()
ti = {t: i for i, t in enumerate(towns.town)}
rows = [[ti[r.town], wi[r.week], int(r.n), int(r.neg), int(r.zone_now), int(r.pred_zone), int(r.fc_pred), int(r.fc_actual), round(r.p_alert * 100), round(r.fc_p * 100)] for r in m.itertuples()]
K = np.load(OUT / "kde_grids.npz"); LON, LAT = (27.6, 28.6), (-26.6, -25.6)
def crop(Z):
    i0, i1 = int(round((LAT[0] - K["lat_e"][0]) / 0.01)), int(round((LAT[1] - K["lat_e"][0]) / 0.01)); j0, j1 = int(round((LON[0] - K["lon_e"][0]) / 0.01)), int(round((LON[1] - K["lon_e"][0]) / 0.01))
    Z = Z[i0:i1, j0:j1]; Z = Z[:Z.shape[0] // 2 * 2, :Z.shape[1] // 2 * 2]; Z = Z.reshape(Z.shape[0] // 2, 2, Z.shape[1] // 2, 2).sum((1, 3))
    Z = np.sqrt(Z / Z.max()); return (Z * 255).round().astype(int).tolist()
eda = json.load(open(OUT / "eda_results.json")); sm = pd.read_csv(OUT / "sentiment_model_comparison.csv"); zc = pd.read_csv(OUT / "zone_model_comparison_nowcast.csv"); zf = pd.read_csv(OUT / "zone_model_comparison.csv")
imp = pd.read_csv(OUT / "zone_feature_importance_nowcast.csv").head(8)
ll = {r["name"]: r for r in eda["leadlag"]}
def trim(r): 
    L = np.array(r["lags_h"]); k = np.abs(L) <= 120; return {"lags": L[k].tolist(), "cc": np.round(np.array(r["cc"])[k], 4).tolist(), "band": round(r["band95"], 4)}
data = {"weeks": [pd.Timestamp(w).strftime("%Y-%m-%d") for w in weeks], "towns": towns.assign(lat=towns.lat.round(4), lon=towns.lon.round(4)).to_dict("records"), "rows": rows,
        "kde": {"neg": crop(K["Zn"]), "att": crop(K["Za"]), "box": [LON[0], LON[1], LAT[0], LAT[1]]},
        "sent": sm[["model", "accuracy", "macro_f1", "f1_Negative", "f1_Neutral", "f1_Positive"]].round(3).to_dict("records"),
        "zone_now": zc[["model", "macro_f1", "pr_auc_alert", "lift", "alert_recall", "recall_red"]].round(3).to_dict("records"),
        "zone_fc": zf[["model", "macro_f1", "pr_auc_alert", "lift", "alert_recall", "recall_red"]].round(3).to_dict("records"),
        "imp": imp.round(4).to_dict("records"), "kde_stats": eda["kde"], "ll": {k: trim(v) for k, v in ll.items() if "daily" not in k and "1-hour" not in k},
        "ll_best": {k: {kk: v[kk] for kk in ("best_lead_h", "best_lead_cc", "peak_lag_h", "peak_cc", "band95")} for k, v in ll.items()}}
json.dump(data, open(OUT / "dashboard_data.json", "w"), separators=(",", ":"))
print(len(json.dumps(data)) / 1e3, "KB;", len(rows), "rows;", len(weeks), "weeks;", len(towns), "towns")
print(towns.assign(inbox=towns.lon.between(*LON) & towns.lat.between(*LAT)).inbox.value_counts().to_dict())
