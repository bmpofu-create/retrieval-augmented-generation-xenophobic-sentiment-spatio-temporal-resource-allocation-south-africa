"""Stage 3 - EDA: spatial KDE (negative sentiment vs recorded attacks) and lead-lag analysis."""
import json, numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.stats import spearmanr
from common import *

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 130})
NEG, ATT, INK = "#C0504D", "#1F4E79", "#333333"
df = pd.read_csv(OUT / "features_tweet.csv.gz", parse_dates=["dt"])
neg = df[df.Sentiment == "Negative"]
# incident table: (town, week) with attacks - one record per incident, weighted by attacks
a = df[df["Number of Attacks"] > 0].copy(); a["week"] = a["dt"].dt.to_period("W").dt.start_time
inc = a.groupby(["Tweet Origin", "week"]).agg(attacks=("Number of Attacks", "max"), lat=("cent_lat", "first"), lon=("cent_lon", "first")).reset_index()
print("incidents", len(inc), "attacks", inc.attacks.sum())

# ---------- KDE on a fixed grid, fixed bandwidth (~4.5 km)
BOX = dict(lon=(27.6, 29.0), lat=(-26.6, -25.3)); STEP = 0.01; SIG = 4
lon_e = np.arange(BOX["lon"][0], BOX["lon"][1] + STEP, STEP); lat_e = np.arange(BOX["lat"][0], BOX["lat"][1] + STEP, STEP)
def kde(lon, lat, w=None):
    H, _, _ = np.histogram2d(lat, lon, bins=[lat_e, lon_e], weights=w)
    Z = gaussian_filter(H, SIG); return Z / Z.sum()
g = neg[(neg.cent_lon.between(*BOX["lon"])) & (neg.cent_lat.between(*BOX["lat"]))]
# use raw tweet coordinates for negative density, town centroids for incidents
gi = inc[(inc.lon.between(*BOX["lon"])) & (inc.lat.between(*BOX["lat"]))]
Zn = kde(g.longitude, g.latitude); Za = kde(gi.lon, gi.lat, gi.attacks)
mask = (Zn > 0) | (Za > 0)
rho = spearmanr(Zn[mask], Za[mask])[0]
def top_mask(Z, q): return Z >= np.quantile(Z[Z > 0], 1 - q)
res = {"gauteng_box_negative_tweets": int(len(g)), "gauteng_box_incidents": int(len(gi)), "spearman_kde_corr": float(rho)}
for q in (0.1, 0.2):
    tm = top_mask(Zn, q); res[f"attack_mass_in_top{int(q*100)}pct_neg_density"] = float(Za[tm].sum() / Za.sum())
    res[f"jaccard_hotspots_top{int(q*100)}"] = float((tm & top_mask(Za, q)).sum() / (tm | top_mask(Za, q)).sum())
# permutation null: random town relabel of incident weights across towns with >=30 tweets
towns = df.groupby("Tweet Origin").agg(lat=("cent_lat", "first"), lon=("cent_lon", "first"), n=("raw_id", "size")).query("n>=30")
towns = towns[towns.lon.between(*BOX["lon"]) & towns.lat.between(*BOX["lat"])]
rng = np.random.default_rng(SEED); tm10 = top_mask(Zn, 0.1); null = []
for _ in range(300):
    pick = towns.sample(len(gi), replace=True, random_state=int(rng.integers(1e9)))
    Zr = kde(pick.lon, pick.lat, gi.attacks.to_numpy()); null.append(Zr[tm10].sum() / Zr.sum())
res["null_attack_mass_top10_mean"] = float(np.mean(null)); res["perm_p_top10"] = float((np.sum(np.array(null) >= res["attack_mass_in_top10pct_neg_density"]) + 1) / 301)

# ---------- town table (>=30 tweets)
tt = df.groupby("Tweet Origin").agg(n=("raw_id", "size"), neg=("is_neg", "sum"), lat=("cent_lat", "first"), lon=("cent_lon", "first"),
                                    province=("Province", "first"), attack_sum=("Number of Attacks", "max")).query("n>=30")
tt["neg_share"] = tt.neg / tt.n
ti = inc.groupby("Tweet Origin").agg(incident_weeks=("week", "size"), attacks_total=("attacks", "sum"))
tt = tt.drop(columns="attack_sum").join(ti).fillna({"incident_weeks": 0, "attacks_total": 0})
res["town_spearman_negcount_vs_attacks"] = float(spearmanr(tt.neg, tt.attacks_total)[0])
res["town_spearman_negshare_vs_attacks"] = float(spearmanr(tt.neg_share, tt.attacks_total)[0])
tt.sort_values("neg", ascending=False).to_csv(OUT / "town_summary.csv")
np.savez_compressed(OUT / "kde_grids.npz", Zn=Zn, Za=Za, lon_e=lon_e, lat_e=lat_e)

fig, ax = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
ext = [lon_e[0], lon_e[-1], lat_e[0], lat_e[-1]]
for k, (Z, cm, ttl) in enumerate([(Zn, "Reds", "Negative-sentiment density"), (Za, "Blues", "Recorded-attack density"), (None, None, "Overlap of top-10% hotspots")]):
    if Z is not None:
        ax[k].imshow(np.ma.masked_less(Z, Z.max() * 0.01), origin="lower", extent=ext, cmap=cm, aspect="auto")
    else:
        ov = np.zeros(Zn.shape); ov[top_mask(Zn, .1)] += 1; ov[top_mask(Za, .1)] += 2
        from matplotlib.colors import ListedColormap
        ax[k].imshow(np.ma.masked_equal(ov, 0), origin="lower", extent=ext, aspect="auto", cmap=ListedColormap([NEG, ATT, "#6B3F7C"]), vmin=1, vmax=3)
        ax[k].plot([], [], "s", c=NEG, label="negative only"); ax[k].plot([], [], "s", c=ATT, label="attacks only"); ax[k].plot([], [], "s", c="#6B3F7C", label="both"); ax[k].legend(loc="lower right", frameon=False)
    ax[k].set_title(ttl, loc="left", color=INK)
    for nme, r in towns.nlargest(9, "n").iterrows(): ax[k].annotate(nme, (r.lon, r.lat), fontsize=7, color=INK, xytext=(3, 3), textcoords="offset points"); ax[k].plot(r.lon, r.lat, ".", c=INK, ms=3)
ax[0].set_ylabel("Latitude"); [x.set_xlabel("Longitude") for x in ax]
fig.suptitle("Gauteng: where hostile discourse and recorded attacks concentrate (fixed ~4.5 km kernel)", x=0.01, ha="left", color=INK)
fig.tight_layout(); fig.savefig(FIG / "kde_gauteng.png"); plt.close()

# ---------- lead-lag: does negative sentiment precede recorded attacks?
def leadlag(path, name, bin_h):
    h = pd.read_csv(path, parse_dates=[0], index_col=0).resample(f"{bin_h}h").sum()
    # restrict to the span where attacks are observable and tweet volume exists
    x_neg = np.log1p(h["neg"].to_numpy()); y_att = np.log1p(h["attacks"].to_numpy())
    # de-trend sentiment: subtract 7-day rolling mean so slow volume trends do not dominate
    w = int(24 * 7 / bin_h); xd = x_neg - pd.Series(x_neg).rolling(w, min_periods=1, center=True).mean().to_numpy()
    yd = y_att - pd.Series(y_att).rolling(w, min_periods=1, center=True).mean().to_numpy()
    maxlag = int(24 * 5 / bin_h); lags = np.arange(-maxlag, maxlag + 1); cc = []
    for L in lags:  # corr(neg[t], att[t+L]); L>0 => sentiment leads attacks by L bins
        xs, ys = (xd[:len(xd) - L], yd[L:]) if L >= 0 else (xd[-L:], yd[:len(yd) + L]); cc.append(np.corrcoef(xs, ys)[0, 1])
    cc = np.array(cc)
    # block-permutation null (circular shift of attacks by >=7 days) for the 95% band of the max |cc|
    rng = np.random.default_rng(SEED); mx = []
    for _ in range(200):
        s = int(rng.integers(w, len(yd) - w)); ysh = np.roll(yd, s)
        c2 = [np.corrcoef(*( (xd[:len(xd)-L], ysh[L:]) if L >= 0 else (xd[-L:], ysh[:len(ysh)+L]) ))[0, 1] for L in lags[::2]]
        mx.append(np.nanmax(np.abs(c2)))
    band = float(np.percentile(mx, 95)); pk = lags[np.nanargmax(cc)] * bin_h
    return dict(name=name, bin_h=bin_h, lags_h=(lags * bin_h).tolist(), cc=np.nan_to_num(cc).tolist(), band95=band,
                peak_lag_h=int(pk), peak_cc=float(np.nanmax(cc)), n_bins=int(len(h)), sig=bool(np.nanmax(cc) > band))
LL = [leadlag(OUT / "ts_hourly_national.csv", "National, 6-hour bins", 6), leadlag(OUT / "ts_hourly_gauteng.csv", "Gauteng, 6-hour bins", 6),
      leadlag(OUT / "ts_hourly_national.csv", "National, 1-hour bins", 1), leadlag(OUT / "ts_hourly_national.csv", "National, daily bins", 24)]
# restrict peak search to leads (L>0) as the operational question: how long do surges PRECEDE attacks?
for r in LL:
    L = np.array(r["lags_h"]); c = np.array(r["cc"]); m = L > 0; r["best_lead_h"] = int(L[m][c[m].argmax()]); r["best_lead_cc"] = float(c[m].max())
    m2 = L < 0; r["best_lag_h_reverse"] = int(L[m2][c[m2].argmax()]); r["best_lag_cc_reverse"] = float(c[m2].max())
json.dump({"kde": res, "leadlag": LL}, open(OUT / "eda_results.json", "w"), indent=1)
fig = plt.figure(figsize=(13, 5.2)); gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.1])
ax0 = fig.add_subplot(gs[:, 0]); axt = fig.add_subplot(gs[0, 1]); axb = fig.add_subplot(gs[1, 1], sharex=axt)
for r, col in [(LL[0], NEG), (LL[1], ATT)]:
    ax0.plot(r["lags_h"], r["cc"], c=col, lw=2, label=r["name"]); ax0.axhline(r["band95"], c=col, ls=":", lw=1)
ax0.axvline(0, c="#999", lw=1); ax0.set_xlabel("Lag (hours): positive = negative sentiment leads recorded attacks"); ax0.set_ylabel("Cross-correlation (de-trended)")
ax0.set_title("Lead-lag: negative-tweet volume vs recorded attacks", loc="left", color=INK); ax0.legend(frameon=False); ax0.text(0.01, 0.04, "dotted = 95% permutation band", transform=ax0.transAxes, fontsize=8, color="#666")
d = pd.read_csv(OUT / "ts_hourly_national.csv", parse_dates=[0], index_col=0).resample("D").sum()
axt.fill_between(d.index, d["neg"], color=NEG, alpha=.35, lw=0); axt.plot(d.index, d["neg"], c=NEG, lw=.8); axt.set_ylabel("negative tweets / day"); axt.set_title("Daily series", loc="left", color=INK); plt.setp(axt.get_xticklabels(), visible=False)
axb.bar(d.index, d["attacks"], color=ATT, width=1.0); axb.set_ylabel("recorded attacks / day")
fig.tight_layout(); fig.savefig(FIG / "lead_lag.png"); plt.close()
print(json.dumps(res, indent=1)); [print(r["name"], "best lead h:", r["best_lead_h"], round(r["best_lead_cc"], 3), "| peak overall", r["peak_lag_h"], round(r["peak_cc"], 3), "| band", round(r["band95"], 3), "| reverse", r["best_lag_h_reverse"], round(r["best_lag_cc_reverse"], 3)) for r in LL]
