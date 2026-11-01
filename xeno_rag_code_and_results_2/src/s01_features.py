"""Stage 2 - structuring & wrangling: time-series velocity, virality and spatial features."""
import numpy as np, pandas as pd
from common import *

df = load_raw()
# ---- integrity checks on the model-ready file
assert df[["Sentiment", "Province", "District", "latitude", "longitude", "Number of Attacks"]].notna().all().all()
assert set(df.Sentiment.unique()) == set(CLASSES)
assert df.latitude.between(-35, -22).all() and df.longitude.between(16, 33).all()

# ---- virality
df["engagement"] = df.Likes + df.Retweets + df["No of replies"]
df["virality"] = np.log1p(df.Likes + 2 * df.Retweets + 1.5 * df["No of replies"])
df["retweet_ratio"] = df.Retweets / (df.Likes + 1)
df["is_neg"] = (df.Sentiment == "Negative").astype(int)
df["hour"], df["dow"] = df.dt.dt.hour, df.dt.dt.dayofweek

# ---- spatial: town centroids (median of reported coords) and distance to Johannesburg CBD
cent = df.groupby("Tweet Origin")[["latitude", "longitude"]].median().rename(columns={"latitude": "cent_lat", "longitude": "cent_lon"})
df = df.join(cent, on="Tweet Origin")
def hav(lat1, lon1, lat2, lon2):
    p = np.pi / 180; a = np.sin((lat2-lat1)*p/2)**2 + np.cos(lat1*p)*np.cos(lat2*p)*np.sin((lon2-lon1)*p/2)**2
    return 12742 * np.arcsin(np.sqrt(a))
df["dist_jhb_km"] = hav(df.cent_lat, df.cent_lon, -26.2041, 28.0473)
df["urban"] = (df.District == "Urban").astype(int)

# ---- rolling past-only windows (window ends at tweet time, inclusive) for 3 spatial scopes
t = ((df.dt - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1)).to_numpy()  # unit-safe epoch seconds
def rolling_counts(scope_col, hours, flag=None):
    out = np.zeros(len(df))
    keys = df[scope_col] if scope_col else pd.Series(0, index=df.index)
    for _, idx in keys.groupby(keys).groups.items():
        idx = np.asarray(idx); ts = t[idx]; order = np.argsort(ts, kind="stable"); ts_s = ts[order]
        w = np.ones(len(idx)) if flag is None else df[flag].to_numpy()[idx][order].astype(float)
        cs = np.concatenate([[0], np.cumsum(w)]); hi = np.arange(1, len(ts_s) + 1)
        lo = np.searchsorted(ts_s, ts_s - hours * 3600, side="right")
        res = np.empty(len(idx)); res[order] = cs[hi] - cs[lo]; out[idx] = res
    return out
for scope, tag in [(None, "nat"), ("Province", "prov"), ("Tweet Origin", "town")]:
    for h in (6, 24):
        df[f"neg_{h}h_{tag}"] = rolling_counts(scope, h, "is_neg")
        df[f"vol_{h}h_{tag}"] = rolling_counts(scope, h)
    df[f"negshare_24h_{tag}"] = df[f"neg_24h_{tag}"] / df[f"vol_24h_{tag}"].clip(lower=1)
    # sentiment velocity: negatives in last 6h vs average 6h rate over the 18h before that
    base = (df[f"neg_24h_{tag}"] - df[f"neg_6h_{tag}"]) / 3.0
    df[f"neg_velocity_{tag}"] = (df[f"neg_6h_{tag}"] + 1) / (base + 1)
vn = df.virality * df.is_neg; df["_vn"] = vn
ts_s = t; cs = np.concatenate([[0], np.cumsum(vn.to_numpy())])
lo = np.searchsorted(ts_s, ts_s - 24 * 3600, side="right"); df["viral_neg_24h_nat"] = cs[np.arange(1, len(df) + 1)] - cs[lo]
df = df.drop(columns="_vn")
# recent attacks (past-only, 7d, same town / province): history of physical unrest
df["att_flag"] = (df["Number of Attacks"] > 0).astype(int)
df["recent_attack_tweets_7d_prov"] = rolling_counts("Province", 24 * 7, "att_flag")
df["recent_attack_tweets_7d_town"] = rolling_counts("Tweet Origin", 24 * 7, "att_flag")
df.to_csv(OUT / "features_tweet.csv.gz", index=False)

# ---- hourly national / Gauteng time series for lead-lag analysis
h = df.set_index("dt")
def hourly(sub):
    r = sub.resample("1h")
    return pd.DataFrame({"n": r.size(), "neg": r["is_neg"].sum(), "virality_neg": (sub.virality * sub.is_neg).resample("1h").sum(),
                         "attacks": sub["Number of Attacks"].resample("1h").sum(), "att_tweets": sub["att_flag"].resample("1h").sum()})
hourly(h).to_csv(OUT / "ts_hourly_national.csv")
hourly(h[h.Province == "Gauteng"]).to_csv(OUT / "ts_hourly_gauteng.csv")
print(df[["neg_6h_town", "neg_24h_town", "neg_velocity_nat", "virality", "dist_jhb_km"]].describe().round(2).to_string())
print("rows", len(df), "cols", df.shape[1])
