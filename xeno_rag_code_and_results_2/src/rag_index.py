"""Dense vector index for RAG retrieval (lexicon + incident records + labelled tweets).

Embeddings: TF-IDF -> TruncatedSVD (LSA) -> L2 norm. FAISS / ChromaDB are not installable in
the build sandbox, so search uses FAISS when importable and exact numpy cosine otherwise.
"""
import numpy as np, pandas as pd, pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

try:
    import faiss  # type: ignore
    HAVE_FAISS = True
except Exception:
    HAVE_FAISS = False

class Embedder:
    def __init__(self, dim=256, seed=42):
        self.tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=120_000, sublinear_tf=True)
        self.svd = TruncatedSVD(n_components=dim, random_state=seed, algorithm="randomized", n_iter=4)
    def fit(self, texts):
        self.svd.fit(self.tfidf.fit_transform(texts)); return self
    def transform(self, texts):
        Z = self.svd.transform(self.tfidf.transform(texts))
        return (Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)).astype(np.float32)

class VectorIndex:
    """Documents with metadata; cosine search with an optional boolean candidate mask."""
    def __init__(self, embedder: Embedder):
        self.emb = embedder; self.meta = None; self.V = None; self._faiss = None
    def add(self, texts, meta: pd.DataFrame):
        V = self.emb.transform(texts)
        self.V = V if self.V is None else np.vstack([self.V, V])
        self.meta = meta.reset_index(drop=True) if self.meta is None else pd.concat([self.meta, meta], ignore_index=True)
        if HAVE_FAISS:
            self._faiss = faiss.IndexFlatIP(self.V.shape[1]); self._faiss.add(self.V)
        return self
    def search_vec(self, Q, k=10, mask=None, exclude_groups=None, batch=2048):
        """Return (idx, sim) arrays (n_q, k). mask: bool (n_docs,) candidates.
        exclude_groups: (n_q,) group id; docs with meta['grp']==that id are excluded (leave-self-out)."""
        n = len(Q); I = np.full((n, k), -1, dtype=np.int64); S = np.full((n, k), -1.0, dtype=np.float32)
        grp = self.meta["grp"].to_numpy() if exclude_groups is not None and "grp" in self.meta else None
        for s in range(0, n, batch):
            sim = Q[s:s+batch] @ self.V.T
            if mask is not None: sim[:, ~mask] = -2
            if grp is not None: sim[grp[None, :] == exclude_groups[s:s+batch, None]] = -2
            top = np.argpartition(-sim, k, axis=1)[:, :k]
            ts = np.take_along_axis(sim, top, 1); o = np.argsort(-ts, 1)
            I[s:s+batch] = np.take_along_axis(top, o, 1); S[s:s+batch] = np.take_along_axis(ts, o, 1)
        return I, S
    def search(self, texts, k=5, doc_type=None, **kw):
        mask = (self.meta["doc_type"] == doc_type).to_numpy() if doc_type else None
        return self.search_vec(self.emb.transform(texts), k, mask=mask, **kw)
    def save(self, path):
        with open(path, "wb") as f: pickle.dump({"emb": self.emb, "V": self.V, "meta": self.meta}, f)
    @staticmethod
    def load(path):
        d = pickle.load(open(path, "rb")); ix = VectorIndex(d["emb"]); ix.V, ix.meta = d["V"], d["meta"]; return ix

def lexicon_docs():
    from lexicon import LEXICON
    rows = [{"doc_type": "lexicon", "key": k, "category": c, "weight": w, "text": f"{k.replace('_',' ')} ({c}): {d}"} for k, c, w, _, d in LEXICON]
    return pd.DataFrame(rows)

def incident_docs(df):
    """Historical incident records: one per (town, ISO week) with recorded attacks > 0."""
    a = df[df["Number of Attacks"] > 0].copy()
    a["week"] = a["dt"].dt.to_period("W").dt.start_time
    g = a.groupby(["Tweet Origin", "Province", "week"]).agg(
        attacks=("Number of Attacks", "max"), n_tweets=("raw_id", "size"),
        neg_share=("Sentiment", lambda s: (s == "Negative").mean())).reset_index()
    g["text"] = ("Recorded incident week of " + g["week"].dt.strftime("%Y-%m-%d") + " in " + g["Tweet Origin"] + ", " + g["Province"]
                 + ": " + g["attacks"].astype(int).astype(str) + " attacks recorded; " + g["n_tweets"].astype(str)
                 + " related tweets, " + (100 * g["neg_share"]).round().astype(int).astype(str) + "% negative.")
    g["doc_type"] = "incident"; g["ts"] = g["week"]; g["town"] = g["Tweet Origin"]
    return g[["doc_type", "town", "Province", "ts", "attacks", "text"]]
