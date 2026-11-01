"""RAG prompt construction + pluggable LLM client (the LLM generation step).

The build sandbox has no LLM/API access, so `classify_with_llm` is NOT executed in the reported
results. Provide any callable `llm(prompt:str)->str` (Claude, GPT, a local Llama ...) to use it.
"""
import json, numpy as np, pandas as pd
from lexicon import lexicon_hits

SYSTEM = ("You classify South African social-media posts about foreign nationals. Use the retrieved context "
          "(lexicon, similar labelled posts, recent local incidents) to interpret slang and code-switching. "
          "Reply ONLY with JSON: {\"sentiment\": \"Negative|Neutral|Positive\", \"severity\": 0-3, \"rationale\": \"...\"}. "
          "Severity (negative posts only): 0 none, 1 grievance, 2 hostile/expulsion rhetoric, 3 explicit threat or call to violence.")

def build_prompt(tweet, index, labelled_mask, tweet_labels, town=None, ts=None, k_lex=4, k_sim=5, k_inc=3):
    lex_i, _ = index.search([tweet], k=k_lex, doc_type="lexicon")
    sim_i, sim_s = index.search([tweet], k=k_sim, mask=None) if labelled_mask is None else index.search_vec(index.emb.transform([tweet]), k=k_sim, mask=labelled_mask)
    lines = ["### Lexicon context"]
    seen = {k for k, _, _ in lexicon_hits(tweet)}
    for i in lex_i[0]:
        m = index.meta.iloc[i]; lines.append(f"- {m['text']}" + (" [MATCHED]" if m["key"] in seen else ""))
    lines.append("### Similar labelled posts")
    for i, s in zip(sim_i[0], sim_s[0]):
        m = index.meta.iloc[i]; lines.append(f"- ({tweet_labels[i]}, sim={s:.2f}) {m['text'][:200]}")
    if town is not None and ts is not None:
        inc = index.meta[(index.meta.doc_type == "incident") & (index.meta.town == town) & (index.meta.ts <= pd.Timestamp(ts))
                         & (index.meta.ts > pd.Timestamp(ts) - pd.Timedelta(days=30))].tail(k_inc)
        lines.append("### Recent local incidents (past 30 days)")
        lines += [f"- {t}" for t in inc["text"]] or ["- none recorded"]
    lines.append(f"### Post\n{tweet}")
    return SYSTEM + "\n\n" + "\n".join(lines)

def classify_with_llm(prompt, llm):
    out = llm(prompt)
    try: return json.loads(out[out.index("{"): out.rindex("}") + 1])
    except Exception: return {"sentiment": None, "severity": None, "rationale": out}
