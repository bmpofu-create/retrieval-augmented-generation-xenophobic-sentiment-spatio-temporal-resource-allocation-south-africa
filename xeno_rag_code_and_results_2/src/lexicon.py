"""Seed domain lexicon for South African xenophobic discourse.

Each entry becomes (a) a regex feature for the classifier / severity score and
(b) a retrievable 'lexicon' document in the RAG vector index.
Weights: hostility contribution to the severity score (0 = context only).
IMPORTANT: seed list - descriptions should be reviewed/extended by a domain expert.
"""
import re

# (key, category, weight, regex on lower-cased punctuation-stripped text, description)
LEXICON = [
 # derogatory / dehumanising
 ("kwerekwere", "derogatory", 3, r"\b(ama)?kwerekwere\b", "Derogatory South African term for (African) foreign nationals."),
 ("illegal_aliens", "derogatory", 2, r"\billegal aliens?\b|\baliens\b", "Dehumanising label for undocumented migrants."),
 ("illegals", "derogatory", 2, r"\billegals\b|\billegal foreigners?\b|\billegal immigrants?\b", "Collective label 'illegals' framing migrants as criminals."),
 ("vermin_slur", "derogatory", 3, r"\b(vermin|cockroach(es)?|parasites?|rats)\b", "Dehumanising metaphors applied to migrants."),
 ("voetsek", "derogatory", 2, r"\bvoetsek\b", "Afrikaans vulgar interjection meaning 'get lost'."),
 # violence / threat
 ("kill", "threat", 3, r"\b(kill|killing|killed|murder\w*)\b", "Language of killing; check whether victim or perpetrator framing."),
 ("burn", "threat", 3, r"\b(burn|burning|burnt|torch\w*|necklac\w*)\b", "Arson / burning imagery associated with township violence."),
 ("attack", "threat", 2, r"\b(attack\w*|beat(en|ing)?|assault\w*|stab\w*|chase[ds]?|chasing|hunt\w*)\b", "Physical attack language."),
 ("looting", "threat", 2, r"\b(loot\w*|raid\w*|vandal\w*|shops? (were )?(closed|burnt))\b", "Looting / property destruction of foreign-owned shops."),
 ("violence", "threat", 2, r"\b(violen\w+|riot\w*|unrest|mob|mobs)\b", "Reference to violence or civil unrest."),
 # expulsion
 ("go_back", "expulsion", 2, r"\bgo back\b|\bsend (them )?back\b|\bback to (their|your) countr\w+", "Calls for migrants to leave / return home."),
 ("deport", "expulsion", 2, r"\bdeport\w*\b", "Deportation demands."),
 ("must_leave", "expulsion", 2, r"\bmust (leave|go)\b|\bleave our\b|\bget out\b|\bkick(ed)? (them )?out\b|\bdrive (them )?out\b", "Explicit expulsion rhetoric."),
 # mobilisation / movements
 ("dudula", "mobilisation", 1, r"dudula", "Operation Dudula: anti-illegal-immigration vigilante movement that began in Soweto (2021); 'dudula' is isiZulu for push/force out."),
 ("put_sa_first", "mobilisation", 1, r"putsouthafricansfirst|put south africans first|sa first|south africans first", "#PutSouthAfricansFirst hashtag activism prioritising citizens over foreign nationals."),
 ("march", "mobilisation", 1, r"\bmarch(es|ing)?\b|\bshut ?down\b|\bdoor to door\b|\bprotest\w*\b", "Marches, protests and shutdown calls that can precede flare-ups."),
 ("actionsa_mm", "mobilisation", 0, r"\bactionsa\b|\bmarch and march\b|\bpatriotic alliance\b", "Political/civic groups active in the immigration debate."),
 # grievance & scapegoating
 ("jobs", "grievance", 1, r"\b(our|sa|south african) jobs?\b|\btak(e|ing|es) (over )?(our )?(jobs|country)\b|\bhiring foreigners\b|\bemploying illegal\b", "Economic grievance: foreigners 'taking jobs'."),
 ("crime", "scapegoat", 1, r"\b(criminals?|thieves|thief|drug\w*|prostitut\w*|counterfeit|lawless)\b", "Scapegoating migrants for crime, drugs and prostitution."),
 ("services", "grievance", 1, r"\b(hospitals?|clinics?|housing|water|taxes|resources|services)\b", "Complaints about pressure on public services."),
 ("replace", "grievance", 1, r"\b(replace|invasion|invaded|influx|flood\w*)\b", "'Replacement' / invasion framing."),
 # context-only: nationality and place anchors
 ("nationality", "context", 0, r"\b(zimbabwe\w*|nigeria\w*|malawi\w*|mozambi\w*|somali\w*|pakistan\w*|bangladesh\w*|congo\w*|ethiopia\w*|ghana\w*|indian\w*)\b", "Nationalities most frequently targeted in discourse."),
 ("hotspot_place", "context", 0, r"\b(yeoville|hillbrow|joburg ?cbd|pretoria ?cbd|jeppestown|diepsloot|alexandra|soweto|mamelodi|tembisa|katlehong|umlazi)\b", "Recurring locations in xenophobia-related posts."),
 # local glossary (language help for the LLM)
 ("wena", "glossary", 0, r"\bwena\b|\bnawe\b|\bangithi\b|\bngalelizwe\b", "isiZulu function words ('you', 'with you', 'isn't it') common in code-switched tweets."),
 ("abelungu", "glossary", 0, r"\babelungu\b", "isiZulu for white people."),
 ("jozi", "glossary", 0, r"\b(jozi|joburg|jhb|mzansi)\b", "Colloquial names for Johannesburg / South Africa."),
 ("undocumented", "glossary", 0, r"\b(undocumented|asylum|home affairs|permits?|visa)\b", "Immigration-status vocabulary; often neutral/policy framing."),
]
COMPILED = [(k, c, w, re.compile(rx), d) for k, c, w, rx, d in LEXICON]
CATS = sorted({c for _, c, *_ in LEXICON})

def lexicon_hits(text: str):
    return [(k, c, w) for k, c, w, rx, _ in COMPILED if rx.search(text)]

def lexicon_matrix(texts):
    """Binary hit matrix (n, n_entries)."""
    import numpy as np
    M = np.zeros((len(texts), len(COMPILED)), dtype=np.float32)
    for j, (_, _, _, rx, _) in enumerate(COMPILED):
        M[:, j] = [1.0 if rx.search(t) else 0.0 for t in texts]
    return M

ENTRY_KEYS = [k for k, *_ in LEXICON]
WEIGHTS = [w for _, _, w, *_ in LEXICON]

def severity_from_matrix(M):
    """Heuristic hostile-intent score = sum of matched weights (capped) -> 0..1."""
    import numpy as np
    raw = M @ np.array(WEIGHTS, dtype=np.float32)
    return np.clip(raw / 6.0, 0, 1), raw

def category_frame(texts):
    """Per-tweet lexicon hit counts by category, plus the severity score."""
    import pandas as pd
    M = lexicon_matrix(texts)
    cats = [c for _, c, *_ in LEXICON]
    data = {}
    for c in sorted(set(cats)):
        idx = [i for i, cat in enumerate(cats) if cat == c]
        data[c] = M[:, idx].sum(1)
    sev, raw = severity_from_matrix(M)
    data["severity"] = sev
    data["severity_raw"] = raw
    return pd.DataFrame(data)
