"""Compare embedding models + a TF-IDF keyword baseline.
Run:  python compare_models.py
Writes results.csv (comparison table) and config.json (best model + tuned threshold, read by app.py).
"""
import json
import random
import statistics
import time

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from eval import IN_SCOPE, OUT_OF_SCOPE, run, accuracy, end_to_end
from faq_data import FAQS

MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-MiniLM-L12-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "BAAI/bge-small-en-v1.5",
    "BAAI/bge-base-en-v1.5",
    "thenlper/gte-base",
]
BASELINE = "TF-IDF keyword baseline"
THRS = [round(0.20 + 0.01 * k, 2) for k in range(76)]


def best_thr(ins, oos):
    sc = {t: end_to_end(ins, oos, t) for t in THRS}
    top = max(sc.values())
    return round(statistics.median([t for t, v in sc.items() if v == top]), 2)


def cv_end_to_end(ins, oos, k=5, seed=42):
    """Threshold tuned on k-1 folds, scored on the held-out fold: the honest number."""
    rng = random.Random(seed)
    i, o = ins[:], oos[:]
    rng.shuffle(i)
    rng.shuffle(o)
    hit = tot = 0
    for f in range(k):
        te_i, te_o = i[f::k], o[f::k]
        tr_i = [x for j, x in enumerate(i) if j % k != f]
        tr_o = [x for j, x in enumerate(o) if j % k != f]
        thr = best_thr(tr_i, tr_o)
        n = len(te_i) + len(te_o)
        hit += end_to_end(te_i, te_o, thr) * n
        tot += n
    return hit / tot


def tfidf_run():
    texts, owner = [], []
    for i, f in enumerate(FAQS):
        for t in [f["q"]] + f["alts"]:
            texts.append(t)
            owner.append(i)
    owner = np.array(owner)
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2)).fit(texts)
    M = vec.transform(texts)

    def sc(q):
        s = cosine_similarity(vec.transform([q]), M)[0]
        return torch.tensor([float(s[owner == i].max()) for i in range(len(FAQS))])

    return [(q, y, sc(q)) for q, y in IN_SCOPE], [(q, sc(q)) for q in OUT_OF_SCOPE]


def summarize(name, ins, oos, ms=None):
    t1, t3 = accuracy(ins)
    correct_min = min(float(s.max()) for _, y, s in ins if int(s.argmax()) == y)
    oos_max = max(float(s.max()) for _, s in oos)
    return {
        "model": name,
        "top1": round(t1, 3),
        "top3": round(t3, 3),
        "e2e_cv": round(cv_end_to_end(ins, oos), 3),
        "gap": round(correct_min - oos_max, 3),   # >0: one threshold can separate correct matches from out-of-scope
        "thr_all_data": best_thr(ins, oos),
        "ms_per_query": ms,
    }


if __name__ == "__main__":
    print(f"{len(IN_SCOPE)} in-scope, {len(OUT_OF_SCOPE)} out-of-scope queries")
    rows = [summarize(BASELINE, *tfidf_run())]
    print(rows[-1])
    for name in MODELS:
        try:
            m = SentenceTransformer(name)
        except Exception as e:  # keep going if one model fails to download
            print(f"SKIP {name}: {e}")
            continue
        ins, oos = run(m, True)
        m.encode("warm up")
        t0 = time.time()
        for q, _ in IN_SCOPE[:30]:
            m.encode(q)
        ms = round((time.time() - t0) / 30 * 1000, 1)
        rows.append(summarize(name, ins, oos, ms))
        print(rows[-1])

    df = pd.DataFrame(rows).sort_values("e2e_cv", ascending=False)
    df.to_csv("results.csv", index=False)
    print("\n" + df.to_string(index=False))

    best = df[df.model != BASELINE].iloc[0]
    json.dump({"model": best.model, "threshold": float(best.thr_all_data)}, open("config.json", "w"), indent=2)
    print(f"\nconfig.json written: {best.model} @ threshold {best.thr_all_data}")
    print("Check the top rows: if the top models are within ~2 points, pick the smaller/faster one and edit config.json by hand.")