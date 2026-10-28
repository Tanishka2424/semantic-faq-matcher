"""Accuracy check for the FAQ matcher. Run:  python eval.py            (default model)
       python eval.py BAAI/bge-small-en-v1.5   (any other sentence-transformers model)
Compares (a) canonical questions only vs (b) questions + alternate phrasings,
reports top-1 / top-3 accuracy, and picks the threshold on a DEV half, then reports on a TEST half.
"""
import random
import statistics
import sys
import torch
from sentence_transformers import SentenceTransformer, util
from faq_data import FAQS

# (query, index of correct FAQ). The first 12 are your original notebook test set.
# Add your OWN queries at the bottom (ideally from classmates who haven't seen the FAQs).
IN_SCOPE = [
    ("Where is my package right now?", 0), ("Can I send back a product I don't like?", 1),
    ("I forgot my login password", 2), ("Can I pay with UPI or credit card?", 3),
    ("When will my parcel arrive?", 4), ("I want to stop my order before it ships", 5),
    ("Who do I talk to if I have a problem?", 6), ("Can I pay when the product arrives?", 7),
    ("How will I get my money back?", 8), ("Is delivery available outside India?", 9),
    ("I entered the wrong address, how to edit it?", 10), ("Any promo codes or offers?", 11),
    # additional queries
    ("Track my parcel", 0), ("Has my item been dispatched yet?", 0), ("I need live status of my purchase", 0),
    ("What if I want to exchange or return something?", 1), ("How many days do I have to return an item?", 1), ("Product is defective, want to give it back", 1),
    ("Password not working", 2), ("How to recover my account login?", 2), ("I can't sign in", 2),
    ("Do you take Google Pay?", 3), ("Is net banking supported?", 3), ("Can I use my debit card here?", 3),
    ("How soon will it reach me?", 4), ("Estimated shipping duration", 4), ("Will it come this week?", 4),
    ("Please cancel my purchase", 5), ("I ordered by mistake, can I undo it?", 5), ("Is it too late to cancel?", 5),
    ("Is there a helpline number?", 6), ("How do I email your team?", 6), ("What are your support hours?", 6),
    ("Can I give cash to the delivery person?", 7), ("Is COD possible for expensive items?", 7), ("Pay after delivery option?", 7),
    ("When does the refunded amount show up in my bank?", 8), ("Return was accepted, where is my payment?", 8), ("How do I claim my money back?", 8),
    ("Can I order from the UK?", 9), ("Do you deliver to Dubai?", 9), ("Is shipping outside the country possible?", 9),
    ("Need to update where it gets delivered", 10), ("Can I ship to my office instead?", 10), ("Wrong pincode entered", 10),
    ("Where can I find coupon codes?", 11), ("Is there a sale going on?", 11), ("Do you have any voucher for first order?", 11),
    # --- second set of queries (not used for any tuning) ---
    ("Where's my order at the moment?", 0), ("How do I follow my shipment's progress?", 0),
    ("What's the time limit for sending an item back?", 1), ("Do you accept returns on used products?", 1),
    ("I can't remember my password", 2), ("How do I get a new password for my account?", 2),
    ("Can I pay using a wallet?", 3), ("Which payment options are there at checkout?", 3),
    ("How many days until my order shows up?", 4), ("What's the usual shipping time?", 4),
    ("I don't want the item anymore, can I call off the order?", 5), ("Can I withdraw my order after placing it?", 5),
    ("How can I reach a real person for help?", 6), ("What number do I call for assistance?", 6),
    ("Is there a pay-on-delivery option for my order?", 7), ("Do you allow cash payments when the courier arrives?", 7),
    ("How many days does it take for a refund to arrive?", 8), ("Where will my refund be credited?", 8),
    ("Can someone in Canada buy from you?", 9), ("Do you send parcels to other countries?", 9),
    ("Can I modify the shipping address after ordering?", 10), ("I need my package sent to a different house", 10),
    ("Are there any ongoing offers?", 11), ("Where do I find discount codes?", 11),
]
OUT_OF_SCOPE = [
    "What is the capital of France?", "Who won the cricket match yesterday?", "Tell me a joke",
    "Do you sell shoes?", "What is the price of an iPhone 15?", "How do I become a seller on your platform?",
    "Is my personal data safe with you?", "Do you have a mobile app?", "What is the weather today?",
    "Write a poem about the sea", "How do I invest in mutual funds?", "Recommend a good movie",
    # --- second set (not used for any tuning); these are close to the shop's topics on purpose ---
    "Do you sell gift cards?", "Can I buy in bulk for my company?", "What is the warranty on electronics?",
    "How do I delete my account?", "Do you have a store near me?", "Is this product available in blue?",
    "Can I change my email address?", "How do I leave a product review?",
    "What's the best phone under 20000?", "Who is the prime minister of India?",
]


def build(model, use_alts):
    texts, owner = [], []
    for i, f in enumerate(FAQS):
        for t in [f["q"]] + (f["alts"] if use_alts else []):
            texts.append(t)
            owner.append(i)
    emb = model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)
    return emb, torch.tensor(owner)


def scores_for(model, emb, owner, query):
    q = model.encode(query, convert_to_tensor=True, normalize_embeddings=True)
    sims = util.cos_sim(q, emb)[0]
    return torch.stack([sims[owner == i].max() for i in range(len(FAQS))])


def run(model, use_alts):
    emb, owner = build(model, use_alts)
    ins = [(q, y, scores_for(model, emb, owner, q)) for q, y in IN_SCOPE]
    oos = [(q, scores_for(model, emb, owner, q)) for q in OUT_OF_SCOPE]
    return ins, oos


def accuracy(ins):
    top1 = sum(int(s.argmax()) == y for _, y, s in ins) / len(ins)
    top3 = sum(y in s.topk(3).indices.tolist() for _, y, s in ins) / len(ins)
    return top1, top3


def end_to_end(ins, oos, thr):
    """Fraction of ALL queries handled correctly: right FAQ answered, or out-of-scope refused."""
    ok = sum((float(s.max()) >= thr) and int(s.argmax()) == y for _, y, s in ins)
    ok += sum(float(s.max()) < thr for _, s in oos)
    return ok / (len(ins) + len(oos))


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "all-MiniLM-L6-v2"
    print("Embedding model:", model_name)
    model = SentenceTransformer(model_name)
    print(f"{len(IN_SCOPE)} in-scope queries, {len(OUT_OF_SCOPE)} out-of-scope queries\n")

    for use_alts in (False, True):
        ins, oos = run(model, use_alts)
        t1, t3 = accuracy(ins)
        print(f"[{'questions + alts' if use_alts else 'canonical questions only'}]  top-1 = {t1:.0%}   top-3 = {t3:.0%}")
    print()

    # Final system = with alts. Choose threshold on a random DEV half, report on the other (TEST) half.
    ins, oos = run(model, True)
    rng = random.Random(42)            # fixed seed: same split every run, hard cases spread across halves
    ins_sh, oos_sh = ins[:], oos[:]
    rng.shuffle(ins_sh)
    rng.shuffle(oos_sh)
    dev_in, test_in = ins_sh[0::2], ins_sh[1::2]
    dev_oos, test_oos = oos_sh[0::2], oos_sh[1::2]
    thresholds = [round(0.20 + 0.01 * k, 2) for k in range(0, 76)]   # 0.20 .. 0.95
    dev_scores = {t: end_to_end(dev_in, dev_oos, t) for t in thresholds}
    top = max(dev_scores.values())
    tied = [t for t, v in dev_scores.items() if v == top]
    best_thr = round(statistics.median(tied), 2)   # middle of all equally-good thresholds, not the lowest
    print(f"Threshold chosen on DEV half: {best_thr:.2f}")
    print(f"DEV  end-to-end correct: {end_to_end(dev_in, dev_oos, best_thr):.0%}")
    print(f"TEST end-to-end correct: {end_to_end(test_in, test_oos, best_thr):.0%}   <- small half, noisy: do NOT quote this")
    t1, t3 = accuracy(test_in)
    print(f"TEST in-scope top-1 = {t1:.0%}, top-3 = {t3:.0%}")
    print(f"TEST out-of-scope refused: {sum(float(s.max()) < best_thr for _, s in test_oos)}/{len(test_oos)}\n")

    correct_sc = [float(s.max()) for _, y, s in ins if int(s.argmax()) == y]
    wrong_sc = [float(s.max()) for _, y, s in ins if int(s.argmax()) != y]
    oos_sc = [float(s.max()) for _, s in oos]
    print("Score distribution (best similarity per query):")
    print(f"  correct in-scope : n={len(correct_sc)}  min={min(correct_sc):.2f}  median={statistics.median(correct_sc):.2f}")
    if wrong_sc:
        print(f"  wrong in-scope   : n={len(wrong_sc)}  min={min(wrong_sc):.2f}  max={max(wrong_sc):.2f}")
    print(f"  out-of-scope     : n={len(oos_sc)}  median={statistics.median(oos_sc):.2f}  max={max(oos_sc):.2f}")
    print("  (if out-of-scope max > correct in-scope min, no single threshold separates them)\n")

    n_miss = (sum(int(s.argmax()) != y or float(s.max()) < best_thr for _, y, s in ins)
              + sum(float(s.max()) >= best_thr for _, s in oos))
    print(f"TOTAL MISSES over all {len(ins) + len(oos)} queries at threshold {best_thr:.2f}: {n_miss}\n")

    print("Threshold sweep on ALL queries (descriptive only, not held-out):")
    print("  thr    in-scope answered correctly   in-scope refused   out-of-scope refused")
    for t in [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:
        c = sum(int(s.argmax()) == y and float(s.max()) >= t for _, y, s in ins)
        r = sum(float(s.max()) < t for _, y, s in ins)
        o = sum(float(s.max()) < t for _, s in oos)
        print(f"  {t:.2f}   {c:>3}/{len(ins)}                       {r:>3}/{len(ins)}            {o:>3}/{len(oos)}")
    print()

    BUDGET = 2   # how many in-scope queries we are willing to refuse
    ok_t = [t for t in thresholds if sum(float(s.max()) < t for _, y, s in ins) <= BUDGET]
    op = max(ok_t)
    op_c = sum(int(s.argmax()) == y and float(s.max()) >= op for _, y, s in ins)
    op_o = sum(float(s.max()) < op for _, s in oos)
    print(f"FAIR COMPARISON POINT (highest threshold that refuses <= {BUDGET} in-scope queries): {op:.2f}")
    print(f"  in-scope answered correctly {op_c}/{len(ins)}, out-of-scope refused {op_o}/{len(oos)}\n")

    print("Misses (all queries, final threshold):")
    for q, y, s in ins:
        p, sc = int(s.argmax()), float(s.max())
        if p != y or sc < best_thr:
            print(f"  IN-SCOPE  '{q}' -> predicted '{FAQS[p]['q']}' ({sc:.2f}), expected '{FAQS[y]['q']}'")
    for q, s in oos:
        if float(s.max()) >= best_thr:
            print(f"  OUT-OF-SCOPE accepted: '{q}' -> '{FAQS[int(s.argmax())]['q']}' ({float(s.max()):.2f})")

    a1, a3 = accuracy(ins)
    print()
    print("=" * 60)
    print(f"ACCURACY SUMMARY  (model: {model_name})")
    print(f"  In-scope questions ({len(ins)}): top-1 = {a1:.0%} ({round(a1 * len(ins))}/{len(ins)}), top-3 = {a3:.0%}")
    print(f"  Cutoff {op:.2f} (refuses at most {BUDGET} real questions): "
          f"out-of-scope refused = {op_o}/{len(oos)}, real questions answered correctly = {op_c}/{len(ins)}")
    print("=" * 60)