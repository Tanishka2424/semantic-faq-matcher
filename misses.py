import json
from sentence_transformers import SentenceTransformer
from eval import run
from faq_data import FAQS

cfg = json.load(open("config.json"))
thr = cfg["threshold"]
ins, oos = run(SentenceTransformer(cfg["model"]), True)

print(f"{cfg['model']} @ {thr}\nIN-SCOPE misses:")
for q, y, s in ins:
    top, sc = int(s.argmax()), float(s.max())
    if top != y or sc < thr:
        kind = "WRONG FAQ" if top != y else "REFUSED"
        print(f"  {sc:.2f} {kind}: {q!r} -> {FAQS[top]['q']!r} (expected {FAQS[y]['q']!r})")
print("OUT-OF-SCOPE wrongly answered:")
for q, s in oos:
    if float(s.max()) >= thr:
        print(f"  {float(s.max()):.2f}: {q!r} -> {FAQS[int(s.argmax())]['q']!r}")
        