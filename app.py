"""PS47 Semantic FAQ Matcher - Streamlit demo.
Run:  pip install streamlit sentence-transformers transformers torch pandas
      streamlit run app.py
Run it once before the jury so the models are already downloaded (no internet needed after that).
"""
import json
import re
from pathlib import Path
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer, util

st.set_page_config(page_title="Semantic FAQ Matcher", page_icon="💬", layout="wide")

from faq_data import FAQS, EXAMPLES

HERE = Path(__file__).parent
# config.json is written by compare_models.py (best model + its tuned threshold).
CFG = json.loads((HERE / "config.json").read_text()) if (HERE / "config.json").exists() else {}
MODEL_NAME = CFG.get("model", "sentence-transformers/all-mpnet-base-v2")
DEFAULT_THR = float(CFG.get("threshold", 0.35))


@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedder():
    return SentenceTransformer(MODEL_NAME)


@st.cache_resource(show_spinner="Loading generator (flan-t5)...")
def load_generator():
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    tok = AutoTokenizer.from_pretrained("google/flan-t5-base")
    mdl = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    return tok, mdl


@st.cache_resource
def build_index(_model):
    texts, owner = [], []
    for i, f in enumerate(FAQS):
        for t in [f["q"]] + f["alts"]:
            texts.append(t)
            owner.append(i)
    emb = _model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)
    return emb, owner


def retrieve(query, model, emb, owner):
    q = model.encode(query, convert_to_tensor=True, normalize_embeddings=True)
    sims = util.cos_sim(q, emb)[0].tolist()
    best = {}
    for s, o in zip(sims, owner):
        best[o] = max(best.get(o, -1.0), s)
    ranked = sorted(best.items(), key=lambda x: x[1], reverse=True)
    return ranked  # [(faq_idx, score), ...] best first


STOP = {"the", "a", "an", "is", "are", "to", "of", "you", "your", "can", "i", "my", "in", "on", "and", "or", "for"}


def content_words(text):
    return {w for w in re.findall(r"[a-z0-9@.\-]+", text.lower()) if w not in STOP}


def generate_reply(query, answer, tok, mdl):
    prompt = ("You are a polite customer support assistant. Answer the customer's question "
              "using ONLY the information given.\n\n"
              f"Information: {answer}\nCustomer question: {query}\nReply:")
    inputs = tok(prompt, return_tensors="pt")
    out = mdl.generate(**inputs, max_new_tokens=80, num_beams=4)
    text = tok.decode(out[0], skip_special_tokens=True).strip()
    # Guard: a small LLM sometimes ignores the context (e.g. "Is there anything else I can help with?").
    # If the reply shares too little with the retrieved answer, fall back to the FAQ answer itself.
    ans_words = content_words(answer)
    overlap = len(content_words(text) & ans_words) / max(1, len(ans_words))
    if overlap < 0.5:
        return answer, True
    return text, False


def keyword_match(query):
    qw = content_words(query)
    scores = [len(qw & content_words(f["q"])) for f in FAQS]
    best = max(range(len(FAQS)), key=lambda i: scores[i])
    return best, scores[best]


# ---------------- UI ----------------
st.title("💬 Semantic FAQ Matcher (RAG)")
st.caption("Embeddings retrieve the closest FAQ by meaning; a generative model phrases the reply; "
           "below the threshold the bot refuses instead of guessing.")

with st.sidebar:
    st.header("Settings")
    st.caption(f"Embedding model: `{MODEL_NAME}`")
    threshold = st.slider("Similarity threshold", 0.0, 1.0, DEFAULT_THR, 0.01,
                          help="Below this, the bot says it has no answer.")
    margin = st.slider("Ambiguity margin (top1 - top2)", 0.0, 0.2, 0.03, 0.01,
                       help="If the top two FAQs are this close, show 'did you mean'.")
    use_gen = st.checkbox("Generate reply with flan-t5 (optional)", value=False)
    show_kw = st.checkbox("Show keyword-baseline result", value=True)

model = load_embedder()
emb, owner = build_index(model)
if use_gen:
    load_generator()

if "query" not in st.session_state:
    st.session_state.query = ""

cols = st.columns(len(EXAMPLES))
for c, ex in zip(cols, EXAMPLES):
    if c.button(ex, use_container_width=True):
        st.session_state.query = ex

query = st.text_input("Ask a question", key="query", placeholder="e.g. Where is my parcel?")

if query.strip():
    ranked = retrieve(query, model, emb, owner)
    best_idx, best_score = ranked[0]
    second_score = ranked[1][1]

    left, right = st.columns([3, 2])
    with left:
        if best_score < threshold:
            st.error("No confident match. The generator was NOT called.")
            st.write("Sorry, I could not find a matching answer. Please contact support.")
        else:
            faq = FAQS[best_idx]
            if best_score - second_score < margin:
                st.warning("Ambiguous: the top two matches are very close. Showing the best one.")
            if use_gen:
                tok, mdl = load_generator()
                reply, fell_back = generate_reply(query, faq["a"], tok, mdl)
                st.success(reply)
                if fell_back:
                    st.caption("Generator output ignored the retrieved context, so the FAQ answer was used.")
            else:
                st.success(faq["a"])
            st.markdown(f"**Retrieved FAQ:** {faq['q']}  \n**Source answer:** {faq['a']}")
        st.metric("Best similarity", f"{best_score:.2f}", f"threshold {threshold:.2f}", delta_color="off")

    with right:
        st.subheader("Top 3 matches")
        top = pd.DataFrame(
            [{"FAQ": FAQS[i]["q"], "score": round(s, 3)} for i, s in ranked[:3]]
        ).set_index("FAQ")
        st.bar_chart(top)
        if show_kw:
            kw_idx, kw_hits = keyword_match(query)
            st.subheader("Keyword baseline")
            if kw_hits == 0:
                st.write("No shared keywords, so it has nothing to return.")
            else:
                st.write(f"Would return: **{FAQS[kw_idx]['q']}** ({kw_hits} shared word(s))")

# ---------------- Model-selection evidence ----------------
results_path = HERE / "results.csv"
with st.expander("Model comparison (from compare_models.py)"):
    if results_path.exists():
        res = pd.read_csv(results_path)
        st.dataframe(res, use_container_width=True, hide_index=True)
        st.bar_chart(res.set_index("model")["e2e_cv"])
        st.caption("e2e_cv = share of ALL queries handled correctly (right FAQ answered, or out-of-scope refused), "
                   "threshold tuned on held-out folds. ~90 queries, so differences under ~2 points are noise.")
    else:
        st.write("Run `python compare_models.py` to generate results.csv.")