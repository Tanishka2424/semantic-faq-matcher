# Semantic FAQ Matcher

A small project where a user types a question in their own words and the system finds the closest FAQ and shows its answer. If nothing is close enough, it says it doesn't know instead of guessing.

This was a learning project to understand how semantic search works, how to compare models fairly, and how to be honest about where a system fails.

## The idea in simple words

1. A model turns every sentence into a list of numbers (an **embedding**). Sentences with similar meaning get similar numbers.
2. We turn all FAQ questions into embeddings once.
3. When a user asks something, we turn it into an embedding too and find the closest FAQ (using cosine similarity).
4. If the best score is below a **threshold**, we refuse. Otherwise we show that FAQ's answer.

Example: "Where is my parcel?" has no words in common with "How can I track my order?", but the meaning is close, so semantic search matches them. Plain keyword search can't.

## What is inside

| File | What it does |
|---|---|
| `faq_data.py` | The 12 FAQs (question, answer, and 3 alternate phrasings each) |
| `eval.py` | Test set (72 in-scope + 22 out-of-scope questions) and accuracy checks |
| `compare_models.py` | Compares all models and a keyword baseline, saves `results.csv` and `config.json` |
| `misses.py` | Prints the questions the chosen model got wrong |
| `app.py` | Streamlit demo |
| `results.csv` | Output of the model comparison |
| `config.json` | The chosen model and its threshold (read by `app.py`) |

## Models used and why

| Model | Why we tried it |
|---|---|
| `all-MiniLM-L6-v2` | Very small and fast, a common default |
| `all-MiniLM-L12-v2` | Deeper version of the one above, a bit more accurate |
| `all-mpnet-base-v2` | Bigger general-purpose model, known to be strong |
| `bge-small-en-v1.5` | Small model built for search/retrieval (**chosen**) |
| `bge-base-en-v1.5` | Bigger sibling of bge-small, to see if size helps |
| `gte-base` | Another retrieval model from a different team, for variety |
| TF-IDF | Plain keyword search, used as the baseline to beat |
| `flan-t5-base` | Optional: rewrites the answer in a friendlier way. Off by default, because the plain FAQ answer is already correct |

## How we compared them

- **Test set:** 72 questions that should match an FAQ, and 22 that should be refused (some are on purpose close to the shop's topics, like "How do I delete my account?").
- **Main score (`e2e_cv`):** the share of all 94 questions handled correctly, meaning the right FAQ is shown, or an out-of-scope question is refused. The refusal threshold is tuned on part of the data and scored on the rest (5-fold cross-validation), so the score is not just memorised.
- **`top1` / `top3`:** how often the correct FAQ is the best / in the best three.
- **`gap`:** lowest score of a correct match minus the highest score of an out-of-scope question. If it is negative, no single threshold can separate them perfectly.

## Results

| Model | top1 | top3 | e2e_cv | gap | ms/query |
|---|---|---|---|---|---|
| **bge-small-en-v1.5** | 0.944 | 1.000 | **0.915** | -0.087 | 13.6 |
| bge-base-en-v1.5 | 0.931 | 1.000 | 0.904 | -0.107 | 40.6 |
| all-MiniLM-L12-v2 | 0.958 | 1.000 | 0.851 | -0.139 | 14.7 |
| gte-base | 0.903 | 1.000 | 0.851 | -0.030 | 27.2 |
| all-MiniLM-L6-v2 | 0.944 | 1.000 | 0.830 | -0.235 | 6.3 |
| all-mpnet-base-v2 | 0.958 | 1.000 | 0.830 | -0.174 | 53.6 |
| TF-IDF keyword baseline | 0.681 | 0.875 | 0.681 | -0.345 | - |

**What we learned from this**

- Semantic search clearly beats keyword search here (0.915 vs 0.681).
- **Top-1 alone would have picked the wrong model.** MiniLM-L12 and mpnet have the best top-1 but accept more out-of-scope questions, so their overall score is lower.
- Bigger is not always better. `bge-small` and `bge-base` are about equal, so we chose the smaller and faster one.
- The chosen model uses a threshold of **0.71**.

## Where it fails (limitations)

With `bge-small` at threshold 0.71, 8 of 94 questions are wrong:

1. **Neighbouring FAQs get mixed up (4 cases).** "Pay after delivery option?" goes to *payment methods* instead of *cash on delivery*. "Has my item been dispatched yet?" goes to *delivery time* instead of *tracking*. "Return was accepted, where is my payment?" goes to *return policy* instead of *refund*.
2. **Near-topic questions that are not in the FAQ get answered (3 cases).** "How do I delete my account?" matches *reset password*. "Can I change my email address?" matches *change delivery address*. "Do you sell gift cards?" matches *cash on delivery*. The model matches on shared topic words, not on what the person actually wants.
3. **One correct match sits right on the threshold (1 case).** "Is net banking supported?" scored exactly 0.71 and was refused.

A higher threshold would fix group 2 but refuse many correct questions, because some wrong in-scope matches score higher (0.78 to 0.84) than the wrong out-of-scope ones (0.74 to 0.79).

**Other limits to keep in mind**

- Only 12 FAQs. A bigger FAQ set will behave differently.
- The test questions and alternate phrasings were written by us, so they are not fully independent of each other.
- The model was picked as the best of 6 on the same 94 questions it was scored on, so 0.915 is slightly optimistic. One question is about 1 point, so small gaps between models are noise.
- English only. No chat memory: each question is handled on its own.
- Out-of-scope detection relies on one threshold, which is a rough tool.

## Future scope

- Collect new questions from people who have not seen the FAQs and keep them as a separate test set that is never used for tuning.
- Add "unsupported topic" entries (account deletion, gift cards, product catalogue) so near-topic questions are refused on purpose.
- Add more alternate phrasings for confusing pairs (cash on delivery vs payment methods) in a general way, not by copying failing questions.
- Try a hybrid of keyword and semantic search, and a re-ranker for the top 3 matches.
- Fine-tune the embedding model on FAQ question pairs.
- When the top two matches are close, ask the user a clarifying question.
- Log real user questions and use the refused ones to find missing FAQs.
- Support more languages.

## How to run

Tested with Python 3.12.

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Compare models (downloads all models the first time, takes a few minutes):

```
python compare_models.py
```

See the questions that fail:

```
python misses.py
```

Start the demo:

```
python -m streamlit run app.py
```

Notes:

- `python -m streamlit run app.py` is used instead of `streamlit run app.py` because the plain command failed on our Windows setup.
- On some locked-down Windows machines, security policies can block `torch`. If that happens, use another Python version or machine.
- After the first run, models are cached, so the demo works without internet if you set `HF_HUB_OFFLINE=1`.

## The demo

- Example buttons and a text box for questions.
- Green box with the matched answer, the matched FAQ and its source answer.
- Red "No confident match" message when the score is under the threshold.
- Yellow "Ambiguous" warning when the top two matches are almost tied.
- Similarity score and a chart of the top 3 matches.
- A side panel showing what plain keyword search would return for the same question.
- Sidebar sliders for the threshold and the ambiguity margin, and an optional flan-t5 reply.
- A section at the bottom with the model comparison table.