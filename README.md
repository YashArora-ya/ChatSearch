# ChatSearch

ChatSearch is a small, fully local hybrid-search app for a **synthetic** group
chat. It lets you search approximately 4,356 messy English/Hindi/Hinglish
messages by meaning, participant, and time. It does not use a real chat export,
an API key, a GPU, or a remote LLM.

The Streamlit app shows the matching message, relevance score, sender,
timestamp, thread name, and nearby conversation messages.

## What is in the project

```text
ChatSearch/
├── app.py                    # Streamlit user interface
├── data/
│   ├── chat.json             # Primary synthetic dataset
│   ├── chat.txt              # Readable WhatsApp-style rendering
│   ├── queries.json          # 40 labelled evaluation queries
│   ├── embeddings.npy        # Generated local index (ignored by Git)
│   └── embeddings.meta.json  # Index compatibility metadata (ignored by Git)
├── src/
│   ├── data_generator.py     # Deterministic chat generator
│   ├── query_generator.py    # Deterministic evaluation-query generator
│   └── search_engine.py      # Local search, filters, context, evaluation CLI
├── tests/
├── requirements.txt
└── .gitignore
```

## Architecture and search process

1. `src/data_generator.py` uses a fixed random seed to create six months of
   synthetic conversation from eight participants: Yash, Priya, Rahul, Aman,
   Rohan, Sneha, Arjun, and Neha.
2. It includes ordinary informal chat, short replies, typos, forwards,
   Hinglish/code-mixed messages, and three longer decision threads (Manali,
   a birthday venue, and a final-year project).
3. `ChatSearchEngine` turns each message into a compact local feature embedding
   once and saves a `float32` NumPy matrix at `data/embeddings.npy`.
4. At search time it loads that matrix, encodes **only the question**, applies
   sender/date filters, and combines normalized cosine similarity (90%) with
   normalized exact query-token coverage (10%). The result is a simple hybrid
   ranker with semantic similarity as its primary signal.
5. The app asks the engine for two messages before and after each match, so the
   result is shown in its conversation context rather than on its own.

The lightweight embedding implementation is deterministic and dependency-light
by design. It avoids the large model startup and repeated message encoding that
can cause OpenBLAS memory problems on a typical Windows laptop. The Streamlit
resource cache keeps the loaded engine, chat data, and embeddings in memory
through UI reruns; it does not rebuild the index for each search.

## Search modes

- **Semantic** — uses hybrid ranking: related wording and intent remain the
  primary signal, with a small exact-keyword contribution for useful terms.
  This handles queries
  such as “Where did everyone finally agree to go?” even when the answer is
  “Chalo, Manali fix hai. Main hotel dekh leta hu.”
- **Person + Topic** — detects a participant name in the question (or uses the
  optional participant control) and ranks that person's relevant messages.
- **Temporal** — parses practical rules for `today`, `yesterday`, `this week`,
  `last week`, `this month`, `last month`, `in January`, `in March`, and
  `during April`. You can also set an exact date range in the UI.

### Deterministic relative-date convention

This is a synthetic, fixed-date demo, so relative expressions do **not** use
the computer's clock. The engine and all 40 labelled evaluation queries use
the reference date **1 September 2026**. Dates are interpreted as complete
calendar ranges (weeks start on Monday):

- `last month` = 1–31 August 2026
- `this month` = 1–30 September 2026
- `last week` = Monday 24–Sunday 30 August 2026
- `this week` = Monday 31 August–Sunday 6 September 2026

An explicit UI date range takes precedence over a phrase in the question.
This convention keeps evaluation reproducible and prevents a search result
from changing merely because it is run on a different day.

## Setup on Windows

Install a supported Python release first (Python 3.11 or 3.12 is a good
choice). From PowerShell in the project root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If activation is blocked by your PowerShell policy, use this just for the
current terminal and retry activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

If an existing `.venv` says that its original Python executable cannot be
found, it was created with a removed Python installation. Close programs using
it, delete that `.venv` folder, and run the four setup commands above to create
a fresh environment. `.venv` is intentionally not committed.

## Generate or regenerate the data

The repository already contains generated chat data and evaluation queries. To
recreate both deterministically (including their shared `2026-09-01` temporal
reference date):

```powershell
python src/data_generator.py
python src/query_generator.py
```

Then create or refresh the local embedding index and run the built-in sample
search plus evaluation:

```powershell
python src/search_engine.py
```

`data/embeddings.npy` and its small `data/embeddings.meta.json` sidecar are
intentionally ignored by Git. They are derived from `data/chat.json`, can be
regenerated with the command above, and should not be hand-edited. Keeping them
out of version control avoids committing machine-generated artifacts while
still making startup fast after they have been created.

## Start the application

From the project root, with the virtual environment activated:

```powershell
streamlit run app.py
```

Open the local URL Streamlit prints, normally
[http://localhost:8501](http://localhost:8501). The app performs no browser
automation and makes no unnecessary network connections.

## Run tests and evaluation

```powershell
python -m pytest
python src/search_engine.py
```

The engine command prints the number of evaluation queries, Top-1/Top-3/Top-5
accuracy, and any failed query IDs for debugging. The test suite covers loading
data and embeddings, semantic and filtered search, exact calendar ranges for
relative temporal phrases, temporal candidate filtering, context retrieval,
invalid input, top-k behaviour, and labelled evaluation queries.

## Example questions for the UI

- `When did we decide on the Manali trip?`
- `Where did everyone finally agree to go?`
- `What did Priya say about the budget?`
- `What travel option did Arjun recommend?`
- `What did we discuss last month?`
- `What venue was finalized during the June discussion?`
- `What did we end up choosing for the final-year work?`
- `Manali ka final kya hua?`

## Limitations

This is a deterministic teaching project, not a production chat archive.
Semantic matches are strongest for the included conversational domains and
their English/Hinglish variants. Relative time phrases are rule-based rather
than a full natural-language calendar system. The data is intentionally
synthetic and should not be used to make claims about real people.
