"""Streamlit interface for searching the synthetic ChatSearch conversation.

Run from the project root with ``streamlit run app.py``. The resource cache is
intentional: it keeps one local search engine, embedding matrix, and query model
in memory while Streamlit reruns this script for UI interactions.
"""

from __future__ import annotations

from datetime import date, datetime
import os
from typing import Any

# Streamlit does not import NumPy during CLI setup; cap its BLAS threads before
# it loads the app's local index, avoiding unnecessary Windows memory pressure.
for _thread_variable in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_thread_variable, "1")

import streamlit as st

from src.search_engine import ChatSearchEngine, TEMPORAL_REFERENCE_DATE


st.set_page_config(page_title="ChatSearch", page_icon="🔎", layout="wide")


@st.cache_resource(show_spinner="Loading the local chat index…")
def load_engine() -> ChatSearchEngine:
    """Create one engine per Streamlit process without rebuilding embeddings."""

    engine = ChatSearchEngine(auto_build=True)
    # The current engine loads an existing index in its constructor. Keeping this
    # compatibility guard also supports an engine with an explicit load_index.
    if getattr(engine, "embeddings", None) is None and hasattr(engine, "load_index"):
        engine.load_index()
    return engine


def message_id(record: dict[str, Any]) -> int | None:
    """Return an ID from either supported message field name."""

    value = record.get("message_id", record.get("id"))
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def safe_format_timestamp(value: Any) -> str:
    """Render an ISO timestamp using a Windows-safe time format."""

    if not value:
        return "Unknown time"
    if isinstance(value, datetime):
        timestamp = value
    else:
        try:
            timestamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return str(value)
    hour = timestamp.strftime("%I").lstrip("0") or "0"
    return f"{timestamp.strftime('%d %B %Y')}, {hour}:{timestamp.strftime('%M %p')}"


def chat_date_bounds(engine: ChatSearchEngine) -> tuple[date, date]:
    """Find useful defaults for the optional date-range controls."""

    values: list[date] = []
    for record in getattr(engine, "messages", []):
        raw_timestamp = record.get("timestamp") if isinstance(record, dict) else None
        if raw_timestamp:
            try:
                values.append(datetime.fromisoformat(str(raw_timestamp)).date())
            except ValueError:
                continue
    # The fallback keeps the UI usable if a custom engine does not expose messages.
    return (min(values), max(values)) if values else (date(2026, 3, 1), date(2026, 8, 31))


def context_records(context: Any) -> list[dict[str, Any]]:
    """Normalise a list or common dictionary-shaped context response."""

    if isinstance(context, list):
        return [item for item in context if isinstance(item, dict)]
    if not isinstance(context, dict):
        return []
    for key in ("messages", "context"):
        value = context.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]

    records: list[dict[str, Any]] = []
    for key in ("before", "match", "message", "after"):
        value = context.get(key)
        if isinstance(value, list):
            records.extend(item for item in value if isinstance(item, dict))
        elif isinstance(value, dict):
            records.append(value)
    return records


def fetch_results(
    engine: ChatSearchEngine,
    query: str,
    top_k: int,
    sender: str | None,
    date_from: date | None,
    date_to: date | None,
    mode: str,
) -> list[dict[str, Any]]:
    """Search with the current API and a small backwards-compatible fallback."""

    common = {"top_k": top_k, "sender": sender, "mode": mode}
    try:
        return engine.search(
            query,
            date_from=date_from,
            date_to=date_to,
            **common,
        )
    except TypeError as error:
        # Older local versions used start_date/end_date. Retry only for that API
        # mismatch; other TypeErrors should remain visible to the user.
        if "date_from" not in str(error) and "date_to" not in str(error):
            raise
        return engine.search(
            query,
            start_date=date_from,
            end_date=date_to,
            **common,
        )


def render_context(engine: ChatSearchEngine, result: dict[str, Any]) -> None:
    """Display nearby messages so a match is never presented in isolation."""

    result_id = message_id(result)
    supplied_context = result.get("context")
    if supplied_context is None and result_id is not None:
        supplied_context = engine.get_context(result_id, before=2, after=2)
    records = context_records(supplied_context)

    if not records:
        st.caption("No surrounding messages are available for this result.")
        return

    for record in records:
        is_match = message_id(record) == result_id
        label = "↳ Match — " if is_match else ""
        st.markdown(
            f"**{label}{record.get('sender', 'Unknown')}** · "
            f"{safe_format_timestamp(record.get('timestamp'))}"
        )
        st.write(record.get("text", ""))


def main() -> None:
    st.title("ChatSearch")
    st.caption("Search your group chat by meaning, person, and time.")

    try:
        engine = load_engine()
    except FileNotFoundError:
        st.error(
            "The local embedding index is missing. Run `python src/search_engine.py` "
            "once from the project root, then restart Streamlit."
        )
        st.stop()
    except Exception as error:  # pragma: no cover - defensive UI boundary
        st.error(f"ChatSearch could not load its local index: {error}")
        st.stop()

    participants = list(getattr(engine, "participants", []))
    if not participants and hasattr(engine, "list_participants"):
        participants = list(engine.list_participants())
    participants = sorted(set(participants))
    earliest, latest = chat_date_bounds(engine)

    with st.form("search_form", clear_on_submit=False):
        query = st.text_input(
            "What are you looking for?",
            placeholder="For example: When did we decide on Manali?",
        )
        mode_label = st.radio(
            "Search mode",
            ("Semantic", "Person + Topic", "Temporal"),
            horizontal=True,
            help=(
                "Semantic searches by meaning; Person + Topic prioritises a "
                "speaker; Temporal understands phrases such as 'last month' and 'in July'."
            ),
        )
        if mode_label == "Temporal":
            st.caption(
                "Relative dates use the synthetic reference date "
                f"{TEMPORAL_REFERENCE_DATE.day} "
                f"{TEMPORAL_REFERENCE_DATE.strftime('%B %Y')} "
                "(the day after this archive ends)."
            )

        controls_left, controls_right = st.columns(2)
        with controls_left:
            selected_participant = st.selectbox(
                "Participant (optional)",
                ["Auto-detect / all participants", *participants],
                help="In Person + Topic mode, leave this on auto-detect if the name is in your question.",
            )
        with controls_right:
            result_count = st.slider("Number of results", min_value=1, max_value=10, value=5)

        use_date_range = st.checkbox("Filter to a specific date range")
        date_from: date | None = None
        date_to: date | None = None
        if use_date_range:
            chosen_dates = st.date_input(
                "Date range",
                value=(earliest, latest),
                min_value=earliest,
                max_value=latest,
            )
            # Streamlit can return a one-item tuple while a user is editing the
            # range, so defer validation instead of raising during a rerun.
            if isinstance(chosen_dates, tuple) and len(chosen_dates) == 2:
                date_from, date_to = chosen_dates

        submitted = st.form_submit_button("Search", type="primary")

    if not submitted:
        with st.expander("Try an example", expanded=False):
            st.write("• When did we decide on the Manali trip?")
            st.write("• What did Priya say about the budget?")
            st.write("• What did we discuss last month?")
        return

    cleaned_query = query.strip()
    if not cleaned_query:
        st.warning("Type a question before searching.")
        return
    if use_date_range:
        if date_from is None or date_to is None:
            st.warning("Choose both a start date and an end date.")
            return
        if date_from > date_to:
            st.warning("The start date must be on or before the end date.")
            return

    mode = {
        "Semantic": "semantic",
        "Person + Topic": "attributed",
        "Temporal": "temporal",
    }[mode_label]
    sender = None if selected_participant.startswith("Auto-detect") else selected_participant

    try:
        with st.spinner("Searching the local chat…"):
            results = fetch_results(
                engine,
                cleaned_query,
                result_count,
                sender,
                date_from,
                date_to,
                mode,
            )
    except ValueError as error:
        st.warning(str(error))
        return
    except Exception as error:  # pragma: no cover - defensive UI boundary
        st.error(f"Search failed: {error}")
        return

    if not results:
        st.info("No messages matched those filters. Try a broader question or remove a filter.")
        return

    st.subheader(f"Results ({len(results)})")
    for rank, result in enumerate(results, start=1):
        score = result.get("score")
        score_text = f"{float(score):.3f}" if isinstance(score, (float, int)) else "n/a"
        thread = result.get("thread") or "general chat"
        with st.expander(f"#{rank} · {result.get('sender', 'Unknown')} · score {score_text}", expanded=rank == 1):
            st.caption(f"{safe_format_timestamp(result.get('timestamp'))} · Thread: {thread}")
            st.write(result.get("text", ""))
            st.markdown("**Conversation context**")
            render_context(engine, result)


if __name__ == "__main__":
    main()
