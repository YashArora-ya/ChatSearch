"""End-to-end checks for the deterministic ChatSearch demo.

The fixture deliberately creates one engine for the entire test session.  This
matches the application's performance goal: load the saved index once and only
embed individual queries while searching.
"""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import date, datetime, time
from pathlib import Path

# Pytest imports NumPy below.  Use the same safe default as the app so test
# runs remain reliable on small Windows machines.
for _thread_variable in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_thread_variable, "1")

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data_generator import PARTICIPANTS, TOTAL_MESSAGES, generate_dataset  # noqa: E402
from search_engine import (  # noqa: E402
    ChatSearchEngine,
    LEXICAL_WEIGHT,
    SEMANTIC_WEIGHT,
    TEMPORAL_REFERENCE_DATE,
)


CHAT_PATH = ROOT / "data" / "chat.json"
QUERIES_PATH = ROOT / "data" / "queries.json"
EMBEDDINGS_PATH = ROOT / "data" / "embeddings.npy"

# Relative phrases must be reproducible for this synthetic archive.  This is
# deliberately a calendar date just after the six-month dataset ends, not the
# computer's clock or the archive's latest timestamp.
EXPECTED_TEMPORAL_REFERENCE_DATE = date(2026, 9, 1)


def _load_json(path: Path):
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _result_id(result: dict) -> int:
    """Accept the JSON field name and the convenient UI alias."""

    if "message_id" in result:
        return int(result["message_id"])
    return int(result["id"])


def _query_by_id(query_id: int) -> dict:
    return next(query for query in _load_json(QUERIES_PATH) if query["id"] == query_id)


def _lexical_tokens(text: str) -> set[str]:
    """Return literal, case-insensitive word tokens for overlap assertions."""

    return set(re.findall(r"\w+", str(text).casefold(), flags=re.UNICODE))


@pytest.fixture(scope="session")
def messages() -> list[dict]:
    return _load_json(CHAT_PATH)


@pytest.fixture(scope="session")
def queries() -> list[dict]:
    return _load_json(QUERIES_PATH)


@pytest.fixture(scope="session")
def engine() -> ChatSearchEngine:
    # The checked-in/generated index must be reused; ChatSearchEngine should
    # build only when the file is missing or incompatible.
    return ChatSearchEngine(auto_build=True)


def test_dataset_has_required_shape_and_known_decisions(messages: list[dict]) -> None:
    assert len(messages) >= 4_000
    assert len(messages) == TOTAL_MESSAGES + 56
    assert {message["sender"] for message in messages} == set(PARTICIPANTS)
    assert [message["message_id"] for message in messages] == list(range(1, len(messages) + 1))

    timestamps = [datetime.fromisoformat(message["timestamp"]) for message in messages]
    assert timestamps == sorted(timestamps)
    assert timestamps[0].date() >= date(2026, 3, 1)
    assert timestamps[-1].date() <= date(2026, 8, 31)

    threads = {message["thread"] for message in messages if message["thread"]}
    assert {"manali_decision", "birthday_decision", "project_decision"} <= threads
    assert any(
        message["text"] == "Chalo, Manali fix hai. Main hotel dekh leta hu."
        and message["thread"] == "manali_decision"
        for message in messages
    )
    assert sum(message["text"].lower().startswith("forwarded:") for message in messages) > 0
    assert sum(len(message["text"].split()) <= 2 for message in messages) > 100


def test_dataset_generator_is_reproducible() -> None:
    first = generate_dataset()
    second = generate_dataset()
    assert len(first) == len(second) == TOTAL_MESSAGES + 56
    assert first == second


def test_evaluation_queries_are_complete_and_resolvable(
    messages: list[dict], queries: list[dict]
) -> None:
    message_by_id = {message["message_id"]: message for message in messages}
    assert len(queries) == 40
    assert [query["id"] for query in queries] == list(range(1, 41))
    assert {query["type"] for query in queries} == {"semantic", "attributed", "temporal"}
    assert sum(query["type"] == "semantic" for query in queries) >= 8
    assert sum(query["type"] == "attributed" for query in queries) >= 8
    assert sum(query["type"] == "temporal" for query in queries) >= 8
    assert sum(bool(query.get("zero_word_overlap")) for query in queries) >= 8
    assert TEMPORAL_REFERENCE_DATE == EXPECTED_TEMPORAL_REFERENCE_DATE

    for query in queries:
        assert query["expected_message_id"] in message_by_id
        # Every labelled query carries the same deterministic clock so
        # evaluation never changes with the developer's machine date.
        assert query.get("reference_date") == TEMPORAL_REFERENCE_DATE.isoformat()
        if query["type"] == "attributed":
            assert query["sender"] in PARTICIPANTS
            assert message_by_id[query["expected_message_id"]]["sender"] == query["sender"]
        if query["type"] == "temporal":
            assert "date_from" in query and "date_to" in query


def test_zero_word_overlap_queries_are_lexically_disjoint(
    messages: list[dict], queries: list[dict]
) -> None:
    """Keep the semantic benchmark honest: flagged pairs share no literal words."""

    message_by_id = {message["message_id"]: message for message in messages}
    zero_overlap_queries = [query for query in queries if query.get("zero_word_overlap")]
    assert len(zero_overlap_queries) >= 8

    for query in zero_overlap_queries:
        answer = message_by_id[query["expected_message_id"]]
        shared_words = _lexical_tokens(query["query"]) & _lexical_tokens(answer["text"])
        assert not shared_words, (
            f"Query #{query['id']} is marked zero_word_overlap but shares "
            f"literal words with its answer: {sorted(shared_words)}"
        )


def test_embeddings_are_float32_and_aligned(messages: list[dict]) -> None:
    embeddings = np.load(EMBEDDINGS_PATH, mmap_mode="r")
    assert embeddings.dtype == np.float32
    assert embeddings.shape[0] == len(messages)
    assert embeddings.ndim == 2 and embeddings.shape[1] > 0
    assert np.isfinite(embeddings[:10]).all()


def test_engine_loads_existing_index(engine: ChatSearchEngine, messages: list[dict]) -> None:
    assert len(engine.messages) == len(messages)
    assert engine.embeddings.dtype == np.float32
    assert engine.embeddings.shape[0] == len(messages)
    assert set(engine.list_participants()) == set(PARTICIPANTS)


def test_semantic_search_finds_manali_decision(engine: ChatSearchEngine) -> None:
    query = _query_by_id(1)
    results = engine.search(query["query"], top_k=5, mode="semantic")
    assert results
    assert query["expected_message_id"] in {_result_id(result) for result in results}
    assert all({"score", "timestamp", "sender", "text", "thread"} <= result.keys() for result in results)


def test_hybrid_scores_are_normalized_and_semantic_first(engine: ChatSearchEngine) -> None:
    """Hybrid ranking must retain a semantic-majority, bounded base score."""

    assert SEMANTIC_WEIGHT > LEXICAL_WEIGHT
    assert SEMANTIC_WEIGHT + LEXICAL_WEIGHT == pytest.approx(1.0)

    results = engine.search("Manali hotel decision", top_k=3, mode="semantic")
    assert results[0]["message_id"] == _query_by_id(1)["expected_message_id"]
    for result in results:
        assert 0.0 <= result["semantic_score"] <= 1.0
        assert 0.0 <= result["lexical_score"] <= 1.0
        assert result["hybrid_score"] == pytest.approx(
            SEMANTIC_WEIGHT * result["semantic_score"]
            + LEXICAL_WEIGHT * result["lexical_score"]
        )
    assert results[0]["lexical_score"] > 0.0


@pytest.mark.parametrize(
    ("query", "expected_query_id"),
    [
        ("Manali ka plan pakka hua kya?", 1),
        ("Priya ne budget kitna btaya?", 16),
    ],
)
def test_hinglish_queries_retrieve_the_expected_messages(
    engine: ChatSearchEngine, query: str, expected_query_id: int
) -> None:
    """Roman-Hindi/code-mixed wording should find the marked decision/answer."""

    expected_id = _query_by_id(expected_query_id)["expected_message_id"]
    mode = "attributed" if expected_query_id == 16 else "semantic"
    results = engine.search(query, top_k=3, mode=mode)
    assert results
    assert _result_id(results[0]) == expected_id


def test_typo_query_still_retrieves_manali_decision(engine: ChatSearchEngine) -> None:
    """Character features must tolerate misspellings in a meaningful query."""

    expected_id = _query_by_id(1)["expected_message_id"]
    results = engine.search("Mnaali trip kab fnal hui?", top_k=3, mode="semantic")
    assert results
    assert _result_id(results[0]) == expected_id


def test_sender_filter_and_invalid_participant(engine: ChatSearchEngine) -> None:
    query = _query_by_id(16)
    results = engine.search(query["query"], top_k=5, sender="Priya", mode="attributed")
    assert results
    assert all(result["sender"] == "Priya" for result in results)
    assert query["expected_message_id"] in {_result_id(result) for result in results}
    assert engine.search("budget", top_k=5, sender="Nobody McNobody") == []


def test_temporal_filter_respects_an_explicit_date_range(engine: ChatSearchEngine) -> None:
    query = _query_by_id(27)
    results = engine.search(
        query["query"],
        top_k=5,
        date_from=query["date_from"],
        date_to=query["date_to"],
        mode="temporal",
    )
    assert results
    assert query["expected_message_id"] in {_result_id(result) for result in results}
    for result in results:
        result_date = datetime.fromisoformat(result["timestamp"]).date()
        assert date.fromisoformat(query["date_from"]) <= result_date <= date.fromisoformat(query["date_to"])



@pytest.mark.parametrize(
    ("phrase", "expected_start", "expected_end"),
    [
        (
            "last month",
            datetime(2026, 8, 1, 0, 0, 0),
            datetime.combine(date(2026, 8, 31), time.max),
        ),
        (
            "this month",
            datetime(2026, 9, 1, 0, 0, 0),
            datetime.combine(date(2026, 9, 30), time.max),
        ),
        (
            "last week",
            datetime(2026, 8, 24, 0, 0, 0),
            datetime.combine(date(2026, 8, 30), time.max),
        ),
        (
            "this week",
            datetime(2026, 8, 31, 0, 0, 0),
            datetime.combine(date(2026, 9, 6), time.max),
        ),
    ],
)
def test_relative_temporal_parser_uses_full_calendar_ranges(
    engine: ChatSearchEngine,
    phrase: str,
    expected_start: datetime,
    expected_end: datetime,
) -> None:
    """Guard against relative dates drifting to the archive latest timestamp."""

    bounds = engine.parse_temporal_query(f"What did we discuss {phrase}?")
    assert bounds is not None
    assert bounds.start == expected_start
    assert bounds.end == expected_end


@pytest.mark.parametrize(
    ("phrase", "expected_start", "expected_end", "expect_results"),
    [
        ("last month", date(2026, 8, 1), date(2026, 8, 31), True),
        ("this month", date(2026, 9, 1), date(2026, 9, 30), False),
        ("last week", date(2026, 8, 24), date(2026, 8, 30), True),
        ("this week", date(2026, 8, 31), date(2026, 9, 6), True),
    ],
)
def test_relative_temporal_search_restricts_candidates_to_parsed_range(
    engine: ChatSearchEngine,
    phrase: str,
    expected_start: date,
    expected_end: date,
    expect_results: bool,
) -> None:
    """The range parser must be applied before ranking temporal candidates."""

    results = engine.search(f"What did we discuss {phrase}?", top_k=5, mode="temporal")
    if not expect_results:
        # The synthetic archive ends on 31 August, so September has no rows.
        assert results == []
        return

    assert results
    for result in results:
        result_date = datetime.fromisoformat(result["timestamp"]).date()
        assert expected_start <= result_date <= expected_end


def test_context_contains_adjacent_messages(engine: ChatSearchEngine) -> None:
    expected_id = _query_by_id(1)["expected_message_id"]
    context = engine.get_context(expected_id, before=2, after=2)
    assert isinstance(context, list)
    assert _result_id(context[0]) == expected_id - 2
    assert _result_id(context[-1]) == expected_id + 2
    assert expected_id in {_result_id(message) for message in context}


def test_empty_query_and_top_k_ranking(engine: ChatSearchEngine) -> None:
    assert engine.search("", top_k=5) == []
    assert engine.search("   ", top_k=5) == []

    results = engine.search("Manali hotel decision", top_k=3)
    assert len(results) == 3
    scores = [result["score"] for result in results]
    assert scores == sorted(scores, reverse=True)


def test_evaluation_queries_run_and_report_metrics(engine: ChatSearchEngine) -> None:
    report = engine.evaluate_queries(QUERIES_PATH, top_k=5)
    assert report["total_queries"] == 40
    for key in ("top_1_accuracy", "top_3_accuracy", "top_5_accuracy"):
        assert 0.0 <= report[key] <= 1.0
    assert report["top_1_accuracy"] <= report["top_3_accuracy"] <= report["top_5_accuracy"]
    assert isinstance(report["failed_queries"], list)
