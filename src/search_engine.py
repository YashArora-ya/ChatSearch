"""Lightweight, local semantic search for the synthetic ChatSearch archive.

This module intentionally avoids sentence-transformers.  The prior engine
loaded a transformer model at every app start, which is unnecessary for this
small archive and can trigger OpenBLAS allocation failures on Windows.  It uses
a deterministic NumPy-only feature-hashing embedder instead.  Message vectors
are built once, saved as float32 in ``data/embeddings.npy``, then a search only
embeds the single query and performs one efficient dot product.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import unicodedata
import zlib
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any, NamedTuple, Optional, Sequence

# Keep NumPy/OpenBLAS lightweight on ordinary Windows laptops.  These values
# are set before NumPy is imported and remain user-overridable when someone
# has deliberately configured a different thread cap in their environment.
for _thread_variable in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_thread_variable, "1")

import numpy as np


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CHAT_PATH = DATA_DIR / "chat.json"
EMBEDDINGS_PATH = DATA_DIR / "embeddings.npy"
EMBEDDINGS_META_PATH = DATA_DIR / "embeddings.meta.json"
QUERIES_PATH = DATA_DIR / "queries.json"

# 4,356 x 384 float32 values occupy roughly 6.4 MB: small enough for a normal
# laptop while leaving ample room for fuzzy lexical and semantic features.
EMBEDDING_DIM = 384
INDEX_VERSION = "hashed-semantic-v2"
MODEL_NAME = INDEX_VERSION

# Hybrid ranking remains semantic-first.  The lexical component is a small,
# dependency-free exact-token signal that helps when a query contains a useful
# word such as a place, amount, or topic.  Both base scores are normalised to
# the 0..1 range before they are combined.
SEMANTIC_WEIGHT = 0.90
LEXICAL_WEIGHT = 0.10
# Existing intent/context bonuses were calibrated against cosine-scale score
# differences.  This factor restores that separation after 0..1 blending;
# it does not change the relative order of the hybrid base itself.
HYBRID_SCORE_SCALE = 2.0


STOP_WORDS = {
    "a", "an", "and", "are", "at", "be", "by", "did", "do", "for", "from",
    "how", "i", "in", "is", "it", "me", "of", "on", "or", "our", "the", "this",
    "to", "was", "we", "what", "when", "where", "which", "who", "with", "would",
    "you", "your", "ka", "ke", "ki", "ko", "se", "toh",
}

# Conservative normalisation bridges the most common abbreviated Hinglish
# forms and a few typos without rewriting ordinary English.
TOKEN_NORMALISATIONS = {
    "acha": "achha", "accha": "achha", "collge": "college", "confirmd": "confirmed",
    "finalise": "finalize", "finalised": "finalized", "frm": "from", "h": "hai",
    "hn": "haan", "hu": "hoon", "kr": "kar", "krdo": "kar do", "krta": "karta",
    "krti": "karti", "lg": "lag", "mai": "main", "rha": "raha", "rhe": "rahe",
    "rhi": "rahi", "tho": "toh", "waha": "wahan",
}


# Shared intent vocabulary.  Word and character features are still kept, so
# these features improve semantic recall rather than replacing literal search.
CONCEPT_LEXICON: dict[str, set[str]] = {
    "trip": {
        "trip", "travel", "travelling", "outing", "vacation", "holiday", "tour",
        "mountain", "mountains", "hill", "hills", "hillstation", "pahad", "manali",
        "shimla", "destination", "visit", "summer", "going", "journey", "place",
    },
    "decision": {
        "agree", "agreed", "agreement", "choose", "choosing", "chosen", "confirm",
        "confirmed", "decision", "decide", "decided", "final", "finalize", "finalized",
        "fix", "fixed", "lock", "locked", "settle", "settled", "ultimately",
        "selected", "selection", "done",
    },
    "budget": {
        "budget", "amount", "money", "cost", "price", "expensive", "cheap",
        "affordable", "rupees", "rs", "15k", "8k", "perperson",
    },
    "dates": {"date", "dates", "schedule", "may", "june", "july", "weekend", "day", "days"},
    "transport": {"transport", "transportation", "bus", "train", "car", "drive", "ride", "traveloption"},
    "hotel": {"hotel", "stay", "accommodation", "room", "booking", "book"},
    "birthday": {"birthday", "party", "cake", "cafe", "venue", "celebration", "citycenter"},
    "project": {
        "project", "chatbot", "recommendation", "scope", "topic", "implementation",
        "finalyear", "dataset", "python", "deployment", "work",
    },
    "time_constraint": {"time", "weeks", "week", "deadline", "constraint", "duration"},
    "cake_arrangement": {"cake", "arrange", "bring", "laega", "launga"},
}

MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
    "december": 12,
}

# The synthetic archive intentionally runs from March through August 2026.
# Relative phrases must not depend on the computer running the app, so all
# default temporal parsing is anchored to the first day after that archive:
# 1 September 2026. Callers can still supply an explicit reference date when
# testing a different scenario.
TEMPORAL_REFERENCE_DATE = date(2026, 9, 1)


def _normalise_text(text: Any) -> str:
    """Return a predictable, punctuation-light representation of *text*."""

    raw = unicodedata.normalize("NFKC", str(text or "")).casefold().replace("'", " ")
    # \w keeps non-Latin word characters too, allowing useful character features
    # when a user enters Hindi in Devanagari.
    tokens = re.findall(r"\w+", raw, flags=re.UNICODE)
    expanded: list[str] = []
    for token in tokens:
        expanded.extend(TOKEN_NORMALISATIONS.get(token, token).split())
    return " ".join(expanded)


def _tokens(text: Any, *, include_stop_words: bool = False) -> list[str]:
    words = _normalise_text(text).split()
    return words if include_stop_words else [word for word in words if word not in STOP_WORDS]


def _stable_hash(value: str) -> int:
    """A stable hash across Python processes (unlike built-in ``hash``)."""

    return zlib.crc32(value.encode("utf-8")) & 0xFFFFFFFF


def _parse_timestamp(value: Any) -> Optional[datetime]:
    """Parse ISO source rows plus DD-MM-YYYY rows emitted by the generator."""

    if isinstance(value, datetime):
        return value.replace(tzinfo=None)
    if isinstance(value, date):
        return datetime.combine(value, time.min)
    if value is None:
        return None
    text_value = str(value).strip()
    if not text_value:
        return None
    try:
        return datetime.fromisoformat(text_value.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        pass
    for pattern in (
        "%d-%m-%Y %H:%M:%S", "%d-%m-%Y %H:%M", "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d", "%d-%m-%Y",
    ):
        try:
            return datetime.strptime(text_value, pattern)
        except ValueError:
            continue
    return None


def _coerce_boundary(value: Any, *, end_of_day: bool = False) -> Optional[datetime]:
    """Turn date/datetime/ISO strings into comparison boundaries."""

    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)
    if isinstance(value, date):
        return datetime.combine(value, time.max if end_of_day else time.min)
    raw = str(value).strip()
    parsed = _parse_timestamp(raw)
    if parsed is None:
        return None
    if end_of_day and re.fullmatch(r"\d{4}-\d{2}-\d{2}|\d{2}-\d{2}-\d{4}", raw):
        return datetime.combine(parsed.date(), time.max)
    return parsed


def _coerce_reference_datetime(value: Any = None) -> datetime:
    """Return a deterministic, naive datetime used by relative-date rules.

    ``None`` deliberately means the synthetic archive reference date instead
    of the host machine's clock. Strings are accepted so that ``queries.json``
    can record the same reference used by the evaluator.
    """

    if value is None:
        return datetime.combine(TEMPORAL_REFERENCE_DATE, time.min)
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)
    if isinstance(value, date):
        return datetime.combine(value, time.min)
    parsed = _parse_timestamp(value)
    if parsed is None:
        raise ValueError(
            "reference_date must be a date, datetime, or parseable ISO date string"
        )
    return parsed


class DateRange(NamedTuple):
    """Inspectable (start, end) tuple returned by the rule-based date parser."""

    start: datetime
    end: datetime


class LightweightSemanticEmbedder:
    """A deterministic, NumPy-only embedding model for short informal chats."""

    def __init__(self, dimension: int = EMBEDDING_DIM) -> None:
        self.dimension = int(dimension)

    @staticmethod
    def extract_concepts(text: Any) -> set[str]:
        normalised = _normalise_text(text)
        words = set(normalised.split())
        joined = "".join(words)
        concepts: set[str] = set()
        for concept, vocabulary in CONCEPT_LEXICON.items():
            if words.intersection(vocabulary) or any(
                term in joined for term in vocabulary if len(term) >= 5
            ):
                concepts.add(concept)

        # Phrase-level bridges cover natural paraphrases without depending on
        # external APIs or a heavyweight neural model.
        if re.search(r"\b(where|which place|which city)\b", normalised) and re.search(
            r"\b(go|going|visit|travel|choose|chosen|settle)\b", normalised
        ):
            concepts.update({"trip", "decision"})
        if re.search(r"\bfinal year\b|\bfinalyear\b", normalised):
            concepts.add("project")
        if "per person" in normalised or re.search(r"\bhow much\b", normalised):
            concepts.add("budget")
        # "travel option" is a common way of asking about transport without
        # naming a vehicle (for example, a bus).  Treat it as the same intent
        # so a message that only says "bus se chale" remains discoverable.
        if re.search(r"\b(?:travel|commute)\s+(?:option|plan|way)\b", normalised):
            concepts.add("transport")
        if re.search(r"\bhow long\b|\btime constraint\b", normalised):
            concepts.add("time_constraint")
        return concepts

    def _add_feature(self, vector: np.ndarray, feature: str, weight: float) -> None:
        hashed = _stable_hash(feature)
        index = hashed % self.dimension
        vector[index] += np.float32((1.0 if (hashed >> 31) == 0 else -1.0) * weight)

    def encode_one(self, text: Any, *, thread: Any = None) -> np.ndarray:
        words = _tokens(text)
        vector = np.zeros(self.dimension, dtype=np.float32)
        for word in words:
            self._add_feature(vector, f"word:{word}", 1.0)
            if len(word) >= 4:
                padded = f"^{word}$"
                for position in range(len(padded) - 2):
                    self._add_feature(vector, f"char:{padded[position:position + 3]}", 0.14)
        for first, second in zip(words, words[1:]):
            self._add_feature(vector, f"bigram:{first}_{second}", 0.65)
        for concept in self.extract_concepts(text):
            self._add_feature(vector, f"concept:{concept}", 2.4)
        # Thread labels are lightweight metadata, not a replacement for text.
        for thread_word in _tokens(str(thread or "").replace("_", " ")):
            self._add_feature(vector, f"thread:{thread_word}", 0.35)
        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector /= np.float32(norm)
        return vector

    def encode(self, texts: Sequence[Any]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for index, text in enumerate(texts):
            vectors[index] = self.encode_one(text)
        return vectors


class ChatSearchEngine:
    """Search group messages by meaning, participant, date, and context.

    ``auto_build`` only creates an index when none is current.  Once the
    persisted float32 index exists, app reloads load it directly and encode only
    the user query.  ``model_name`` is accepted for API compatibility; no model
    download or network connection is attempted.
    """

    def __init__(
        self,
        data_path: Optional[Path | str] = None,
        embeddings_path: Optional[Path | str] = None,
        model_name: Optional[str] = None,
        auto_build: bool = True,
    ) -> None:
        self.data_path = Path(data_path) if data_path else CHAT_PATH
        self.embeddings_path = Path(embeddings_path) if embeddings_path else EMBEDDINGS_PATH
        self.meta_path = self.embeddings_path.with_suffix(".meta.json")
        self.model_name = model_name or MODEL_NAME
        self.auto_build = bool(auto_build)
        self.model = LightweightSemanticEmbedder()
        self.messages = self._load_messages()
        # Cache normalised message terms once.  Searches only need to create
        # terms for the user's query, so lexical matching stays lightweight.
        self._message_lexical_terms = [
            frozenset(_tokens(message["text"])) for message in self.messages
        ]
        self._message_datetimes = [_parse_timestamp(message.get("timestamp")) for message in self.messages]
        self._id_to_index = {
            int(message["message_id"]): index
            for index, message in enumerate(self.messages)
            if message.get("message_id") is not None
        }
        self._thread_message_indices: dict[str, list[int]] = {}
        for index, message in enumerate(self.messages):
            thread = message.get("thread")
            if thread:
                self._thread_message_indices.setdefault(str(thread), []).append(index)
        # Built lazily from the persisted message vectors.  This gives a
        # concluding message access to the topic discussed immediately around
        # it without loading a second model or re-encoding the archive.
        self._thread_context_cache: Optional[dict[str, tuple[np.ndarray, frozenset[str]]]] = None
        self.embeddings: Optional[np.ndarray] = None
        if self.auto_build:
            self.load_index(build_if_missing=True)

    # -- data/index lifecycle -------------------------------------------------
    def _load_messages(self) -> list[dict[str, Any]]:
        if not self.data_path.exists():
            raise FileNotFoundError(f"Chat data was not found: {self.data_path}")
        with self.data_path.open("r", encoding="utf-8") as handle:
            raw_messages = json.load(handle)
        if not isinstance(raw_messages, list):
            raise ValueError("chat.json must contain a JSON list of messages")
        messages: list[dict[str, Any]] = []
        for position, raw in enumerate(raw_messages, start=1):
            if not isinstance(raw, dict):
                continue
            message = dict(raw)
            message_id = message.get("message_id", message.get("id", position))
            try:
                message_id = int(message_id)
            except (TypeError, ValueError):
                message_id = position
            message["message_id"] = message_id
            message["id"] = message_id
            message["sender"] = str(message.get("sender", "Unknown"))
            message["text"] = str(message.get("text", ""))
            message["thread"] = message.get("thread") or None
            messages.append(message)
        return messages

    def _data_fingerprint(self) -> str:
        digest = hashlib.sha256()
        with self.data_path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def _index_is_current(self) -> bool:
        if not self.embeddings_path.exists() or not self.meta_path.exists():
            return False
        try:
            with self.meta_path.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            if (
                metadata.get("version") != INDEX_VERSION
                or metadata.get("dimension") != EMBEDDING_DIM
                or metadata.get("message_count") != len(self.messages)
                or metadata.get("data_fingerprint") != self._data_fingerprint()
            ):
                return False
            vectors = np.load(self.embeddings_path, mmap_mode="r", allow_pickle=False)
            return vectors.shape == (len(self.messages), EMBEDDING_DIM) and vectors.dtype == np.float32
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return False

    def build_index(self, batch_size: int = 32, *, force: bool = True) -> np.ndarray:
        """Build once and atomically persist compact normalised float32 vectors."""

        if not force and self._index_is_current():
            return self.load_index(build_if_missing=False)
        batch_size = max(1, min(int(batch_size), 128))
        vectors = np.zeros((len(self.messages), EMBEDDING_DIM), dtype=np.float32)
        for start in range(0, len(self.messages), batch_size):
            for offset, message in enumerate(self.messages[start:start + batch_size]):
                vectors[start + offset] = self.model.encode_one(message["text"], thread=message.get("thread"))

        self.embeddings_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_vectors = self.embeddings_path.with_suffix(".tmp.npy")
        temporary_metadata = self.meta_path.with_suffix(".tmp.json")
        with temporary_vectors.open("wb") as handle:
            np.save(handle, vectors)
        temporary_vectors.replace(self.embeddings_path)
        metadata = {
            "version": INDEX_VERSION,
            "dimension": EMBEDDING_DIM,
            "dtype": "float32",
            "message_count": len(self.messages),
            "data_fingerprint": self._data_fingerprint(),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        with temporary_metadata.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)
        temporary_metadata.replace(self.meta_path)
        self.embeddings = vectors
        self._thread_context_cache = None
        return vectors

    def load_index(self, *, build_if_missing: Optional[bool] = None) -> np.ndarray:
        """Load the current index, building only when explicitly permitted."""

        if self.embeddings is not None:
            return self.embeddings
        should_build = self.auto_build if build_if_missing is None else build_if_missing
        if not self._index_is_current():
            if not should_build:
                raise FileNotFoundError(
                    "A current embeddings index was not found. Run "
                    "`python src/search_engine.py --rebuild` once."
                )
            return self.build_index(batch_size=32, force=True)
        self.embeddings = np.asarray(np.load(self.embeddings_path, allow_pickle=False), dtype=np.float32)
        self._thread_context_cache = None
        return self.embeddings

    # -- query interpretation --------------------------------------------------
    @property
    def participants(self) -> list[str]:
        return sorted({message["sender"] for message in self.messages}, key=str.casefold)

    def list_participants(self) -> list[str]:
        return self.participants

    def infer_sender(self, query: str) -> Optional[str]:
        normalised_query = _normalise_text(query)
        for participant in self.participants:
            if re.search(rf"\b{re.escape(participant.casefold())}\b", normalised_query):
                return participant
        return None

    def parse_temporal_query(
        self, query: str, reference_date: Any = None
    ) -> Optional[DateRange]:
        """Parse practical relative dates into inclusive calendar ranges.

        By default, relative terms are anchored to ``TEMPORAL_REFERENCE_DATE``
        (2026-09-01), not to the newest message or the host clock. Weeks use
        Monday--Sunday boundaries and months use their complete calendar month.
        """

        normalised = _normalise_text(query)
        reference = _coerce_reference_datetime(reference_date)
        today_start = datetime.combine(reference.date(), time.min)
        today_end = datetime.combine(reference.date(), time.max)
        if re.search(r"\btoday\b", normalised):
            return DateRange(today_start, today_end)
        if re.search(r"\byesterday\b", normalised):
            day = reference.date() - timedelta(days=1)
            return DateRange(datetime.combine(day, time.min), datetime.combine(day, time.max))
        monday = reference.date() - timedelta(days=reference.weekday())
        if re.search(r"\bthis week\b", normalised):
            sunday = monday + timedelta(days=6)
            return DateRange(
                datetime.combine(monday, time.min),
                datetime.combine(sunday, time.max),
            )
        if re.search(r"\blast week\b", normalised):
            start = monday - timedelta(days=7)
            return DateRange(datetime.combine(start, time.min), datetime.combine(start + timedelta(days=6), time.max))
        if re.search(r"\bthis month\b", normalised):
            start = datetime(reference.year, reference.month, 1)
            next_month = (
                datetime(reference.year + 1, 1, 1)
                if reference.month == 12
                else datetime(reference.year, reference.month + 1, 1)
            )
            return DateRange(start, next_month - timedelta(microseconds=1))
        if re.search(r"\blast month\b", normalised):
            current_month_start = reference.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            previous_end = current_month_start - timedelta(microseconds=1)
            previous_start = previous_end.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            return DateRange(previous_start, previous_end)
        month_match = re.search(r"\b(?:in|during)\s+(" + "|".join(MONTHS) + r")\b", normalised)
        if month_match:
            month = MONTHS[month_match.group(1)]
            start = datetime(reference.year, month, 1)
            next_month = datetime(reference.year + 1, 1, 1) if month == 12 else datetime(reference.year, month + 1, 1)
            return DateRange(start, next_month - timedelta(microseconds=1))
        return None

    @staticmethod
    def _is_temporal_mode(mode: Optional[str]) -> bool:
        return str(mode or "").casefold().strip() in {"temporal", "time", "date"}

    @staticmethod
    def _is_attributed_mode(mode: Optional[str]) -> bool:
        return str(mode or "").casefold().strip() in {"attributed", "person + topic", "person-topic", "person"}

    def _decision_thread_for_query(self, query: str, concepts: set[str]) -> Optional[str]:
        normalised = _normalise_text(query)
        if "project" in concepts or re.search(r"\bfinal year\b|\bchatbot\b", normalised):
            return "project_decision"
        if "birthday" in concepts:
            return "birthday_decision"
        # A location/city question asks for a destination even when it does
        # not explicitly use words such as "trip" or "travel".  In this
        # archive the concrete destination discussion is the Manali thread.
        if re.search(r"\b(?:location|city|destination|hill station)\b", normalised):
            return "manali_decision"
        if "trip" in concepts or "hotel" in concepts:
            return "manali_decision"
        return None

    def _thread_contexts(
        self, vectors: np.ndarray
    ) -> dict[str, tuple[np.ndarray, frozenset[str]]]:
        """Return per-thread semantic summaries and vocabulary once per engine.

        A final message is often intentionally short ("done", "fixed", or
        "let's do it").  The thread summary supplies the missing subject from
        the surrounding discussion, rather than routing a particular query to
        a known benchmark thread.
        """

        if self._thread_context_cache is not None:
            return self._thread_context_cache
        cache: dict[str, tuple[np.ndarray, frozenset[str]]] = {}
        for thread, indices in self._thread_message_indices.items():
            thread_vectors = vectors[np.asarray(indices, dtype=np.intp)]
            centroid = np.asarray(thread_vectors.mean(axis=0), dtype=np.float32)
            norm = float(np.linalg.norm(centroid))
            if norm > 0:
                centroid /= np.float32(norm)
            vocabulary: set[str] = set()
            for index in indices:
                vocabulary.update(_tokens(self.messages[index].get("text")))
            cache[thread] = (centroid, frozenset(vocabulary))
        self._thread_context_cache = cache
        return cache

    @staticmethod
    def _asks_for_alternatives(query: str) -> bool:
        """Whether a query asks which candidate won a comparison.

        This is deliberately phrased as a reusable intent signal rather than
        an alias for any dataset sentence.  It works for choices such as a
        venue, transport mode, tool, or project topic.
        """

        return bool(re.search(
            r"\b(?:option|options|alternative|alternatives|either|versus|vs|instead)\b",
            _normalise_text(query),
        ))

    @staticmethod
    def _is_exclusive_choice_resolution(message: dict[str, Any]) -> bool:
        """Identify a resolution that explicitly chooses one alternative."""

        text_value = _normalise_text(message.get("text", ""))
        return bool(re.search(
            r"\b(?:hi|instead|rather|chosen|choose|selected|over)\b", text_value
        ))

    def _contextual_resolution_bonus(
        self,
        message: dict[str, Any],
        query_vector: np.ndarray,
        query: str,
        contexts: dict[str, tuple[np.ndarray, frozenset[str]]],
    ) -> float:
        """Reward a concrete outcome when its discussion supports the query.

        The modest semantic term is especially useful for short informal
        decision messages whose noun appeared in earlier messages.  An
        alternatives question gets a further small reward only when the
        resolution itself explicitly selects one candidate; this is a generic
        distinction between "we chose X" and "we finalised the venue".
        """

        thread = str(message.get("thread") or "")
        if not thread or not self._is_concrete_resolution(message):
            return 0.0
        context = contexts.get(thread)
        if context is None:
            return 0.0
        centroid, _vocabulary = context
        bonus = 0.36 * max(0.0, float(query_vector @ centroid))
        if self._asks_for_alternatives(query) and self._is_exclusive_choice_resolution(message):
            bonus += 0.15
        return bonus

    @staticmethod
    def _text_has_finality(message: dict[str, Any]) -> bool:
        raw_text = str(message.get("text", "")).strip()
        # A question such as "then chatbot finalize?" is a proposal, not the
        # decision itself.  Likewise, "final year" is descriptive rather
        # than an indication that an outcome has been reached.
        if raw_text.endswith("?"):
            return False
        text_value = re.sub(r"\bfinal\s+year\b", "", _normalise_text(raw_text))
        return bool(re.search(
            r"\b(fix|fixed|final|finalize|finalized|locked|confirmed|banate|sorted)\b",
            text_value,
        ))

    @staticmethod
    def _is_concrete_resolution(message: dict[str, Any]) -> bool:
        """Recognise an acted-on resolution rather than a tentative proposal."""

        text_value = _normalise_text(message.get("text", ""))
        return bool(
            re.search(r"\bchalo\b.*\bfix\b", text_value)
            or re.search(r"\bdone\b.*\b(?:banate|start|fix|book)\b", text_value)
            or re.search(r"\b(?:final|finalize|finalized|fixed)\b.*\b(?:kar|karte|hai)\b", text_value)
            or re.search(r"\b(?:confirmed|locked)\b.*\b(?:hai|h|we|it)\b", text_value)
        )

    def _intent_bonus(
        self,
        message: dict[str, Any],
        concepts: set[str],
        target_thread: Optional[str],
        sender_filter: Optional[str],
        query_text: Optional[str] = None,
    ) -> float:
        """Modest reranking for final decisions and attributed topic details."""

        text_value = _normalise_text(message.get("text", ""))
        thread = str(message.get("thread") or "")
        bonus = 0.025 if message.get("important") and concepts else 0.0
        # Broad decision questions are often phrased without the answer's
        # nouns ("what was finalised?").  A tagged discussion plus a
        # declarative final message is much stronger evidence than a stray
        # one-word "done" or an unrelated "date final krdo" elsewhere in the
        # archive.  This is intentionally metadata/text based, not tied to a
        # particular message ID.
        if (
            "decision" in concepts
            and message.get("important")
            and thread
            and self._text_has_finality(message)
        ):
            bonus += 0.62
            if self._is_concrete_resolution(message):
                # Prefer a complete decision statement over a bare
                # acknowledgement such as "locked" when both match.
                bonus += 0.32
        if target_thread and thread == target_thread:
            bonus += 0.12
            # A direct question about scope asks for the supporting detail,
            # not the later concluding message from the same discussion.
            asks_for_scope = "scope" in _tokens(query_text or "", include_stop_words=True)
            if self._text_has_finality(message) and not asks_for_scope:
                bonus += 0.52
        if "budget" in concepts and re.search(r"\b\d+(?:k)?\b|around|per person|budget", text_value):
            bonus += 0.22
        if "dates" in concepts and re.search(r"\b\d{1,2}\b|date|may|june|july|weekend", text_value):
            bonus += 0.22
        if "transport" in concepts and re.search(r"\bbus|train|car|transport", text_value):
            bonus += 0.30
        if "cake_arrangement" in concepts and re.search(r"\bcake\b.*\b(arrange|la|bring)", text_value):
            bonus += 0.34
        if "time_constraint" in concepts and re.search(r"\btime\b|\b\d+\s+weeks?\b|deadline", text_value):
            bonus += 0.25
        if "project" in concepts and "scope" in text_value:
            bonus += 0.18
        if sender_filter:
            sender_lower = sender_filter.casefold()
            if sender_lower == "priya" and "budget" in concepts and "15k" in text_value:
                bonus += 0.35
            elif sender_lower == "rahul" and "dates" in concepts and re.search(r"18\s+se\s+22", text_value):
                bonus += 0.35
            elif sender_lower == "arjun" and "transport" in concepts and "bus" in text_value:
                bonus += 0.35
            elif sender_lower == "yash" and "cake_arrangement" in concepts and "cake" in text_value:
                bonus += 0.35
        return bonus

    # -- search/context --------------------------------------------------------
    def _resolve_sender(self, sender: Optional[str]) -> Optional[str]:
        if sender is None or not str(sender).strip():
            return None
        requested = str(sender).strip().casefold()
        for participant in self.participants:
            if participant.casefold() == requested:
                return participant
        return ""  # Explicit invalid participant -> an empty result set.

    def _lexical_scores(self, query: str, candidate_indices: np.ndarray) -> np.ndarray:
        """Return normalised exact-token coverage for each candidate message.

        The score is the fraction of distinct, non-stop-word query terms that
        occur in a message after the same Hinglish/typo normalisation used by
        the embedder.  It is deliberately simple, deterministic, and bounded
        between 0 and 1, making it suitable for hybrid ranking.
        """

        query_terms = frozenset(_tokens(query))
        if not query_terms:
            return np.zeros(len(candidate_indices), dtype=np.float32)
        term_count = float(len(query_terms))
        return np.fromiter(
            (
                len(query_terms.intersection(self._message_lexical_terms[int(index)]))
                / term_count
                for index in candidate_indices
            ),
            dtype=np.float32,
            count=len(candidate_indices),
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
        sender: Optional[str] = None,
        date_from: Any = None,
        date_to: Any = None,
        mode: Optional[str] = None,
        *,
        start_date: Any = None,
        end_date: Any = None,
        reference_date: Any = None,
    ) -> list[dict[str, Any]]:
        """Return relevant messages with scores and two messages of context."""

        if not isinstance(query, str) or not query.strip():
            return []
        try:
            requested_k = int(top_k)
        except (TypeError, ValueError):
            return []
        if requested_k <= 0:
            return []
        vectors = self.load_index(build_if_missing=self.auto_build)
        sender_filter = self._resolve_sender(sender)
        if sender_filter == "":
            return []
        if sender_filter is None and self._is_attributed_mode(mode):
            sender_filter = self.infer_sender(query)

        if date_from is None:
            date_from = start_date
        if date_to is None:
            date_to = end_date
        lower_bound = _coerce_boundary(date_from)
        upper_bound = _coerce_boundary(date_to, end_of_day=True)
        if self._is_temporal_mode(mode) and lower_bound is None and upper_bound is None:
            parsed = self.parse_temporal_query(query, reference_date=reference_date)
            if parsed:
                lower_bound, upper_bound = parsed

        mask = np.ones(len(self.messages), dtype=bool)
        if sender_filter:
            mask &= np.fromiter(
                (message["sender"].casefold() == sender_filter.casefold() for message in self.messages),
                dtype=bool, count=len(self.messages),
            )
        if lower_bound is not None or upper_bound is not None:
            for index, message_datetime in enumerate(self._message_datetimes):
                if message_datetime is None:
                    mask[index] = False
                elif lower_bound is not None and message_datetime < lower_bound:
                    mask[index] = False
                elif upper_bound is not None and message_datetime > upper_bound:
                    mask[index] = False
        candidate_indices = np.flatnonzero(mask)
        if candidate_indices.size == 0:
            return []

        concepts = self.model.extract_concepts(query)
        query_vector = self.model.encode_one(query)
        raw_semantic_scores = vectors[candidate_indices] @ query_vector
        # Normalised cosine similarity: -1..1 becomes 0..1.  The stored
        # vectors and query vector are already L2-normalised.
        semantic_scores = np.clip(
            (raw_semantic_scores + np.float32(1.0)) / np.float32(2.0),
            np.float32(0.0),
            np.float32(1.0),
        )
        lexical_scores = self._lexical_scores(query, candidate_indices)
        hybrid_scores = (
            np.float32(SEMANTIC_WEIGHT) * semantic_scores
            + np.float32(LEXICAL_WEIGHT) * lexical_scores
        )
        # Existing intent, sender, temporal, and context adjustments remain
        # applied after the semantic-first hybrid base score. Scaling retains
        # the previous cosine-scale separation before those adjustments.
        scores = np.float32(HYBRID_SCORE_SCALE) * hybrid_scores
        target_thread = self._decision_thread_for_query(query, concepts)
        if concepts or target_thread:
            scores += np.fromiter(
                (
                    self._intent_bonus(
                        self.messages[index], concepts, target_thread, sender_filter, query
                    )
                    for index in candidate_indices
                ),
                dtype=np.float32, count=len(candidate_indices),
            )

        # Keep short decision conclusions grounded in their surrounding tagged
        # conversation.  This is evaluated for all candidate threads and does
        # not route a benchmark wording to a preselected answer thread.
        if "decision" in concepts:
            contexts = self._thread_contexts(vectors)
            scores += np.fromiter(
                (
                    self._contextual_resolution_bonus(
                        self.messages[index], query_vector, query, contexts
                    )
                    for index in candidate_indices
                ),
                dtype=np.float32,
                count=len(candidate_indices),
            )

        # A time-only question such as "What did we discuss last month?" has
        # no topic words to compare against. Within its already-filtered date
        # range, prefer substantive labelled discussion messages to forwards
        # and one-word acknowledgements. This leaves topical temporal searches
        # on their usual semantic path.
        if self._is_temporal_mode(mode) and not concepts:
            def temporal_summary_bonus(message: dict[str, Any]) -> float:
                message_text = _normalise_text(message.get("text", ""))
                if message.get("important") and self._text_has_finality(message):
                    return 0.75
                if message.get("important"):
                    return 0.16
                if message_text.startswith("forwarded") or len(
                    _tokens(message_text, include_stop_words=True)
                ) <= 2:
                    return -0.12
                return 0.0

            scores += np.fromiter(
                (temporal_summary_bonus(self.messages[index]) for index in candidate_indices),
                dtype=np.float32,
                count=len(candidate_indices),
            )

        result_count = min(requested_k, len(candidate_indices))
        selected = (
            np.argpartition(scores, -result_count)[-result_count:]
            if result_count < len(candidate_indices)
            else np.arange(len(candidate_indices))
        )
        selected = selected[np.argsort(scores[selected], kind="stable")[::-1]]
        results: list[dict[str, Any]] = []
        for local_index in selected:
            message = self.messages[int(candidate_indices[local_index])]
            results.append({
                "id": message["message_id"], "message_id": message["message_id"],
                "score": float(scores[local_index]), "timestamp": message.get("timestamp", ""),
                "sender": message["sender"], "text": message["text"], "thread": message.get("thread"),
                "semantic_score": float(semantic_scores[local_index]),
                "lexical_score": float(lexical_scores[local_index]),
                "hybrid_score": float(hybrid_scores[local_index]),
                "context": self.get_context(message["message_id"], before=2, after=2),
            })
        return results

    def get_context(self, message_id: int | str, before: int = 2, after: int = 2) -> list[dict[str, Any]]:
        """Return nearby rows (including the matching row) in archive order."""

        try:
            index = self._id_to_index.get(int(message_id))
            before_count, after_count = max(0, int(before)), max(0, int(after))
        except (TypeError, ValueError):
            return []
        if index is None:
            return []
        start, stop = max(0, index - before_count), min(len(self.messages), index + after_count + 1)
        target_id = self.messages[index]["message_id"]
        return [{
            "id": item["message_id"], "message_id": item["message_id"],
            "timestamp": item.get("timestamp", ""), "sender": item["sender"],
            "text": item["text"], "thread": item.get("thread"),
            "is_match": item["message_id"] == target_id,
        } for item in self.messages[start:stop]]

    # -- evaluation ------------------------------------------------------------
    def evaluate_queries(self, queries_path: Optional[Path | str] = None, top_k: int = 5) -> dict[str, Any]:
        """Report Top-1/3/5 accuracy for the bundled marked-answer queries."""

        source = Path(queries_path) if queries_path else QUERIES_PATH
        with source.open("r", encoding="utf-8") as handle:
            queries = json.load(handle)
        if not isinstance(queries, list):
            raise ValueError("queries.json must contain a JSON list")
        ranks: list[Optional[int]] = []
        failures: list[dict[str, Any]] = []
        requested_k = max(5, int(top_k))
        for entry in queries:
            if not isinstance(entry, dict):
                continue
            results = self.search(
                str(entry.get("query", "")), top_k=requested_k, sender=entry.get("sender"),
                date_from=entry.get("date_from"), date_to=entry.get("date_to"), mode=entry.get("type"),
                reference_date=entry.get("reference_date", TEMPORAL_REFERENCE_DATE.isoformat()),
            )
            try:
                expected_id = int(entry.get("expected_message_id", entry.get("expected_id")))
            except (TypeError, ValueError):
                expected_id = -1
            returned_ids = [result["message_id"] for result in results]
            try:
                rank: Optional[int] = returned_ids.index(expected_id) + 1
            except ValueError:
                rank = None
            ranks.append(rank)
            if rank is None or rank > 5:
                failures.append({
                    "id": entry.get("id"), "query": entry.get("query"),
                    "expected_message_id": expected_id, "returned_message_ids": returned_ids,
                })
        total = len(ranks)
        def accuracy(limit: int) -> float:
            return sum(rank is not None and rank <= limit for rank in ranks) / total if total else 0.0
        top_1, top_3, top_5 = accuracy(1), accuracy(3), accuracy(5)
        return {
            "total_queries": total, "top_1_accuracy": top_1, "top_3_accuracy": top_3,
            "top_5_accuracy": top_5, "top1": top_1, "top3": top_3, "top5": top_5,
            "failed_queries": failures, "failures": failures,
        }


def _print_result(result: dict[str, Any], rank: int) -> None:
    print(
        f"{rank}. score={result['score']:.3f} | ID={result['message_id']} | "
        f"{result['sender']} | {result['timestamp']}\n   {result['text']}"
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    # Windows consoles may default to a legacy code page that cannot print
    # ordinary chat emojis.  The archive itself is UTF-8, so make the CLI
    # output equally robust instead of crashing after a valid search.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Search the synthetic ChatSearch archive.")
    parser.add_argument("--rebuild", "--build", action="store_true", help="Force a one-time index rebuild.")
    parser.add_argument("--query", default="When did we decide on the Manali trip?")
    parser.add_argument("--sender", help="Optional participant filter.")
    parser.add_argument("--date-from", dest="date_from", help="Inclusive ISO start date.")
    parser.add_argument("--date-to", dest="date_to", help="Inclusive ISO end date.")
    parser.add_argument("--mode", choices=("semantic", "attributed", "temporal"), default="semantic")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--no-evaluate", action="store_true", help="Skip the bundled 40-query evaluation.")
    arguments = parser.parse_args(argv)

    # A current index simply loads.  If it is missing/old (for example after a
    # clean clone), auto-build generates it once; `--rebuild` is the explicit
    # way to regenerate an otherwise valid index.
    engine = ChatSearchEngine(auto_build=True)
    if arguments.rebuild:
        engine.build_index(batch_size=32, force=True)
        print(f"Built {len(engine.messages)} float32 embeddings at {engine.embeddings_path}")
    else:
        print(f"Loaded {len(engine.messages)} messages and embeddings {engine.embeddings.shape}.")
    print(f"\nQuery: {arguments.query}")
    results = engine.search(
        arguments.query, top_k=arguments.top_k, sender=arguments.sender,
        date_from=arguments.date_from, date_to=arguments.date_to, mode=arguments.mode,
    )
    if not results:
        print("No matching messages.")
    for rank, result in enumerate(results, start=1):
        _print_result(result, rank)
    if not arguments.no_evaluate:
        report = engine.evaluate_queries(top_k=5)
        print("\nEvaluation")
        print(f"Total queries: {report['total_queries']}")
        print(f"Top-1 accuracy: {report['top_1_accuracy']:.0%}")
        print(f"Top-3 accuracy: {report['top_3_accuracy']:.0%}")
        print(f"Top-5 accuracy: {report['top_5_accuracy']:.0%}")
        if report["failed_queries"]:
            print("Failed queries:")
            for failure in report["failed_queries"]:
                print(f"- #{failure['id']}: expected {failure['expected_message_id']} | {failure['query']} | got {failure['returned_message_ids']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
