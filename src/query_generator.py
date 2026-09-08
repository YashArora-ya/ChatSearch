import json
from pathlib import Path

from search_engine import TEMPORAL_REFERENCE_DATE


DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "chat.json"
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "queries.json"


def load_messages():
    with open(DATA_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def find_message(messages, thread, text):
    for message in messages:
        if (
            message["thread"] == thread
            and message["text"] == text
        ):
            return message["message_id"]

    raise ValueError(f"Message not found: {text}")


def build_queries(messages):

    manali = find_message(
        messages,
        "manali_decision",
        "Chalo, Manali fix hai. Main hotel dekh leta hu.",
    )

    budget = find_message(
        messages,
        "manali_decision",
        "around 15k max rakhte hain",
    )

    dates = find_message(
        messages,
        "manali_decision",
        "18 se 22 May kaisa hai?",
    )

    transport = find_message(
        messages,
        "manali_decision",
        "bus se chale toh budget me aa jayega",
    )

    birthday_final = find_message(
        messages,
        "birthday_decision",
        "Theek hai, City Center cafe final karte hain.",
    )

    birthday_budget = find_message(
        messages,
        "birthday_decision",
        "budget 8k ke andar rakhte hain",
    )

    birthday_cake = find_message(
        messages,
        "birthday_decision",
        "main cake arrange kar dunga",
    )

    project_final = find_message(
        messages,
        "project_decision",
        "Done, chatbot hi banate hain. Kal se implementation start.",
    )

    project_scope = find_message(
        messages,
        "project_decision",
        "chatbot scope manageable hai",
    )

    project_time = find_message(
        messages,
        "project_decision",
        "time sirf 3 weeks hai",
    )

    queries = [

        # --------------------------------------------------
        # SEMANTIC SEARCH
        # --------------------------------------------------

        {
            "id": 1,
            "query": "When did we decide on the Manali trip?",
            "type": "semantic",
            "expected_message_id": manali,
        },

        {
            "id": 2,
            "query": "Which place did everyone finally agree to visit?",
            "type": "semantic",
            "expected_message_id": manali,
            # The answer says "Manali fix hai"; this wording deliberately
            # does not repeat those answer terms.
            "zero_word_overlap": True,
        },

        {
            "id": 3,
            "query": "What destination was locked for the summer outing?",
            "type": "semantic",
            "expected_message_id": manali,
            "zero_word_overlap": True,
        },

        {
            "id": 4,
            "query": "When was the hill trip finalized?",
            "type": "semantic",
            "expected_message_id": manali,
            "zero_word_overlap": True,
        },

        {
            "id": 5,
            "query": "Where were we planning to go for the mountains?",
            "type": "semantic",
            "expected_message_id": manali,
            "zero_word_overlap": True,
        },

        {
            "id": 6,
            "query": "What was the agreed travel destination?",
            "type": "semantic",
            "expected_message_id": manali,
            "zero_word_overlap": True,
        },

        {
            "id": 7,
            "query": "Which location did the group settle on?",
            "type": "semantic",
            "expected_message_id": manali,
            "zero_word_overlap": True,
        },

        {
            "id": 8,
            "query": "What place was ultimately chosen for the trip?",
            "type": "semantic",
            "expected_message_id": manali,
            "zero_word_overlap": True,
        },

        {
            "id": 9,
            "query": "What was the final decision regarding the mountain vacation?",
            "type": "semantic",
            "expected_message_id": manali,
            "zero_word_overlap": True,
        },

        {
            "id": 10,
            "query": "Which city became the final choice?",
            "type": "semantic",
            "expected_message_id": manali,
        },

        {
            "id": 11,
            "query": "Where did the group agree to travel together?",
            "type": "semantic",
            "expected_message_id": manali,
        },

        {
            "id": 12,
            "query": "What destination got confirmed by everyone?",
            "type": "semantic",
            "expected_message_id": manali,
        },

        {
            "id": 13,
            "query": "Which place did the friends settle on?",
            "type": "semantic",
            "expected_message_id": manali,
        },

        {
            "id": 14,
            "query": "What was finalized for the summer plan?",
            "type": "semantic",
            "expected_message_id": manali,
        },

        {
            "id": 15,
            "query": "Which hill station was selected?",
            "type": "semantic",
            "expected_message_id": manali,
        },


        # --------------------------------------------------
        # ATTRIBUTED SEARCH
        # --------------------------------------------------

        {
            "id": 16,
            "query": "What did Priya say about the trip budget?",
            "type": "attributed",
            "sender": "Priya",
            "expected_message_id": budget,
        },

        {
            "id": 17,
            "query": "How much money did Priya suggest keeping for the trip?",
            "type": "attributed",
            "sender": "Priya",
            "expected_message_id": budget,
        },

        {
            "id": 18,
            "query": "What amount did Priya consider reasonable?",
            "type": "attributed",
            "sender": "Priya",
            "expected_message_id": budget,
        },

        {
            "id": 19,
            "query": "What dates did Rahul propose?",
            "type": "attributed",
            "sender": "Rahul",
            "expected_message_id": dates,
        },

        {
            "id": 20,
            "query": "What travel dates were suggested by Rahul?",
            "type": "attributed",
            "sender": "Rahul",
            "expected_message_id": dates,
        },

        {
            "id": 21,
            "query": "What did Arjun suggest for transportation?",
            "type": "attributed",
            "sender": "Arjun",
            "expected_message_id": transport,
        },

        {
            "id": 22,
            "query": "What travel option did Arjun recommend?",
            "type": "attributed",
            "sender": "Arjun",
            "expected_message_id": transport,
        },

        {
            "id": 23,
            "query": "What did Sneha say about the birthday plan?",
            "type": "attributed",
            "sender": "Sneha",
            "expected_message_id": birthday_final,
        },

        {
            "id": 24,
            "query": "What budget did Rahul suggest for the birthday?",
            "type": "attributed",
            "sender": "Rahul",
            "expected_message_id": birthday_budget,
        },

        {
            "id": 25,
            "query": "Who offered to arrange the cake?",
            "type": "attributed",
            "sender": "Yash",
            "expected_message_id": birthday_cake,
        },


        # --------------------------------------------------
        # TEMPORAL SEARCH
        # --------------------------------------------------

        {
            "id": 26,
            "query": "What was discussed around the Manali planning period?",
            "type": "temporal",
            "date_from": "2026-05-18",
            "date_to": "2026-05-19",
            "expected_message_id": manali,
        },

        {
            "id": 27,
            "query": "What decision was made during the May trip discussion?",
            "type": "temporal",
            "date_from": "2026-05-18",
            "date_to": "2026-05-19",
            "expected_message_id": manali,
        },

        {
            "id": 28,
            "query": "What did the group settle on during the May conversation?",
            "type": "temporal",
            "date_from": "2026-05-18",
            "date_to": "2026-05-19",
            "expected_message_id": manali,
        },

        {
            "id": 29,
            "query": "What happened in the birthday planning discussion in June?",
            "type": "temporal",
            "date_from": "2026-06-25",
            "date_to": "2026-06-26",
            "expected_message_id": birthday_final,
        },

        {
            "id": 30,
            "query": "What venue was finalized during the June discussion?",
            "type": "temporal",
            "date_from": "2026-06-25",
            "date_to": "2026-06-26",
            "expected_message_id": birthday_final,
        },

        {
            "id": 31,
            "query": "What project decision happened in July?",
            "type": "temporal",
            "date_from": "2026-07-20",
            "date_to": "2026-07-21",
            "expected_message_id": project_final,
        },

        {
            "id": 32,
            "query": "What did everyone decide during the July project discussion?",
            "type": "temporal",
            "date_from": "2026-07-20",
            "date_to": "2026-07-21",
            "expected_message_id": project_final,
        },

        {
            "id": 33,
            "query": "What was finalized near the end of July?",
            "type": "temporal",
            "date_from": "2026-07-20",
            "date_to": "2026-07-21",
            "expected_message_id": project_final,
        },

        {
            "id": 34,
            "query": "What was the project scope discussed in July?",
            "type": "temporal",
            "date_from": "2026-07-20",
            "date_to": "2026-07-21",
            "expected_message_id": project_scope,
        },

        {
            "id": 35,
            "query": "What time constraint was mentioned during the July project conversation?",
            "type": "temporal",
            "date_from": "2026-07-20",
            "date_to": "2026-07-21",
            "expected_message_id": project_time,
        },


        # --------------------------------------------------
        # DIFFICULT / ZERO WORD OVERLAP
        # --------------------------------------------------

        {
            "id": 36,
            "query": "Which destination got locked in?",
            "type": "semantic",
            "expected_message_id": manali,
            "zero_word_overlap": True,
        },

        {
            "id": 37,
            "query": "Where are we actually going?",
            "type": "semantic",
            "expected_message_id": manali,
            "zero_word_overlap": True,
        },

        {
            "id": 38,
            "query": "What did the group finally settle on?",
            "type": "semantic",
            "expected_message_id": birthday_final,
            "zero_word_overlap": True,
        },

        {
            "id": 39,
            "query": "What did we end up choosing for the final-year work?",
            "type": "semantic",
            "expected_message_id": project_final,
            "zero_word_overlap": True,
        },

        {
            "id": 40,
            "query": "Which option became the final one?",
            "type": "semantic",
            "expected_message_id": project_final,
            "zero_word_overlap": True,
        },
    ]

    # The archive ends in August 2026.  Carry the same deterministic clock on
    # every labelled query so evaluation never depends on the developer's
    # machine date.  The engine only needs it for relative temporal phrases;
    # putting it on every record keeps the evaluation file self-describing.
    for query in queries:
        query["reference_date"] = TEMPORAL_REFERENCE_DATE.isoformat()

    return queries


def main():
    messages = load_messages()
    queries = build_queries(messages)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(queries, file, ensure_ascii=False, indent=2)

    print(f"Created {len(queries)} evaluation queries.")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
