import json
import random
from datetime import datetime, timedelta
from pathlib import Path


# --------------------------------------------------
# Configuration
# --------------------------------------------------

SEED = 20260908
TOTAL_MESSAGES = 4300
PARTICIPANTS = [
    "Yash",
    "Priya",
    "Rahul",
    "Aman",
    "Neha",
    "Rohan",
    "Sneha",
    "Arjun",
]

START_DATE = datetime(2026, 3, 1, 8, 0)
END_DATE = datetime(2026, 8, 31, 23, 0)

random.seed(SEED)


# --------------------------------------------------
# Message templates
# --------------------------------------------------

NORMAL_MESSAGES = [
    "bhai kya scene hai?",
    "haan bro done",
    "kal college aa rhe ho?",
    "aaj kaafi kaam hai yaar",
    "lol 😂",
    "same bro",
    "okay",
    "haan",
    "k",
    "done 👍",
    "wait 5 min",
    "abhi dekha message",
    "kya plan hai?",
    "lets see",
    "sahi hai",
    "nicee",
    "bhai ye wala better lag rha",
    "kal milte hain",
    "mai thodi der me aata hu",
    "haan bilkul",
    "not sure yaar",
    "check krta hu",
    "send kar dena",
    "mil gaya",
    "thanks bro",
    "haha 😂",
    "seriously?",
    "achha okay",
    "ruk ja",
    "coming",
]


HINGLISH_MESSAGES = [
    "bhai budget thoda zyada ho jayega",
    "ye idea mujhe sahi lag rha hai",
    "kal tak confirm kr dena",
    "sab log free ho kya?",
    "meeting ka time fix karo",
    "ye wala option better hai imo",
    "kuch jugaad kar lenge",
    "abhi decide mat karo",
    "pehle details dekh lete hain",
    "haan ye workable hai",
    "iska kya karna hai?",
    "mujhe lagta hai ye best rahega",
    "thoda late ho jayega",
    "main check karke batata hu",
    "ye toh mast hai",
]


TYPO_MESSAGES = [
    "manali fix h",
    "kal collge aa rha?",
    "hotel book krna h",
    "budget kitna hoga bhai?",
    "mai confirm krta hu",
    "ye place accha lg rha",
    "date final krdo",
    "sab ready h?",
    "koi prob nahi",
    "mai bhi aaunga",
]


FORWARDED_MESSAGES = [
    "Forwarded: Limited period travel offer available now",
    "Forwarded: Check this amazing hotel deal",
    "Forwarded: College event registration link",
    "Forwarded: Weekend trip package details",
    "Forwarded: Important notice - please read",
]


# --------------------------------------------------
# Helper functions
# --------------------------------------------------

def random_timestamp():
    """Generate a random timestamp within our 6-month period."""

    total_seconds = int((END_DATE - START_DATE).total_seconds())

    random_seconds = random.randint(0, total_seconds)

    return START_DATE + timedelta(seconds=random_seconds)


def generate_message_text():
    """Generate realistic messy chat text."""

    choice = random.random()

    if choice < 0.55:
        return random.choice(NORMAL_MESSAGES)

    if choice < 0.82:
        return random.choice(HINGLISH_MESSAGES)

    if choice < 0.94:
        return random.choice(TYPO_MESSAGES)

    return random.choice(FORWARDED_MESSAGES)


def format_chat_message(message):
    """
    Convert structured message into a WhatsApp-like
    chat export line.
    """

    timestamp = message["timestamp"]

    dt = datetime.fromisoformat(timestamp)

    date = dt.strftime("%d/%m/%Y")
    time = dt.strftime("%I:%M %p")

    return f"{date}, {time} - {message['sender']}: {message['text']}"


# --------------------------------------------------
# Important decision threads
# --------------------------------------------------

def create_manali_thread():
    """
    Important semantic-search thread.

    The final decision message deliberately uses wording
    that differs from many possible search queries.
    """

    base_time = datetime(2026, 5, 18, 19, 30)

    messages = [
        ("Rahul", "Guys summer me kahi nikalna chahiye"),
        ("Aman", "haan bhai kuch pahad side"),
        ("Priya", "budget bhi dekhna padega"),
        ("Neha", "Shimla ya Manali?"),
        ("Yash", "Manali sounds good"),
        ("Rohan", "haan waha weather bhi mast hoga"),
        ("Sneha", "per person kitna soch rahe ho?"),
        ("Priya", "around 15k max rakhte hain"),
        ("Arjun", "15k manageable hai"),
        ("Yash", "dates bhi check kar lo"),
        ("Rahul", "18 se 22 May kaisa hai?"),
        ("Priya", "mere liye okay"),
        ("Aman", "same"),
        ("Neha", "works"),
        ("Rohan", "hotel mai dekh lunga"),
        ("Sneha", "transport ka kya scene?"),
        ("Arjun", "bus se chale toh budget me aa jayega"),
        ("Rahul", "sab agree ho toh book kar dete hain"),
        # Important answer message
        ("Yash", "Chalo, Manali fix hai. Main hotel dekh leta hu."),
        ("Priya", "done 👍"),
        ("Rahul", "perfect"),
    ]

    return build_thread(messages, base_time, "manali_decision")


def create_birthday_thread():
    """
    Second important decision thread.
    """

    base_time = datetime(2026, 6, 25, 18, 15)

    messages = [
        ("Sneha", "Guys birthday ke liye kuch plan karte hain"),
        ("Priya", "restaurant chalein?"),
        ("Rahul", "haan but expensive nahi hona chahiye"),
        ("Neha", "cafe bhi option hai"),
        ("Aman", "parking ka bhi dekhna"),
        ("Yash", "Saturday evening free hoon"),
        ("Rohan", "main bhi"),
        ("Arjun", "venue decide karna hai pehle"),
        ("Priya", "City Center wala cafe kaisa hai?"),
        ("Sneha", "reviews theek hain"),
        ("Rahul", "budget 8k ke andar rakhte hain"),
        ("Neha", "works for me"),
        ("Aman", "cake kaun laega?"),
        ("Yash", "main cake arrange kar dunga"),
        ("Rohan", "then sorted"),
        # Important answer
        ("Sneha", "Theek hai, City Center cafe final karte hain."),
        ("Priya", "haan done"),
        ("Rahul", "locked"),
    ]

    return build_thread(messages, base_time, "birthday_decision")


def create_project_thread():
    """
    Third important decision thread.
    """

    base_time = datetime(2026, 7, 20, 20, 0)

    messages = [
        ("Arjun", "Final year project ka topic choose karna hai"),
        ("Yash", "AI based kuch bana sakte hain"),
        ("Rahul", "recommendation system?"),
        ("Priya", "chatbot bhi option hai"),
        ("Neha", "data kaha se milega?"),
        ("Aman", "public datasets available hain"),
        ("Rohan", "deployment bhi simple hona chahiye"),
        ("Sneha", "Python use karenge toh easy rahega"),
        ("Yash", "time sirf 3 weeks hai"),
        ("Priya", "chatbot scope manageable hai"),
        ("Rahul", "haan recommendation thoda bada ho jayega"),
        ("Arjun", "then chatbot finalize?",
         ),
        ("Neha", "I am okay"),
        ("Aman", "same"),
        # Important answer
        ("Arjun", "Done, chatbot hi banate hain. Kal se implementation start."),
        ("Yash", "perfect"),
        ("Priya", "lets go 🔥"),
    ]

    return build_thread(messages, base_time, "project_decision")


def build_thread(messages, base_time, thread_name):
    """Build structured messages for an important thread."""

    result = []

    for index, (sender, text) in enumerate(messages):
        timestamp = base_time + timedelta(minutes=index * 3)

        result.append(
            {
                "timestamp": timestamp.isoformat(),
                "sender": sender,
                "text": text,
                "thread": thread_name,
                "important": True,
            }
        )

    return result


# --------------------------------------------------
# Main dataset generation
# --------------------------------------------------

def generate_dataset():
    # Reset the generator for every call so importing this module in a test or
    # notebook remains reproducible too (not only a fresh CLI invocation).
    random.seed(SEED)

    messages = []

    # Generate normal background conversation.
    for _ in range(TOTAL_MESSAGES):
        timestamp = random_timestamp()
        sender = random.choice(PARTICIPANTS)
        text = generate_message_text()

        messages.append(
            {
                "timestamp": timestamp.isoformat(),
                "sender": sender,
                "text": text,
                "thread": None,
                "important": False,
            }
        )

    # Add our three important threads.
    messages.extend(create_manali_thread())
    messages.extend(create_birthday_thread())
    messages.extend(create_project_thread())

    # Sort chronologically.
    messages.sort(key=lambda x: x["timestamp"])

    # Assign stable message IDs.
    for message_id, message in enumerate(messages, start=1):
        message["message_id"] = message_id

    return messages


def save_dataset(messages):
    """Save both JSON and readable chat export."""

    data_dir = Path(__file__).resolve().parent.parent / "data"
    data_dir.mkdir(exist_ok=True)

    json_path = data_dir / "chat.json"
    txt_path = data_dir / "chat.txt"

    # Save structured data.
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(messages, file, ensure_ascii=False, indent=2)

    # Save chat-export style text.
    with open(txt_path, "w", encoding="utf-8") as file:
        for message in messages:
            file.write(format_chat_message(message))
            file.write("\n")

    print(f"Generated {len(messages)} messages.")
    print(f"JSON: {json_path}")
    print(f"Chat export: {txt_path}")


def main():
    messages = generate_dataset()
    save_dataset(messages)


if __name__ == "__main__":
    main()
