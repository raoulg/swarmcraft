import random

ADJECTIVES = [
    ("agile", "🤸"),
    ("analog", "🕹️"),
    ("bold", "🛡️"),
    ("bouncing", "🏀"),
    ("bratty", "💅"),
    ("clever", "🧠"),
    ("colorful", "🌈"),
    ("cosmic", "🌌"),
    ("cyber", "🤖"),
    ("electric", "⚡"),
    ("funky", "🕺"),
    ("hyper", "🚀"),
    ("lunar", "🌙"),
    ("magic", "✨"),
    ("majestic", "👑"),
    ("medieval", "🏰"),
    ("mystic", "🔮"),
    ("neon", "🚥"),
    ("nerdy", "🤓"),
    ("ninja", "🥷"),
    ("pixelated", "👾"),
    ("punky", "🎸"),
    ("purple", "💜"),
    ("quantum", "⚛️"),
    ("solar", "☀️"),
    ("stellar", "🌟"),
    ("spiralling", "🌀"),
    ("super", "🦸"),
    ("swift", "💨"),
    ("turbo", "🏎️"),
    ("ultra", "💎"),
    ("untamed", "🐾"),
    ("zen", "🧘"),
]

ANIMALS = [
    ("bat", "🦇"),
    ("wolf", "🐺"),
    ("fox", "🦊"),
    ("owl", "🦉"),
    ("hawk", "🦅"),
    ("shark", "🦈"),
    ("dolphin", "🐬"),
    ("whale", "🐋"),
    ("octopus", "🐙"),
    ("panther", "🐆"),
    ("tiger", "🐅"),
    ("lion", "🦁"),
    ("bear", "🐻"),
    ("deer", "🦌"),
    ("rabbit", "🐰"),
    ("squirrel", "🐿️"),
    ("otter", "🦦"),
    ("penguin", "🐧"),
    ("dragon", "🐉"),
    ("unicorn", "🦄"),
    ("kraken", "🦑"),
    ("snake", "🐍"),
    ("dragon", "🐲"),
    ("salamander", "🦎"),
    ("butterfly", "🦋"),
    ("hummingbird", "🐦"),
    ("flamingo", "🦩"),
    ("mycelium", "🍄"),
    ("dinosaur", "🦕"),
    ("ant", "🐜"),
    ("beetle", "🪲"),
    ("crab", "🦀"),
    ("lobster", "🦞"),
    ("duck", "🦆"),
]


def generate_participant_name() -> tuple[str, list[str]]:
    """Generate a fun participant name like 'vibrant bat' and return it with emojis"""
    adj_name, adj_emoji = random.choice(ADJECTIVES)
    animal_name, animal_emoji = random.choice(ANIMALS)
    number = random.randint(1, 99)

    name = f"{adj_name}-{animal_name}-{number}"
    emojis = [adj_emoji, animal_emoji]

    return name, emojis


def generate_session_code() -> str:
    """Generate a 6-character session code"""
    import string

    chars = string.ascii_uppercase + string.digits
    # Avoid confusing characters
    chars = chars.replace("0", "").replace("O", "").replace("1", "").replace("I", "")
    return "".join(random.choice(chars) for _ in range(6))
