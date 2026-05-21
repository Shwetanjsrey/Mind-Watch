import re
from src.emotion_analyzer import analyze_emotions
from src.explain_prediction import get_severe_probability


HIGH_RISK_PHRASES = [
    "want to die",
    "kill myself",
    "end my life",
    "ending my life",
    "take my life",
    "suicide",
    "suicidal",
    "self harm",
    "hurt myself",
    "hopeless",
    "worthless",
    "no reason to live",
    "want to disappear",
    "give up",
    "i give up",
    "can't go on",
    "cannot go on",
    "don't want to continue",
    "all hope is gone",
    "lose soon"
]

MEDIUM_RISK_PHRASES = [
    "empty",
    "alone",
    "numb",
    "drained",
    "anxious",
    "panic",
    "overwhelmed",
    "stressed",
    "depressed",
    "sad",
    "crying",
    "broken",
    "tired",
    "exhausted",
    "meaningless",
    "failure",
    "no friends",
    "no family"
]


INDIAN_ACTIONS = {
    "LOW": [
        "Practice sleep hygiene, hydration, regular meals, and movement.",
        "Reach out to trusted friends or family.",
        "If emotional distress persists, contact iCall (TISS): 9152987821"
    ],
    "MEDIUM": [
        "Reach out for emotional support.",
        "iCall (TISS): 9152987821",
        "Vandrevala Foundation: 1860-2662-345",
        "https://icallhelpline.org"
    ],
    "HIGH": [
        "Immediate support strongly recommended.",
        "iCall (TISS): 9152987821",
        "Vandrevala Foundation: 1860-2662-345",
        "Snehi: 044-24640050",
        "NIMHANS: 080-46110007",
        "Fortis Stress Helpline: 8376804102"
    ]
}


NEGATIVE_EMOTIONS = {
    "sadness",
    "disappointment",
    "fear",
    "grief",
    "nervousness",
    "anger",
    "annoyance",
    "remorse",
    "embarrassment",
    "confusion"
}

POSITIVE_EMOTIONS = {
    "gratitude",
    "joy",
    "admiration",
    "approval",
    "optimism",
    "relief",
    "love",
    "caring",
    "excitement",
    "amusement"
}


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_matches(text, phrases):
    matches = []
    for phrase in phrases:
        if phrase in text:
            matches.append(phrase)
    return matches


def extract_signal_tokens(matches):
    tokens = set()

    stopwords = {
        "i", "me", "my", "myself",
        "he", "his", "she", "her",
        "for", "the", "a", "an",
        "to", "and", "or", "of",
        "in", "on", "at", "with",
        "is", "are", "was", "were",
        "it", "this", "that",
        "want", "life", "continue"
    }

    for phrase in matches:
        for token in phrase.split():
            token = token.lower()
            if token not in stopwords and len(token) > 2:
                tokens.add(token)

    return list(tokens)


def build_highlighted_text(text, high_tokens, medium_tokens):
    words = text.split()
    rendered = []

    for word in words:
        clean_word = re.sub(r"[^\w']", "", word.lower())

        if clean_word in high_tokens:
            rendered.append(f'<span class="hl-high">{word}</span>')
        elif clean_word in medium_tokens:
            rendered.append(f'<span class="hl-medium">{word}</span>')
        else:
            rendered.append(word)

    return " ".join(rendered)


def compute_probabilities(emotions, severe_prob, high_matches, medium_matches):
    positive = 0.0
    moderate = 0.0

    for emotion, score in emotions.items():
        e = emotion.lower()

        if e in POSITIVE_EMOTIONS:
            positive += score

        elif e in NEGATIVE_EMOTIONS:
            moderate += score

        elif e == "neutral":
            positive += score * 0.85

    explicit_risk = (
        len(high_matches) * 0.45 +
        len(medium_matches) * 0.18
    )

    if len(high_matches) == 0 and len(medium_matches) == 0:
        severe_weight = severe_prob * 0.25
    else:
        severe_weight = severe_prob * 0.65

    crisis = min(
        1.0,
        severe_weight + explicit_risk
    )

    total = positive + moderate + crisis

    if total == 0:
        return 60.0, 25.0, 15.0

    low = (positive / total) * 100
    med = (moderate / total) * 100
    high = (crisis / total) * 100

    return round(low, 1), round(med, 1), round(high, 1)


def classify_risk(score):
    if score < 32:
        return "LOW"
    elif score < 62:
        return "MEDIUM"
    return "HIGH"


def risk_color(level):
    if level == "LOW":
        return "#34d399"
    elif level == "MEDIUM":
        return "#fbbf24"
    return "#f87171"


def analyze_risk(text):
    cleaned = clean_text(text)

    severe_prob = get_severe_probability(cleaned)

    emotions = analyze_emotions(cleaned)

    high_matches = extract_matches(cleaned, HIGH_RISK_PHRASES)
    medium_matches = extract_matches(cleaned, MEDIUM_RISK_PHRASES)

    high_tokens = extract_signal_tokens(high_matches)
    medium_tokens = extract_signal_tokens(medium_matches)

    low_prob, med_prob, high_prob = compute_probabilities(
        emotions,
        severe_prob,
        high_matches,
        medium_matches
    )

    final_score = round(
        (med_prob * 0.5) + high_prob,
        1
    )

    risk_level = classify_risk(final_score)

    confidence = round(
        max(low_prob, med_prob, high_prob),
        1
    )

    highlighted = build_highlighted_text(
        text,
        high_tokens,
        medium_tokens
    )

    return {
        "risk_score": final_score,
        "risk_level": risk_level,
        "risk_color": risk_color(risk_level),
        "confidence": confidence,
        "uncertain": confidence < 55,
        "emotions": emotions,
        "high_matches": high_matches,
        "medium_matches": medium_matches,
        "high_tokens": high_tokens,
        "medium_tokens": medium_tokens,
        "highlighted_text": highlighted,
        "actions": INDIAN_ACTIONS[risk_level],
        "word_count": len(cleaned.split()),
        "high_count": len(high_matches),
        "medium_count": len(medium_matches),
        "severe_probability": round(severe_prob * 100, 1),
        "low_probability": low_prob,
        "medium_probability": med_prob,
        "high_probability": high_prob
    }