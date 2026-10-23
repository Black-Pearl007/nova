"""Rule-based baseline: the same keyword patterns Nova's original rule engine uses,
checked in the same priority order (commands first, then feelings, then topics, then small talk).
Returns 'unknown' when no rule fires, which counts as a wrong answer in evaluation."""
import re

RULES = [
    ("breathing", r"\b(breath|breathing|calm me|calm down|grounding)\b"),
    ("game", r"\b(trivia|quiz me|quiz|riddle|riddles|guess the number|would you rather|rock paper scissors|rps|play a game|games|game time|let'?s play)\b"),
    ("joke", r"\b(cheer me up|joke|funny|make me laugh|laugh)\b"),
    ("music", r"\b(music|song|songs|playlist|tune|tunes|listen|sing|play (?:me|something|some|a))\b"),
    ("sad", r"\b(sad|unhappy|depressed|crying|heartbroken|miserable|feeling low|feel low|upset|down today)\b"),
    ("stressed", r"\b(stress|stressed|overwhelm|overwhelmed|pressure|burnt out|burnout|tension)\b"),
    ("anxious", r"\b(anxious|anxiety|worried|worry|scared|afraid|nervous|panic)\b"),
    ("angry", r"\b(angry|furious|mad at|irritated|annoyed|annoying|frustrated|rage)\b"),
    ("lonely", r"\b(lonely|feel alone|all alone|nobody cares|no friends)\b"),
    ("bored", r"\b(bored|boring|nothing to do)\b"),
    ("happy", r"\b(happy|excited|great day|awesome|amazing|good news|celebrat\w*|i passed|got the job|got selected)\b"),
    ("study_help", r"\b(study|studying|revise|revision|homework|assignment|learn faster|memori[sz]e|procrastinat\w*|can'?t focus|distract\w*|exam|exams)\b"),
    ("career_help", r"\b(career|job|resume|placement|college|masters|future|higher studies|interview)\b"),
    ("sleep_help", r"\b(sleep|insomnia|can'?t sleep|awake all night)\b"),
    ("how_are_you", r"\b(how are you|how r u|how do you do|what'?s up|sup)\b"),
    ("thanks", r"\b(thank you|thanks|thx|ty)\b"),
    ("goodbye", r"\b(bye|goodbye|good night|see you|gtg)\b"),
    ("greeting", r"^\s*(hello|hi|hey|hii+|hola|vanakkam|good morning|good evening|good afternoon|yo)\b"),
]
COMPILED = [(label, re.compile(p, re.I)) for label, p in RULES]


def predict(text: str) -> str:
    for label, rx in COMPILED:
        if rx.search(text):
            return label
    return "unknown"
