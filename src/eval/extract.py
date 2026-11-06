import re

MCQ = "mcq-4-choices"
FILLING = "filling"

# A line that starts with "Answer:", allowing markdown around the label ("**Answer:** B", "**Answer**: B")
ANSWER_LINE = re.compile(r"^[\W_]*answer[\W_]*?:(.*)$", re.IGNORECASE)
# One letter and nothing else, allowing punctuation and markdown around it ("(B)", "**B**", "B.")
MCQ_LETTER = re.compile(r"^[\W_]*([A-Da-d])[\W_]*$")
# Longest first, so "-->" isn't read as "-" followed by "->"
ARROWS = ("-->", "->", "⟶", "→")

def find_answer_text(text: str) -> str | None:
    """
    Finds what follows the last "Answer:" line of a response.

    Args:
        text (str): Full response.
    Returns:
        The text after the label, or None if there's no such line.
    """
    for line in reversed(text.splitlines()):
        match = ANSWER_LINE.match(line.strip())
        if match:
            return match.group(1).strip()
    return None

def normalize_equation(equation: str) -> str:
    """
    Normalizes a chemical equation for comparison. It stays case-sensitive, as CO and Co are different.

    Args:
        equation (str): Chemical equation.
    Returns:
        The equation without whitespace or surrounding markdown, with "=" as its only arrow.
    """
    normalized = "".join(equation.split()).strip("*`$.")
    for arrow in ARROWS:
        normalized = normalized.replace(arrow, "=")
    return normalized

def extract_answer(text: str, answer_type: str) -> str | None:
    """
    Extracts the final answer of a response. Used for solution acceptance, the GRPO reward and the eval.

    Args:
        text (str): Full response.
        answer_type (str): The row's type (mcq-4-choices or filling).
    Returns:
        The uppercase letter or the normalized equation, or None if the response has no valid answer.
    """
    if answer_type not in (MCQ, FILLING):
        raise ValueError(f"Unknown answer type: {answer_type}")

    answer = find_answer_text(text)
    if not answer:
        return None
    if answer_type == MCQ:
        match = MCQ_LETTER.match(answer)
        return match.group(1).upper() if match else None
    return normalize_equation(answer) or None

def is_correct(text: str, gold: str, answer_type: str, truncated: bool = False) -> bool:
    """
    Decides whether a response is correct. This is the only definition of correct in the project.

    Args:
        text (str): Full response.
        gold (str): Correct letter (MCQ) or balanced equation (filling).
        answer_type (str): The row's type (mcq-4-choices or filling).
        truncated (bool): Whether the response hit the length cap, which always makes it incorrect.
    Returns:
        Whether the response is correct.
    """
    if not gold:
        raise ValueError("The gold answer is empty.")
    if truncated:
        return False

    answer = extract_answer(text, answer_type)
    if answer is None:
        return False
    return answer == (gold if answer_type == MCQ else normalize_equation(gold))
