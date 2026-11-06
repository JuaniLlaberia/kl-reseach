import re

MCQ = "mcq-4-choices"
FILLING = "filling"

# A line that starts with "Answer:", allowing markdown around the label ("**Answer:** B", "**Answer**: B")
ANSWER_LINE = re.compile(r"^[\W_]*answer[\W_]*?:(.*)$", re.IGNORECASE)
# One letter and nothing else, allowing punctuation and markdown around it ("(B)", "**B**", "B.")
MCQ_LETTER = re.compile(r"^[\W_]*([A-Da-d])[\W_]*$")
# Longest first, so "-->" isn't read as "-" followed by "->"
ARROWS = ("\\longrightarrow", "\\rightarrow", "\\to", "-->", "->", "⟶", "→")
HYDRATE_DOTS = ("\\cdot", "·")
# LaTeX delimiters and spacing, which carry no chemistry. A plain [ or ] does, as in Na[Al(OH)4]
LATEX_NOISE = ("\\(", "\\)", "\\[", "\\]", "\\,", "\\;", "\\!")
LATEX_SUBSCRIPT = re.compile(r"_\{?(\d+)\}?")
LATEX_TEXT = re.compile(r"\\(?:text|mathrm|ce)\{([^{}]*)\}")
# ^{+} and ^+ become {+}, the dataset's way of writing a charge
LATEX_CHARGE = re.compile(r"\^(?:\{([^{}]*)\}|([+-]))")
UNICODE_SUBSCRIPTS = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")

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
    Normalizes a chemical equation for comparison, so that only its chemistry counts and not how it's typeset.
    It stays case-sensitive, as CO and Co are different.

    Args:
        equation (str): Chemical equation, in plain text, LaTeX or with Unicode subscripts.
    Returns:
        The equation in the dataset's plain form, without whitespace, with "=" as its only arrow.
    """
    normalized = "".join(equation.split())
    for noise in LATEX_NOISE:
        normalized = normalized.replace(noise, "")
    # Subscripts go first, as \mathrm{H_{2}O} has braces inside the command
    normalized = LATEX_SUBSCRIPT.sub(r"\1", normalized)
    normalized = LATEX_TEXT.sub(r"\1", normalized)
    normalized = LATEX_CHARGE.sub(lambda match: "{" + (match.group(1) or match.group(2)) + "}", normalized)
    normalized = normalized.translate(UNICODE_SUBSCRIPTS).strip("*`$.")
    for arrow in ARROWS:
        normalized = normalized.replace(arrow, "=")
    for dot in HYDRATE_DOTS:
        normalized = normalized.replace(dot, "*")
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
