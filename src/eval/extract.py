import re

MCQ = "mcq-4-choices"
FILLING = "filling"

# A label for the final answer anywhere in a line: "Answer:", "Final answer:", "The correct answer is:", or "Final line:",
# which small models copy from the prompt. Markdown may sit between the word and the colon ("**Answer**: B")
ANSWER_LABEL = re.compile(r"\b(?:answer(?:\s+is)?|final\s+line)\b[*_\s]*:", re.IGNORECASE)
# A letter at the start, alone ("B", "(B)", "**B**", "B.") or followed by its option's text ("B. 22", "D (4)", "<D> 12").
# A letter followed by a word isn't one ("A total of", "Cannot be", "A or B")
MCQ_LETTER = re.compile(r"^[\W_]*([A-Da-d])(?:[\W_]*$|\s*[.):\]>]|\s+\()")
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
    Finds a response's final answer: what follows the last answer label, or the next non-empty line if the label
    stands alone. With no label anywhere, it's the last non-empty line, as small models often end on the answer alone.
    So the eval measures the task rather than how closely the format was followed, which SFT learns from its targets
    and GRPO has to find.

    Args:
        text (str): Full response.
    Returns:
        The answer's text, which may be empty, or None if the response is empty.
    """
    lines = [line.strip() for line in text.splitlines()]
    for index in range(len(lines) - 1, -1, -1):
        labels = list(ANSWER_LABEL.finditer(lines[index]))
        if labels:
            answer = lines[index][labels[-1].end():].strip()
            return answer or next((line for line in lines[index + 1:] if line), "")
    return next((line for line in reversed(lines) if line), None)

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
