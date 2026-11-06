import pytest

from src.eval.extract import FILLING, MCQ, extract_answer, is_correct, normalize_equation

@pytest.mark.parametrize("text, expected", [
    ("Some reasoning.\nAnswer: B", "B"),
    ("Answer: B", "B"),
    ("**Answer: B**", "B"),
    ("**Answer:** B", "B"),
    ("**Answer**: B", "B"),
    ("Answer: (B)", "B"),
    ("Answer: B.", "B"),
    ("Answer: **B**", "B"),
    ("answer: b", "B"),
    ("ANSWER: B", "B"),
    ("  Answer:   B  ", "B"),
    ("Answer: B\nHope this helps!", "B"),
    # The last Answer line wins, as models often correct themselves
    ("Answer: A\nWait, that's wrong.\nAnswer: C", "C"),
])
def test_mcq_valid_answers(text, expected):
    assert extract_answer(text, MCQ) == expected

@pytest.mark.parametrize("text", [
    "",
    "The answer is B.",
    "I think it's B",
    "Answer:",
    "Answer:\nB",
    "Answer: A or B",
    "Answer: A, B",
    "Answer: E",
    "Answer: B) 18",
    "Answer: 18",
    "Answer: <letter>",
    "Final answer: B",
    # A later invalid line isn't rescued by an earlier valid one
    "Answer: B\nAnswer: not sure",
])
def test_mcq_invalid_answers(text):
    assert extract_answer(text, MCQ) is None

GOLD_EQUATION = "Na2CrO4(aq) + 2LiBr(aq) = Li2CrO4(aq) + 2NaBr(s)"

@pytest.mark.parametrize("answer", [
    GOLD_EQUATION,
    "Na2CrO4(aq)+2LiBr(aq)=Li2CrO4(aq)+2NaBr(s)",
    "Na2CrO4(aq)  +  2 LiBr(aq) = Li2CrO4(aq) + 2 NaBr(s)",
    "Na2CrO4(aq) + 2LiBr(aq) -> Li2CrO4(aq) + 2NaBr(s)",
    "Na2CrO4(aq) + 2LiBr(aq) --> Li2CrO4(aq) + 2NaBr(s)",
    "Na2CrO4(aq) + 2LiBr(aq) → Li2CrO4(aq) + 2NaBr(s)",
    f"**{GOLD_EQUATION}**",
    f"`{GOLD_EQUATION}`",
    f"${GOLD_EQUATION}$",
    f"{GOLD_EQUATION}.",
])
def test_filling_correct_after_normalizing(answer):
    assert is_correct(f"Balancing it.\nAnswer: {answer}", GOLD_EQUATION, FILLING)

@pytest.mark.parametrize("answer", [
    # Wrong coefficient
    "Na2CrO4(aq) + LiBr(aq) = Li2CrO4(aq) + NaBr(s)",
    # Formulas are case-sensitive
    "na2cro4(aq) + 2libr(aq) = li2cro4(aq) + 2nabr(s)",
    # Phases are part of the answer
    "Na2CrO4 + 2LiBr = Li2CrO4 + 2NaBr",
    # The order of terms has to be kept
    "2LiBr(aq) + Na2CrO4(aq) = Li2CrO4(aq) + 2NaBr(s)",
    "",
])
def test_filling_incorrect(answer):
    assert not is_correct(f"Answer: {answer}", GOLD_EQUATION, FILLING)

def test_normalization_keeps_the_closing_parenthesis():
    assert normalize_equation("A(aq) = B(s).") == "A(aq)=B(s)"

def test_mcq_correct_and_incorrect():
    assert is_correct("Reasoning\nAnswer: B", "B", MCQ)
    assert not is_correct("Reasoning\nAnswer: A", "B", MCQ)

def test_truncated_response_is_never_correct():
    assert not is_correct("Reasoning\nAnswer: B", "B", MCQ, truncated=True)

def test_empty_gold_fails():
    with pytest.raises(ValueError):
        is_correct("Answer: B", "", MCQ)

def test_unknown_type_fails():
    with pytest.raises(ValueError):
        extract_answer("Answer: B", "open-ended")
