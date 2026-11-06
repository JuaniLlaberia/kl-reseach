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

# Real GPT-4o answer lines from the 50-prompt test (06/11/26), which the first version of the matcher wrongly rejected
@pytest.mark.parametrize("line, gold", [
    (r"Answer: \( 2 \text{Sn} + \text{O}_2 = 2 \text{SnO} \)", "2Sn + O2 = 2SnO"),
    (r"Answer: \( 2\text{KOH} + \text{SO}_2 = \text{K}_2\text{SO}_3 + \text{H}_2\text{O} \)", "2KOH + SO2 = K2SO3 + H2O"),
    ("Answer: CaCO₃(s) + 2 HCl(l) → CaCl₂(s) + CO₂(g) + H₂O(l)", "CaCO3(s) + 2HCl(l) = CaCl2(s) + CO2(g) + H2O(l)"),
    (r"Answer: \( 2 \text{MnCl}_3 + 3 \text{K}_2\text{CO}_3 = 6 \text{KCl} + (\text{CO}_3)_3\text{Mn}_2 \)", "2MnCl3 + 3K2CO3 = 6KCl + (CO3)3Mn2"),
])
def test_real_correct_answers_are_accepted(line, gold):
    assert is_correct(f"Some reasoning.\n\n{line}", gold, FILLING)

# The two answers from that test that really were wrong, which must stay wrong
@pytest.mark.parametrize("line, gold", [
    ("Answer: 2 KOH + KNO2 + KMnO4 = K2MnO4 + KNO3 + H2O", "2KOH + KNO2 + 2KMnO4 = 2K2MnO4 + KNO3 + H2O"),
    (r"Answer: \(\text{N}_2\text{H}_4 + \text{N}_2\text{O}_4 = 2\text{N}_2 + 2\text{H}_2\text{O}\)", "2N2H4 + N2O4 = 3N2 + 4H2O"),
])
def test_real_wrong_answers_are_rejected(line, gold):
    assert not is_correct(line, gold, FILLING)

@pytest.mark.parametrize("typeset, plain", [
    (r"$2\mathrm{H_{2}O}$", "2H2O"),
    (r"\[ \ce{H2O} \]", "H2O"),
    (r"**\( \text{O}_{2} \)**", "O2"),
    (r"A\,+\,B \rightarrow C", "A+B=C"),
    (r"A \to B", "A=B"),
    (r"A \longrightarrow B", "A=B"),
    (r"\text{CuSO}_4 \cdot 5\text{H}_2\text{O}", "CuSO4*5H2O"),
    ("CuSO₄·5H₂O", "CuSO4*5H2O"),
    (r"\text{Ag}^{+}", "Ag{+}"),
    (r"\text{OH}^-", "OH{-}"),
    (r"\text{Al}^{3+}", "Al{3+}"),
])
def test_typesetting_is_removed(typeset, plain):
    assert normalize_equation(typeset) == plain

@pytest.mark.parametrize("gold", [
    "ClO3{-} + 2Ag{+} + 2OH{-} = ClO4{-} + 2Ag + H2O",
    "4Co(NO3)2*6H2O + 19NH3 + NaNO3 = 2[(NH3)5Co(O2)Co(NH3)5](NO3)4*2H2O + NaOH + 18H2O",
    "6NH4F + 2NaOH + Na[Al(OH)4] = Na3AlF6 + 6NH3 + 6H2O",
    "6OH + I({-}) = IO3({-}) + 3H2O",
])
def test_plain_brackets_and_charges_in_golds_survive(gold):
    assert normalize_equation(gold) == "".join(gold.split())
    assert normalize_equation(normalize_equation(gold)) == normalize_equation(gold)

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
