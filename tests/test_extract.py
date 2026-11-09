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
    # Rejected until 07/11/26, when Qwen2.5-1.5B-Instruct failed the format on 54% of MCQ responses, mostly in these ways
    ("Answer:\nB", "B"),
    ("Answer: B) 18", "B"),
    ("Final answer: B", "B"),
    ("Reasoning.\n\nB. 22", "B"),
    ("Answer: <D> 12", "D"),
    ("The correct answer is: B", "B"),
    # A label beats a later line without one
    ("Answer: B\nHope this helps!", "B"),
])
def test_mcq_valid_answers(text, expected):
    assert extract_answer(text, MCQ) == expected

@pytest.mark.parametrize("text", [
    "",
    "The answer is B.",
    "I think it's B",
    "Answer:",
    "Answer: A or B",
    "Answer: A, B",
    "Answer: E",
    "Answer: 18",
    "Answer: <letter>",
    # A later invalid line isn't rescued by an earlier valid one
    "Answer: B\nAnswer: not sure",
    # A word that starts with a letter isn't an option
    "Reasoning.\nA total of 16 rotatable bonds.",
    "Answer: Cannot be determined",
    "Answer: B and C",
])
def test_mcq_invalid_answers(text):
    assert extract_answer(text, MCQ) is None

# Real endings of Qwen2.5-1.5B-Instruct responses on chemistry train prompts (07/11/26), with the letter each one states
@pytest.mark.parametrize("ending, expected", [
    ("Therefore, there are a total of 2 possible rotatable bonds in the molecule.\n\nFinal answer: 23", None),
    ("Thus, the total number of hydrogen bond donors is 18.\n\n**Final answer: A. 16**", "A"),
    ("we arrive at the answer of 17 hydrogen bond donors.\n\nFinal line: D. 17", "D"),
    ("Therefore, the correct answer is 16.\n\nD. 16", "D"),
    ("Therefore, the correct answer is:\n\nC. 4", "C"),
    ("The correct answer is: <letter>B. 19", None),
    ("C(=O)N1'.\n\nFinal line:\nD. 21", "D"),
    ("Therefore, the answer is B. 21.\n\n-----------------\nD. 7", "D"),
    ("totaling 16 rotatable bonds.", None),
    ("Final answer: <117>", None),
    ("So, **B. 495.700** is the molar mass of the molecule.", None),
    ("The molar weight of this molecule is 1467.700. Answer: A", "A"),
    ("Therefore, the molar weight of the molecule is **117.120 g/mol**. \n\nFinal answer: B. 117.120", "B"),
    ("This corresponds to option B.", None),
    ("Final Answer: Cannot be determined from the provided information.", None),
    ("**Final Answer:** A. 2225.600", "A"),
    ('print("Final Answer: A")\n```\n\nFinal Answer: A', "A"),
    ("B. 2880.600", "B"),
    ("Answer:\nC. 3278.900", "C"),
    ("Answer:\nUnable to determine the correct match.", None),
    ("Answer: D (4)", "D"),
    ("Answer: F. The question cannot be correctly answered with the given information.", None),
])
def test_real_qwen_mcq_endings(ending, expected):
    assert extract_answer(f"Some reasoning.\n\n{ending}", MCQ) == expected

# Real filling endings from the same responses: the equation is read wherever it was stated.
# Only the start is checked in the first, as the model's own LaTeX is broken further on
@pytest.mark.parametrize("ending, start", [
    (r"Final line: Answer: \[ \text{MgSO}_4 + 2\text{NH}_4\text{Cl} \rightarrow \text{MgCl}_2 + 2\text{(NH}_4\text){}_2\text{SO}_4 \]", "MgSO4+2NH4Cl=MgCl2+"),
    ("The balanced equation is:\nCr₂O₃ + 2Al = Al₂O₃ + 2Cr", "Cr2O3+2Al=Al2O3+2Cr"),
])
def test_real_qwen_filling_endings(ending, start):
    assert extract_answer(f"Balancing it.\n\n{ending}", FILLING).startswith(start)

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
