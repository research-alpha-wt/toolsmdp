from core.reward import (
    extract_answer, exact_match, compute_reward,
    normalize_answer, normalize_number,
)


class TestNormalization:

    def test_lowercase(self):
        assert normalize_answer("Paris") == "paris"

    def test_strip_articles(self):
        assert normalize_answer("the United States") == "united states"

    def test_strip_punctuation(self):
        assert normalize_answer("Hello, world!") == "hello world"

    def test_collapse_whitespace(self):
        assert normalize_answer("  hello   world  ") == "hello world"

    def test_normalize_number_integer(self):
        assert normalize_number("42") == "42"

    def test_normalize_number_float(self):
        assert normalize_number("3.14") == "3.14"

    def test_normalize_number_with_commas(self):
        assert normalize_number("1,000,000") == "1000000"

    def test_normalize_number_with_dollar(self):
        assert normalize_number("$500") == "500"

    def test_normalize_number_not_a_number(self):
        assert normalize_number("hello") is None


class TestExtractGSM8K:

    def test_hash_pattern(self):
        text = "Some reasoning...\n#### 42"
        assert extract_answer(text, "gsm8k") == "42"

    def test_hash_with_spaces(self):
        text = "Reasoning\n####   100"
        assert extract_answer(text, "gsm8k") == "100"

    def test_hash_negative(self):
        text = "So the answer is\n#### -5"
        assert extract_answer(text, "gsm8k") == "-5"

    def test_fallback_last_number(self):
        text = "The answer is 42."
        assert extract_answer(text, "gsm8k") == "42"

    def test_chain_of_thought_with_hash(self):
        text = "Step 1: 5 * 3 = 15\nStep 2: 15 + 27 = 42\n#### 42"
        assert extract_answer(text, "gsm8k") == "42"


class TestExtractMATH:

    def test_boxed(self):
        text = r"Therefore $\boxed{42}$"
        assert extract_answer(text, "math") == "42"

    def test_boxed_fraction(self):
        text = r"The answer is $\boxed{\frac{1}{2}}$"
        assert extract_answer(text, "math") == r"\frac{1}{2}"

    def test_nested_braces(self):
        text = r"$\boxed{2^{10}}$"
        assert extract_answer(text, "math") == "2^{10}"

    def test_multiple_boxed_takes_last(self):
        text = r"First $\boxed{1}$, then $\boxed{2}$"
        assert extract_answer(text, "math") == "2"

    def test_fallback_number(self):
        text = "The value is 3.14159"
        assert extract_answer(text, "math") == "3.14159"


class TestExtractQA:

    def test_answer_is_pattern(self):
        text = "After researching, the answer is Paris."
        assert extract_answer(text, "hotpotqa") == "Paris"

    def test_answer_colon_pattern(self):
        text = "Analysis complete.\nAnswer: Barack Obama"
        assert extract_answer(text, "hotpotqa") == "Barack Obama"

    def test_fallback_last_line(self):
        text = "Some reasoning\nMore analysis\nParis"
        assert extract_answer(text, "hotpotqa") == "Paris"

    def test_musique(self):
        text = "The answer is Tokyo."
        assert extract_answer(text, "musique") == "Tokyo"

    def test_2wiki(self):
        text = "Answer: William Shakespeare"
        assert extract_answer(text, "2wiki") == "William Shakespeare"

    def test_empty_text(self):
        assert extract_answer("", "hotpotqa") is None

    def test_none_text(self):
        assert extract_answer(None, "hotpotqa") is None


class TestExtractFinQA:

    def test_numeric_answer(self):
        text = "The percentage is 15.5"
        assert extract_answer(text, "finqa") == "15.5"

    def test_negative_number(self):
        text = "The change was -3.2%"
        assert extract_answer(text, "finqa") == "-3.2"

    def test_large_number(self):
        text = "Revenue was 1,234,567"
        assert extract_answer(text, "finqa") == "1234567"


class TestExactMatch:

    def test_exact_string(self):
        assert exact_match("Paris", "Paris") is True

    def test_case_insensitive(self):
        assert exact_match("paris", "Paris") is True

    def test_with_articles(self):
        assert exact_match("the United States", "United States") is True

    def test_numeric_match(self):
        assert exact_match("42", "42") is True

    def test_numeric_with_comma(self):
        assert exact_match("1000", "1,000") is True

    def test_float_integer_match(self):
        assert exact_match("42.0", "42") is True

    def test_no_match(self):
        assert exact_match("Paris", "London") is False

    def test_none_pred(self):
        assert exact_match(None, "Paris") is False

    def test_list_gold_match(self):
        assert exact_match("NYC", ["New York City", "NYC", "New York"]) is True

    def test_list_gold_no_match(self):
        assert exact_match("London", ["Paris", "Tokyo"]) is False

    def test_list_gold_normalized(self):
        assert exact_match("new york city", ["New York City", "NYC"]) is True


class TestComputeReward:

    def test_correct_gsm8k(self):
        text = "Step 1: 5+3=8\n#### 8"
        assert compute_reward(text, "8", "gsm8k") == 1.0

    def test_wrong_gsm8k(self):
        text = "Step 1: 5+3=9\n#### 9"
        assert compute_reward(text, "8", "gsm8k") == 0.0

    def test_correct_hotpotqa(self):
        text = "The answer is Paris."
        assert compute_reward(text, "Paris", "hotpotqa") == 1.0

    def test_correct_nq_with_aliases(self):
        text = "The answer is NYC."
        assert compute_reward(text, ["New York City", "NYC"], "nq") == 1.0

    def test_wrong_answer(self):
        text = "The answer is Berlin."
        assert compute_reward(text, "Paris", "hotpotqa") == 0.0

    def test_no_answer_extracted(self):
        assert compute_reward("", "Paris", "hotpotqa") == 0.0
