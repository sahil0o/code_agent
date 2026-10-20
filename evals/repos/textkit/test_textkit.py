from report import format_report, reading_time, top_words
from words import count_words, tokenize

SAMPLE = "Cats and dogs. Cats chase mice, dogs chase cats! The owl is wise."


def test_tokenize():
    assert tokenize("Hello, World 42!") == ["hello", "world"]


def test_count_words_keeps_words_of_min_length():
    counts = count_words(SAMPLE)
    assert counts["owl"] == 1


def test_count_words_skips_stopwords_and_short_words():
    counts = count_words(SAMPLE)
    assert "the" not in counts and "is" not in counts


def test_top_words_order():
    assert top_words(SAMPLE, n=3) == [("cats", 3), ("chase", 2), ("dogs", 2)]


def test_reading_time_rounds_up():
    assert reading_time("word " * 201) == 2


def test_reading_time_empty():
    assert reading_time("") == 0


def test_format_report():
    assert format_report("apple apple pear").splitlines()[0] == "apple: 2"