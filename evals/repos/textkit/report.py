import math

from config import REPORT_TOP_N, WORDS_PER_MINUTE
from words import count_words, tokenize


def top_words(text, n=REPORT_TOP_N):
    """Return the n most common words as (word, count) pairs.

    Sorted by count, highest first; ties are broken alphabetically.
    """
    counts = count_words(text)
    ranked = sorted(counts.items(), key=lambda pair: (pair[1], pair[0]))
    return ranked[:n]


def reading_time(text, wpm=WORDS_PER_MINUTE):
    """Return the whole number of minutes needed to read text, rounded UP. Empty text takes 0 minutes."""
    return len(tokenize(text)) // wpm


def format_report(text):
    """Return a short report: one 'word: count' line per top word."""
    return "\n".join(f"{word}: {count}" for word, count in top_words(text))