import re
from collections import Counter

from config import MIN_WORD_LENGTH, STOPWORDS


def tokenize(text):
    """Split text into lowercase words, ignoring punctuation and numbers."""
    return re.findall(r"[a-z]+", text.lower())


def count_words(text):
    """Count words, skipping stopwords and words with fewer than MIN_WORD_LENGTH letters.

    Words with exactly MIN_WORD_LENGTH letters are counted.
    """
    words = [w for w in tokenize(text) if len(w) > MIN_WORD_LENGTH and w not in STOPWORDS]
    return Counter(words)