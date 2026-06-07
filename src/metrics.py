import re

from jiwer import cer, process_words


WHITESPACE_PATTERN = re.compile(r"\s+")


def normalize_text(text):
    if text is None or text != text:
        return ""
    if not isinstance(text, str):
        text = str(text)
    normalized = text.strip().lower()
    normalized = WHITESPACE_PATTERN.sub(" ", normalized)
    return normalized


def tokenize(text):
    normalized = normalize_text(text)
    if not normalized:
        return []
    return normalized.split(" ")


def compute_text_metrics(reference, hypothesis):
    normalized_reference = normalize_text(reference)
    normalized_hypothesis = normalize_text(hypothesis)

    reference_tokens = tokenize(reference)
    hypothesis_tokens = tokenize(hypothesis)

    first_word_match = False
    last_word_match = False
    if reference_tokens and hypothesis_tokens:
        first_word_match = reference_tokens[0] == hypothesis_tokens[0]
        last_word_match = reference_tokens[-1] == hypothesis_tokens[-1]

    return {
        "cer": cer(normalized_reference, normalized_hypothesis),
        "exact_match_rate": float(normalized_reference == normalized_hypothesis),
        "first_word_match_rate": float(first_word_match),
        "first_word_error_rate": float(not first_word_match),
        "last_word_match_rate": float(last_word_match),
        "last_word_error_rate": float(not last_word_match),
        "normalized_reference": normalized_reference,
        "normalized_hypothesis": normalized_hypothesis,
    }


def compute_word_csid(reference, hypothesis):
    output = process_words(normalize_text(reference), normalize_text(hypothesis))
    return {
        "word_c": float(output.hits),
        "word_s": float(output.substitutions),
        "word_d": float(output.deletions),
        "word_i": float(output.insertions),
    }
