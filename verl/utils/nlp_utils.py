# ...existing code...
from collections import defaultdict
import json
import math
import re
from functools import lru_cache

# A small stopword list for a cheap "content word ratio" / conceptual density proxy.
# (You can expand/replace this via config if desired.)
_EN_STOPWORDS = {
    "a","an","the","and","or","but","if","then","else","when","while","of","to","in","on","for","from","by","with",
    "as","at","into","about","over","under","after","before","between","through","during",
    "is","am","are","was","were","be","been","being","do","does","did","doing","have","has","had","having",
    "i","me","my","mine","we","us","our","ours","you","your","yours","he","him","his","she","her","hers","it","its",
    "they","them","their","theirs",
    "this","that","these","those","there","here",
    "not","no","nor","so","too","very","can","could","may","might","must","shall","should","will","would",
    "what","which","who","whom","why","how",
}

_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")

def _simple_word_tokenize(text: str) -> list[str]:
    if not text:
        return []
    return [m.group(0).lower() for m in _WORD_RE.finditer(text)]

def _count_syllables_heuristic(word: str) -> int:
    """
    Heuristic syllable counter (good enough for batch-level readability trends).
    """
    w = (word or "").lower()
    w = re.sub(r"[^a-z]", "", w)
    if not w:
        return 0
    vowels = "aeiouy"
    syllables = 0
    prev_is_vowel = False
    for ch in w:
        is_vowel = ch in vowels
        if is_vowel and not prev_is_vowel:
            syllables += 1
        prev_is_vowel = is_vowel
    # silent trailing 'e'
    if w.endswith("e") and syllables > 1:
        syllables -= 1
    return max(1, syllables)

def _split_sentences_heuristic(text: str) -> int:
    if not text:
        return 0
    # Count sentence-like segments.
    parts = re.split(r"[.!?]+", text)
    return sum(1 for p in parts if p.strip())

def _readability_metrics(text: str) -> dict:
    """
    Returns:
      - flesch_reading_ease
      - fk_grade (Flesch–Kincaid Grade Level)
    """
    words = _simple_word_tokenize(text)
    n_words = len(words)
    n_sent = _split_sentences_heuristic(text)
    if n_words == 0 or n_sent == 0:
        return {"flesch_reading_ease": 0.0, "fk_grade": 0.0}

    syllables = sum(_count_syllables_heuristic(w) for w in words)
    wps = n_words / max(1, n_sent)  # words per sentence
    spw = syllables / max(1, n_words)  # syllables per word

    # Standard formulas
    flesch = 206.835 - 1.015 * wps - 84.6 * spw
    fkgl = 0.39 * wps + 11.8 * spw - 15.59

    # Keep bounded-ish for logging sanity
    return {
        "flesch_reading_ease": float(flesch),
        "fk_grade": float(fkgl),
    }

def _vocab_metrics(words: list[str]) -> dict:
    """
    Vocabulary / lexical statistics.
    """
    n = len(words)
    if n == 0:
        return {
            "word_count": 0,
            "unique_word_count": 0,
            "type_token_ratio": 0.0,
            "lexical_entropy": 0.0,
            "avg_word_len": 0.0,
        }
    from collections import Counter
    c = Counter(words)
    uniq = len(c)
    ttr = uniq / n
    # Shannon entropy over word distribution
    ent = 0.0
    for freq in c.values():
        p = freq / n
        ent -= p * math.log(p + 1e-12)
    avg_len = sum(len(w) for w in words) / n
    return {
        "word_count": int(n),
        "unique_word_count": int(uniq),
        "type_token_ratio": float(ttr),
        "lexical_entropy": float(ent),
        "avg_word_len": float(avg_len),
    }

def _conceptual_density_proxy(words: list[str]) -> float:
    """
    Cheap proxy: content-word ratio (alphabetic, non-stopword).
    """
    if not words:
        return 0.0
    content = [w for w in words if w.isalpha() and (w not in _EN_STOPWORDS)]
    return float(len(content) / len(words))

@lru_cache(maxsize=8)
def _load_category_lexicon(path: str | None) -> dict[str, set[str]]:
    """
    Load a category lexicon from JSON.

    Supported JSON formats:
    1) { "joy": ["happy", ...], "anger": ["mad", ...] }
    2) { "happy": ["joy"], "mad": ["anger","negative"] }  (word -> categories)

    Returns: {category: set(words)}
    """
    if not path:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception:
        return {}

    cat2words: dict[str, set[str]] = defaultdict(set)

    if isinstance(obj, dict):
        # Heuristic: if values look like word lists => category->words
        sample_val = next(iter(obj.values()), None)
        if isinstance(sample_val, list) and (len(sample_val) == 0 or isinstance(sample_val[0], str)):
            for cat, words in obj.items():
                if not isinstance(cat, str) or not isinstance(words, list):
                    continue
                for w in words:
                    if isinstance(w, str) and w:
                        cat2words[cat].add(w.lower())
        else:
            # Assume word->categories
            for w, cats in obj.items():
                if not isinstance(w, str):
                    continue
                if isinstance(cats, str):
                    cats = [cats]
                if not isinstance(cats, list):
                    continue
                for cat in cats:
                    if isinstance(cat, str) and cat:
                        cat2words[cat].add(w.lower())

    return dict(cat2words)

def _emotion_category_rates(words: list[str], cat2words: dict[str, set[str]]) -> dict[str, float]:
    """
    Returns per-category rates in [0,1] as fraction of words that match the category.
    """
    if not words or not cat2words:
        return {}
    n = len(words)
    counts = {cat: 0 for cat in cat2words.keys()}
    for w in words:
        for cat, lex in cat2words.items():
            if w in lex:
                counts[cat] += 1
    return {f"emotion_rate/{cat}": float(cnt / n) for cat, cnt in counts.items()}

def _count_tokens(text: str, tokenizer=None) -> int:
    """
    Count tokens in `text` using a HF tokenizer if provided; fall back to a crude heuristic.
    """
    if text is None:
        return 0
    text = str(text)
    if tokenizer is not None:
        # HF tokenizers: tokenizer.encode(...) returns a list of token ids
        return len(tokenizer.encode(text, add_special_tokens=False))
    # Fallback: rough proxy (words)
    return len(text.split())


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _text_after_thinking(text: str) -> str:
    """
    If text contains <think>...</think>, return only the content after </think>.
    Otherwise return the full text.
    """
    if not text or not isinstance(text, str):
        return text or ""
    match = _THINK_RE.search(text)
    if match:
        return text[match.end():].strip()
    if "</think>" in text:
        return text.split("</think>", 1)[-1].strip()
    return text


def _detect_script(text: str) -> str:
    """
    Return the dominant Unicode script of *text* as one of:
    "latin", "cjk", "cyrillic", "arabic", "devanagari", or "other".
    Only alphabetic / ideographic codepoints are counted; digits,
    punctuation, and whitespace are ignored.
    """
    counts: dict[str, int] = {}
    for ch in text:
        cp = ord(ch)
        if 0x0041 <= cp <= 0x024F or 0x1E00 <= cp <= 0x1EFF:
            bucket = "latin"
        elif (0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF
              or 0x3040 <= cp <= 0x30FF or 0xAC00 <= cp <= 0xD7AF):
            bucket = "cjk"
        elif 0x0400 <= cp <= 0x04FF:
            bucket = "cyrillic"
        elif 0x0600 <= cp <= 0x06FF or 0x0750 <= cp <= 0x077F:
            bucket = "arabic"
        elif 0x0900 <= cp <= 0x097F:
            bucket = "devanagari"
        else:
            continue
        counts[bucket] = counts.get(bucket, 0) + 1
    if not counts:
        return "other"
    return max(counts, key=counts.get)


def compute_language_penalty(
    prompt: str,
    response: str,
    weight: float = 0.0,
    exclude_thinking: bool = False,
) -> tuple[float, str, str]:
    """
    Penalize responses whose dominant script differs from the prompt's.

    Returns (penalty, prompt_script, response_script).
    penalty is in [-weight, 0]: full penalty when the scripts mismatch,
    zero when they match or when detection is inconclusive.
    """
    if weight <= 0.0:
        return 0.0, "", ""

    if exclude_thinking:
        prompt = _text_after_thinking(prompt)
        response = _text_after_thinking(response)
    prompt_script = _detect_script(prompt or "")
    resp_text = _THINK_RE.sub("", response or "").strip()
    response_script = _detect_script(resp_text)

    if prompt_script == "other" or response_script == "other":
        return 0.0, prompt_script, response_script
    if prompt_script == response_script:
        return 0.0, prompt_script, response_script
    return -weight, prompt_script, response_script


def compute_length_penalty(
    response: str,
    tokenizer=None,
    max_tokens: int = 512,
    weight: float = 0.0,
    exclude_thinking: bool = False,
) -> tuple[float, dict[str, int]]:
    """
    Returns (penalty_component, token_stats).

    token_stats contains:
      - response_n_token: token count of the full response
      - thinking_n_token: tokens inside/attributed to thinking content
      - answer_n_token: tokens in the post-thinking answer content

    penalty_component is <= 0.0 (so you can add it to total_reward directly).
    """
    response_text = "" if response is None else str(response)
    answer_text = _text_after_thinking(response_text)

    response_n = _count_tokens(response_text, tokenizer)
    answer_n = _count_tokens(answer_text, tokenizer)
    thinking_n = max(0, response_n - answer_n)

    token_stats = {
        "response_n_tokens": int(response_n),
        "thinking_n_tokens": int(thinking_n),
        "answer_n_tokens": int(answer_n),
    }

    n_for_penalty = answer_n if exclude_thinking else response_n

    if weight <= 0.0 or max_tokens is None or max_tokens <= 0:
        return 0.0, token_stats

    overflow = max(0, n_for_penalty - max_tokens)
    # Linear penalty in [-weight, 0] once overflow reaches max_tokens again (you can change this)
    penalty = -weight * (overflow / max_tokens)
    return float(penalty), token_stats