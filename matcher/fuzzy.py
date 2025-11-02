from typing import Optional, Tuple, List
from rapidfuzz import process, fuzz
from matcher.models import SubjectTarget, SubjectAlias, Lang
from matcher.utils import normalize_label

# thresholds
TOKEN_RATIO_OK = 0.90      # accept if ≥ 0.90
TOKEN_RATIO_MAYBE_LOW = 0.80   # send to LLM if between 0.85..0.90
TOKEN_RATIO_MAYBE_HIGH = 0.90

def _score_ensemble(q: str, s: str, *, score_cutoff: float = 0) -> float:
    """Return a single 0..100 score. Must accept score_cutoff."""
    a = fuzz.token_sort_ratio(q, s, score_cutoff=score_cutoff)
    b = fuzz.token_set_ratio(q, s, score_cutoff=score_cutoff)
    c = fuzz.WRatio(q, s, score_cutoff=score_cutoff)
    return max(a, b, c)


def _candidate_pairs() -> List[tuple[str, str, str]]:
    """
    Returns [(norm_label, code, lang)], FR first then EN.
    Includes canonical FR titles and aliases.
    """
    pairs = []
    # FR canonical titles
    for t in SubjectTarget.objects.filter(is_active=True):
        pairs.append((t.norm_label, t.code, 'fr'))
    # FR aliases
    for a in SubjectAlias.objects.filter(target__is_active=True, language=Lang.FR).select_related('target'):
        pairs.append((a.norm_label, a.target.code, 'fr'))
    # EN aliases
    for a in SubjectAlias.objects.filter(target__is_active=True, language=Lang.EN).select_related('target'):
        pairs.append((a.norm_label, a.target.code, 'en'))
    # de-dup norm_label→first seen keeps FR precedence
    seen, out = set(), []
    for nl, code, lg in pairs:
        if (nl, code, lg) in seen:
            continue
        seen.add((nl, code, lg))
        out.append((nl, code, lg))
    return out

# cache at module import; refresh via management command if you edit catalog
CANDIDATES = None
def refresh_candidates():
    global CANDIDATES
    CANDIDATES = _candidate_pairs()

def fuzzy_match(label: str) -> Tuple[Optional[str], str, float]:
    """
    Returns (code|None, method, score 0..1).
    1) deterministic exact FR
    2) deterministic exact EN
    3) token_sort_ratio fuzzy on FR+EN (FR has more entries due to order)
    """
    q = normalize_label(label)
    if not q:
        return None, "empty", 0.0

    # exact FR on canonical titles
    t = SubjectTarget.objects.filter(is_active=True, norm_label=q).first()
    if t:
        return t.code, "exact_target_fr", 1.0

    # exact FR alias
    a = SubjectAlias.objects.filter(target__is_active=True, language=Lang.FR, norm_label=q) \
        .select_related('target').first()
    if a:
        return a.target.code, "exact_alias_fr", 1.0

    # exact EN alias
    a = SubjectAlias.objects.filter(target__is_active=True, language=Lang.EN, norm_label=q) \
        .select_related('target').first()
    if a:
        return a.target.code, "exact_alias_en", 1.0

    # fuzzy
    global CANDIDATES
    if CANDIDATES is None:
        refresh_candidates()
    labels = [nl for nl, _, _ in CANDIDATES]

    # use custom scorer
    res = process.extractOne(q, labels, scorer=_score_ensemble)
    if not res:
        return None, "no_fuzzy", 0.0

    matched_norm, score_pct, idx = res
    code = CANDIDATES[idx][1]
    score = score_pct / 100.0

    if score >= TOKEN_RATIO_OK:
        return code, "token_fuzzy_ok", score
    if TOKEN_RATIO_MAYBE_LOW <= score < TOKEN_RATIO_MAYBE_HIGH:
        return code, "token_fuzzy_maybe", score
    return None, "token_fuzzy_low", score