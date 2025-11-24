from typing import Optional, Tuple, List
import logging
from rapidfuzz import process, fuzz
from matcher.models import SubjectTarget, SubjectAlias, Lang
from matcher.utils import normalize_label

logger = logging.getLogger(__name__)

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
    def norm(v: str | None) -> str:
        return normalize_label(v) if v else ""

    pairs = []
    target_qs = SubjectTarget.objects.filter(is_active=True)
    target_count = target_qs.count()
    alias_fr_qs = SubjectAlias.objects.filter(target__is_active=True, language=Lang.FR).select_related('target')
    alias_en_qs = SubjectAlias.objects.filter(target__is_active=True, language=Lang.EN).select_related('target')

    # FR canonical titles
    for t in target_qs:
        pairs.append((t.norm_label or norm(t.title_fr) or norm(t.title_en) or norm(t.code), t.code, 'fr'))
    # FR aliases
    for a in alias_fr_qs:
        pairs.append((a.norm_label or norm(a.label), a.target.code, 'fr'))
    # EN aliases
    for a in alias_en_qs:
        pairs.append((a.norm_label or norm(a.label), a.target.code, 'en'))
    # de-dup norm_label→first seen keeps FR precedence
    seen, out = set(), []
    for nl, code, lg in pairs:
        if not nl:
            continue
        if (nl, code, lg) in seen:
            continue
        seen.add((nl, code, lg))
        out.append((nl, code, lg))
    logger.debug(
        "Matcher candidates rebuilt from DB targets=%s aliases_fr=%s aliases_en=%s raw_pairs=%s deduped=%s",
        target_count, alias_fr_qs.count(), alias_en_qs.count(), len(pairs), len(out)
    )
    return out

# cache at module import; refresh via management command if you edit catalog
CANDIDATES = None
def refresh_candidates():
    global CANDIDATES
    CANDIDATES = _candidate_pairs()
    logger.debug("Cached %s matcher candidates", len(CANDIDATES))

def fuzzy_match(label: str) -> Tuple[Optional[str], str, float]:
    """
    Returns (code|None, method, score 0..1).
    1) deterministic exact FR
    2) deterministic exact EN
    3) token_sort_ratio fuzzy on FR+EN (FR has more entries due to order)
    """
    q = normalize_label(label)
    if not q:
        logger.debug("Received empty/blank label: %r", label)
        return None, "empty", 0.0

    # exact FR on canonical titles
    t = SubjectTarget.objects.filter(is_active=True, norm_label=q).first()
    if t:
        logger.debug("Exact FR target hit for '%s' -> %s", label, t.code)
        return t.code, "exact_target_fr", 1.0

    # exact FR alias
    a = SubjectAlias.objects.filter(target__is_active=True, language=Lang.FR, norm_label=q) \
        .select_related('target').first()
    if a:
        logger.debug("Exact FR alias hit for '%s' -> %s", label, a.target.code)
        return a.target.code, "exact_alias_fr", 1.0

    # exact EN alias
    a = SubjectAlias.objects.filter(target__is_active=True, language=Lang.EN, norm_label=q) \
        .select_related('target').first()
    if a:
        logger.debug("Exact EN alias hit for '%s' -> %s", label, a.target.code)
        return a.target.code, "exact_alias_en", 1.0

    # fuzzy
    global CANDIDATES
    if CANDIDATES is None:
        refresh_candidates()
    labels = [nl for nl, _, _ in CANDIDATES]

    # use custom scorer
    res = process.extractOne(q, labels, scorer=_score_ensemble)
    if not res:
        logger.debug("No fuzzy match for '%s'; candidate_pool=%s", label, len(labels))
        return None, "no_fuzzy", 0.0

    matched_norm, score_pct, idx = res
    code = CANDIDATES[idx][1]
    score = score_pct / 100.0

    if score >= TOKEN_RATIO_OK:
        logger.debug("Fuzzy OK for '%s' -> %s score=%.3f norm=%s", label, code, score, matched_norm)
        return code, "token_fuzzy_ok", score
    if TOKEN_RATIO_MAYBE_LOW <= score < TOKEN_RATIO_MAYBE_HIGH:
        logger.debug("Fuzzy maybe for '%s' -> %s score=%.3f norm=%s", label, code, score, matched_norm)
        return code, "token_fuzzy_maybe", score
    logger.debug("Fuzzy low for '%s' best=%s score=%.3f norm=%s", label, code, score, matched_norm)
    return None, "token_fuzzy_low", score
