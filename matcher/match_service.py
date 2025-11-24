# matcher/match_service.py
from typing import List, Dict, Optional
import logging
import requests
from django.conf import settings

from matcher import fuzzy as fuzzy_module
from matcher.fuzzy import (
    fuzzy_match,
    TOKEN_RATIO_OK,
    TOKEN_RATIO_MAYBE_LOW,
    TOKEN_RATIO_MAYBE_HIGH,
    refresh_candidates,
)
from matcher.llm_fallback import map_with_llm
from matcher.utils import normalize_label

logger = logging.getLogger(__name__)

# thresholds
LLM_MIN_CONF = 0.60
NEAR_MISS_LOW = 0.75  # trigger LLM for fuzzy_low near-misses

# Optional settings (add to Django settings.py or set via env)
# SPRING_SUGGEST_URL = "http://localhost:8081/internal/suggestions"
# SPRING_SUGGEST_BEARER = "eyJhbGciOi..."  # m2m token if you protect the endpoint
# SPRING_SUGGEST_DEFAULT_LANG = "fr"

def _post_suggestion(
        src_label: str,
        proposed_target_code: str,
        score: float,
        method: str,
        language: Optional[str] = None,
) -> None:
    """
    Fire-and-forget POST to Spring suggestions endpoint. Never raise.
    """
    url = getattr(settings, "SPRING_SUGGEST_URL", None)
    if not url:
        logger.debug("SPRING_SUGGEST_URL not configured; skipping suggestion for %r", src_label)
        return
    lang = language or getattr(settings, "SPRING_SUGGEST_DEFAULT_LANG", "fr")
    payload = {
        "src_label": src_label,
        "norm_label": normalize_label(src_label),
        "proposed_target_code": proposed_target_code,
        "language": lang,
        "score": float(score or 0.0),
        "method": method,
        "reason": "auto-mapped by LLM",
    }
    headers = {"Content-Type": "application/json"}
    bearer = getattr(settings, "SPRING_SUGGEST_BEARER", None)
    if bearer:
        headers["Authorization"] = f"Bearer {bearer}"
    try:
        requests.post(url, json=payload, headers=headers, timeout=5)
        logger.debug(
            "Posted mapping suggestion to Spring url=%s target=%s score=%.3f method=%s",
            url, proposed_target_code, score, method,
        )
    except Exception as exc:
        # Do not block or crash matching on telemetry failures
        logger.warning("Failed to post suggestion to Spring: %s", exc, exc_info=True)


def match_subjects(subjects: List[str]) -> Dict:
    """
    Returns:
      {
        "matched": [<unique target codes>],
        "coverage_pct": float,
        "trace": [{"src": str, "target": str|None, "method": str, "score": float}, ...]
      }
    """
    logger.debug("Starting subject matching for %s subjects: %r", len(subjects), subjects)
    refresh_candidates()
    try:
        logger.debug("Candidate cache size after refresh: %s", len(fuzzy_module.CANDIDATES or []))
    except Exception:
        logger.debug("Could not compute candidate cache size (will continue).")
    results = []
    matched_codes = set()

    for s in subjects:
        logger.debug("Matching subject label=%r", s)
        code, method, score = fuzzy_match(s)

        # Should we ask the LLM?
        call_llm = (
                method == "token_fuzzy_maybe"
                or (method == "token_fuzzy_low" and (score or 0) >= NEAR_MISS_LOW)
        )

        if call_llm:
            llm = None
            try:
                logger.debug("Calling LLM fallback for %r (method=%s score=%.3f)", s, method, score or 0)
                llm = map_with_llm(s)
            except Exception as exc:
                logger.exception("LLM fallback failed for %r: %s", s, exc)
                results.append({
                    "src": s,
                    "target": None,
                    "method": "llm_error",
                    "score": round(float(score or 0.0), 3),
                })
                continue

            if llm and llm.get("target_id"):
                conf = float(llm.get("confidence", 0) or 0)
                if conf >= LLM_MIN_CONF:
                    code, method, score = llm["target_id"], "llm_fallback", conf
                    # Suggest to Spring so an alias can be approved and created later
                    _post_suggestion(
                        src_label=s,
                        proposed_target_code=code,
                        score=conf,
                        method="llm_fallback",
                        language=getattr(settings, "SPRING_SUGGEST_DEFAULT_LANG", "fr"),
                    )
                else:
                    method = "llm_reject"
            else:
                method = "llm_none"

        logger.debug(
            "Match result for %r -> target=%s method=%s score=%.3f",
            s, code, method, float(score or 0.0),
        )
        results.append({
            "src": s,
            "target": code,
            "method": method,
            "score": round(float(score or 0.0), 3),
        })
        if code:
            matched_codes.add(code)

    # Coverage over active targets
    from matcher.models import SubjectTarget
    total_targets = SubjectTarget.objects.filter(is_active=True).count()
    coverage = (len(matched_codes) / total_targets * 100.0) if total_targets else 0.0
    logger.debug(
        "Matching finished: matched_codes=%s total_targets=%s coverage_pct=%.2f",
        sorted(matched_codes), total_targets, coverage
    )

    return {
        "matched": sorted(matched_codes),
        "coverage_pct": round(coverage, 2),
        "trace": results,
    }
