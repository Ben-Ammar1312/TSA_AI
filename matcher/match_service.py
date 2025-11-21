# matcher/match_service.py
from typing import List, Dict, Optional
import requests
from django.conf import settings

from matcher.fuzzy import (
    fuzzy_match,
    TOKEN_RATIO_OK,
    TOKEN_RATIO_MAYBE_LOW,
    TOKEN_RATIO_MAYBE_HIGH,
    refresh_candidates,
)
from matcher.llm_fallback import map_with_llm
from matcher.utils import normalize_label

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
    except Exception:
        # Do not block or crash matching on telemetry failures
        pass


def match_subjects(subjects: List[str]) -> Dict:
    """
    Returns:
      {
        "matched": [<unique target codes>],
        "coverage_pct": float,
        "trace": [{"src": str, "target": str|None, "method": str, "score": float}, ...]
      }
    """
    refresh_candidates()
    results = []
    matched_codes = set()

    for s in subjects:
        code, method, score = fuzzy_match(s)

        # Should we ask the LLM?
        call_llm = (
                method == "token_fuzzy_maybe"
                or (method == "token_fuzzy_low" and (score or 0) >= NEAR_MISS_LOW)
        )

        if call_llm:
            llm = None
            try:
                llm = map_with_llm(s)
            except Exception:
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

    return {
        "matched": sorted(matched_codes),
        "coverage_pct": round(coverage, 2),
        "trace": results,
    }