# matcher/match_service.py
from typing import List, Dict
from matcher.fuzzy import (
    fuzzy_match,
    TOKEN_RATIO_OK,
    TOKEN_RATIO_MAYBE_LOW,
    TOKEN_RATIO_MAYBE_HIGH,
    refresh_candidates,
)
from matcher.llm_fallback import map_with_llm

# thresholds
LLM_MIN_CONF = 0.60
NEAR_MISS_LOW = 0.75  # trigger LLM for fuzzy_low near-misses


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
            llm = None  # <-- ensure defined
            try:
                llm = map_with_llm(s)
            except Exception as e:
                # record the error path explicitly
                results.append({
                    "src": s,
                    "target": None,
                    "method": "llm_error",
                    "score": round(float(score or 0.0), 3),
                })
                continue  # go to next subject

            # LLM responded
            if llm and llm.get("target_id"):
                conf = float(llm.get("confidence", 0) or 0)
                if conf >= LLM_MIN_CONF:
                    code, method, score = llm["target_id"], "llm_fallback", conf
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