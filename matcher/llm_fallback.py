# matcher/llm_fallback.py
import json
from typing import Optional, Dict
import requests
from matcher.models import SubjectTarget, SubjectAlias, Lang

# Ollama settings
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODEL = "llama3.1:8b-instruct-q4_K_M"  # fits M1 16GB well (q4_K)

def _target_lines() -> str:
    """
    Build a compact closed list: `code : French title | (1 EN alias if exists)`
    We keep it short to help the small context window and steer toward FR.
    """
    rows = []
    qs = SubjectTarget.objects.filter(is_active=True).order_by("code")
    for t in qs:
        # canonical FR title
        fr_title = t.title_fr or ""
        # one EN alias (optional)
        en_alias = SubjectAlias.objects.filter(target=t, language=Lang.EN) \
            .values_list("label", flat=True)[:1]
        if en_alias:
            rows.append(f"{t.code} : {fr_title} | {en_alias[0]}")
        else:
            rows.append(f"{t.code} : {fr_title}")
    return "\n".join(rows)

PROMPT_SYS = """Tu dois mapper un intitulé de matière à UN identifiant parmi la liste FERMÉE ci-dessous.

RÈGLES:
- Privilégie le FRANÇAIS et le sens (pas seulement l’égalité exacte).
- Si aucun identifiant ne convient clairement, renvoie null.
- Format de sortie STRICT et SEUL autorisé:
  {"target_id": "<code|null>", "confidence": <nombre entre 0 et 1>}

EXEMPLES:
- Sujet: "Théorie des automates"
  Sortie: {"target_id": "info.theorie.langages", "confidence": 0.72}
- Sujet: "Algo et structures de donnée"
  Sortie: {"target_id": "info.algo.ds", "confidence": 0.68}
"""

def map_with_llm(subject_name: str) -> Optional[Dict]:
    """
    Ask the local Ollama model to map a free-text subject to one target code.
    Returns {"target_id": str|None, "confidence": float} or None on hard failure.
    """
    payload = {
        "model": MODEL,
        "prompt": (
            f"{PROMPT_SYS}\n\n"
            f"LISTE DES CIBLES:\n{_target_lines()}\n\n"
            f"Sujet: \"{subject_name}\"\n"
            f"Réponse:"
        ),
        "format": "json",
        "options": {"num_ctx": 4096, "temperature": 0.1},
        "stream": False,
    }

    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=120)
        r.raise_for_status()
        resp = r.json()  # {'response': '<json-string>', ...}
        data = json.loads(resp.get("response", "{}"))
    except Exception:
        # Network/model error or malformed JSON from the model
        return None

    tid_raw = (data.get("target_id") or "").strip()
    conf = float(data.get("confidence", 0) or 0)

    # Normalize empty -> None
    tid = tid_raw if tid_raw else None
    return {"target_id": tid, "confidence": conf}