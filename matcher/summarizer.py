import logging
import os
import tempfile
from typing import Optional

import requests

from matcher import llm_fallback

logger = logging.getLogger(__name__)

# Optional local Whisper (open source, no API key). Install with: pip install openai-whisper ffmpeg-python
try:
    import whisper  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    whisper = None

_whisper_model = None


def transcribe_audio(path: str) -> Optional[str]:
    """
    Best-effort transcription using local Whisper model (no API key).
    """
    global _whisper_model
    if whisper is None:
        logger.warning("whisper not installed; skipping transcription")
        return None
    try:
        if _whisper_model is None:
            # small balances speed/quality; change if you have GPU and want better quality
            _whisper_model = whisper.load_model("small")
        result = _whisper_model.transcribe(path, fp16=False)
        text = (result.get("text") or "").strip()
        logger.debug("Transcription succeeded, length=%s chars", len(text))
        return text or None
    except Exception as exc:  # pragma: no cover - runtime failure path
        logger.error("Whisper transcription failed: %s", exc)
        return None


def summarize_text(transcript: str) -> Optional[str]:
    """
    Use the same LLM stack as mapping (Ollama) to summarize a transcript.
    """
    if not transcript:
        return None

    payload = {
        "model": llm_fallback.MODEL,
        "prompt": (
            "You are a meeting assistant. Summarize the following call transcript "
            "in 5 concise bullet points plus a one-line title. Keep it short and factual.\n\n"
            f"TRANSCRIPT:\n{transcript}\n\nSUMMARY:\n"
        ),
        "options": {"num_ctx": 4096, "temperature": 0.2},
        "stream": False,
    }

    try:
        r = requests.post(llm_fallback.OLLAMA_URL, json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()
        content = (data.get("response") or "").strip()
        return content or None
    except Exception as exc:  # pragma: no cover - runtime failure path
        logger.error("LLM summary failed: %s", exc)
        return None


def summarize_audio_upload(django_file) -> str:
    """
    End-to-end helper for the Django view: save the upload, transcribe, summarize.
    Always returns a string (may be a fallback message).
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
        for chunk in django_file.chunks():
            tmp.write(chunk)
        tmp_path = tmp.name

    transcript = transcribe_audio(tmp_path)
    summary = summarize_text(transcript or "")

    try:
        os.remove(tmp_path)
    except Exception:
        pass

    if summary:
        return summary
    if transcript:
        return "Transcript captured, but summarization failed."
    return "Recording received, but transcription unavailable. Please check API keys."


def summarize_from_url(url: str) -> str:
    """
    Download audio from a URL and summarize. Uses the same Whisper/Ollama flow.
    """
    if not url:
        return "Recording URL missing."
    tmp_path = None
    try:
        import requests
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
            tmp_path = tmp.name
            with requests.get(url, stream=True, timeout=30) as resp:
                resp.raise_for_status()
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        tmp.write(chunk)
        transcript = transcribe_audio(tmp_path)
        summary = summarize_text(transcript or "")
        if summary:
            return summary
        if transcript:
            return "Transcript captured from URL, but summarization failed."
        return "Could not transcribe the downloaded audio."
    except Exception as exc:
        logger.error("Failed to download or summarize from URL %s: %s", url, exc)
        return "Recording could not be fetched from URL."
    finally:
        if tmp_path:
            try:
                os.remove(tmp_path)
            except Exception:
                pass
