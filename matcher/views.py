from rest_framework import viewsets, status
from rest_framework.permissions import IsAuthenticated
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from django.db.models import Q
import logging

from matcher.match_service import match_subjects
from matcher.utils import normalize_label

from .models import SubjectTarget, SubjectAlias, Categorie
from .serializers import SubjectTargetSerializer, SubjectAliasSerializer

logger = logging.getLogger(__name__)

class SubjectTargetViewSet(viewsets.ModelViewSet):
    queryset = SubjectTarget.objects.all()
    serializer_class = SubjectTargetSerializer


class SubjectAliasViewSet(viewsets.ModelViewSet):
    queryset = SubjectAlias.objects.select_related("target").all()
    serializer_class = SubjectAliasSerializer
    permission_classes = [IsAuthenticated]

    # simple filters: ?language=fr&target_code=info.algo.ds&q=tri
    def get_queryset(self):
        qs = super().get_queryset()
        lang = self.request.query_params.get("language")
        tcode = self.request.query_params.get("target_code")
        q = self.request.query_params.get("q")
        if lang:
            qs = qs.filter(language=lang)
        if tcode:
            qs = qs.filter(target__code=tcode)
        if q:
            qs = qs.filter(Q(label__icontains=q) | Q(norm_label__icontains=q))
        return qs

    def perform_create(self, serializer):
        # ensure norm_label is set by serializer.validate()
        serializer.save()

    def perform_update(self, serializer):
        serializer.save()


def _as_int(val):
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def _sync_targets_from_payload(raw_targets):
    """
    Optional hook: accept a target catalog alongside the match request.
    Target dicts can contain code, title/title_fr/name, categorie, level, coef.
    """
    if not isinstance(raw_targets, list) or not raw_targets:
        return 0

    # Track existing codes to deactivate stale targets after sync
    existing_codes = set(SubjectTarget.objects.values_list("code", flat=True))
    seen_codes = set()
    synced = 0
    for item in raw_targets:
        if not isinstance(item, dict):
            continue
        code = str(item.get("code") or "").strip()
        if not code:
            continue

        seen_codes.add(code)
        title_fr = (
            item.get("title_fr")
            or item.get("title")
            or item.get("name")
            or code
        )
        title_en = item.get("title_en") or item.get("titleEn")
        cat = (item.get("categorie") or item.get("category") or "").lower()
        categorie = cat if cat in Categorie.values else Categorie.AUTRE

        defaults = {
            "title_fr": title_fr,
            "title_en": title_en,
            "categorie": categorie,
            "level": _as_int(item.get("level")),
            "norm_label": normalize_label(title_fr) or normalize_label(code),
            # override is_active to true for anything present in payload
            "is_active": True,
            "version": _as_int(item.get("version")) or 1,
            "coef": _as_int(item.get("coef") or item.get("coefficient")) or 0,
        }
        SubjectTarget.objects.update_or_create(code=code, defaults=defaults)
        synced += 1

    # Deactivate targets that are not present in the incoming catalog
    stale = existing_codes - seen_codes
    if stale:
        SubjectTarget.objects.filter(code__in=stale).update(is_active=False)

    return synced


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def match_view(request):
    """
    Accepts {"subjects": ["math", "algo"], "targets":[...]} and returns the matcher trace
    with target metadata (title, level, coef) included.
    """
    raw_subjects = request.data.get("subjects") if hasattr(request, "data") else None
    raw_targets = request.data.get("targets") if hasattr(request, "data") else None
    try:
        body_len = len(getattr(request, "body", b"") or b"")
    except Exception:
        body_len = -1
    logger.debug(
        "Incoming match request content_type=%s body_len=%s payload=%r",
        request.META.get("CONTENT_TYPE"), body_len, raw_subjects
    )

    if raw_targets:
        synced = _sync_targets_from_payload(raw_targets)
        logger.debug("Synced %s targets from request before matching", synced)

    # Accept both list of strings and list of objects with 'rawName'/'raw_name'
    subjects = []
    if isinstance(raw_subjects, list):
        for item in raw_subjects:
            if isinstance(item, str):
                subjects.append(item)
            elif isinstance(item, dict):
                val = (
                    item.get("rawName")
                    or item.get("raw_name")
                    or item.get("name")
                    or item.get("subject")
                )
                if isinstance(val, str):
                    subjects.append(val)
    elif isinstance(raw_subjects, str):
        subjects.append(raw_subjects)

    if not subjects:
        logger.debug("Match request empty subjects; returning empty response")
        return Response({"matched": [], "coverage_pct": 0.0, "trace": []})

    try:
        result = match_subjects(subjects)
    except Exception as exc:
        logger.exception("Matcher raised error for subjects=%r: %s", subjects, exc)
        return Response(
            {"matched": [], "coverage_pct": 0.0, "trace": [], "error": str(exc)},
            status=status.HTTP_200_OK,
        )

    # Preload target metadata for the matched codes
    targets = {
        t.code: t
        for t in SubjectTarget.objects.filter(code__in=result.get("matched", []))
    }

    # Enrich trace with titles and coefs
    enriched = []
    for item in result.get("trace", []):
        code = item.get("target")
        target = targets.get(code)
        enriched.append(
            {
                "src": item.get("src"),
                "target": code,
                "method": item.get("method"),
                "score": item.get("score"),
                "target_title": target.display_title if target else None,
                "target_level": target.level if target else None,
                "target_coef": target.coef if target else None,
            }
        )

    result["trace"] = enriched
    logger.debug("Match response trace_count=%s coverage_pct=%s matched_codes=%r",
                 len(enriched), result.get("coverage_pct"), result.get("matched"))
    return Response(result)
