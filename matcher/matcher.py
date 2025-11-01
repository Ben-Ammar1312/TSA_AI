from matcher.models import SubjectTarget, SubjectAlias, Lang
from matcher.utils import normalize_label

def deterministic_lookup(label: str):
    q = normalize_label(label)

    # 1) exact FR on targets
    hit = SubjectTarget.objects.filter(norm_label=q, is_active=True).first()
    if hit: return hit.code, "exact_target_fr"

    # 2) exact FR on aliases
    hit = SubjectAlias.objects.filter(norm_label=q, language=Lang.FR, target__is_active=True) \
        .select_related('target').first()
    if hit: return hit.target.code, "exact_alias_fr"

    # 3) exact EN on aliases
    hit = SubjectAlias.objects.filter(norm_label=q, language=Lang.EN, target__is_active=True) \
        .select_related('target').first()
    if hit: return hit.target.code, "exact_alias_en"

    return None, "unmatched"