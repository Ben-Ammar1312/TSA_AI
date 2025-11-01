import re, unicodedata

_ROMAN_MAP = [(r'\biv\b','4'), (r'\biii\b','3'), (r'\bii\b','2'), (r'\bi\b','1')]

def normalize_label(s: str) -> str:
    if not s:
        return ''
    s = unicodedata.normalize("NFKD", s).encode("ascii","ignore").decode()
    s = s.lower()
    s = s.replace('-', ' ').replace('_', ' ')
    for pat, rep in _ROMAN_MAP:
        s = re.sub(pat, rep, s)
    s = re.sub(r'[^a-z0-9\s]+', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s