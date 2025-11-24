import csv
from pathlib import Path
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction
from matcher.models import SubjectTarget, SubjectAlias, Categorie, Lang
from matcher.utils import normalize_label

ALLOWED_CATS = {c.value for c in Categorie}
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data"

class Command(BaseCommand):
    help = "Sync catalog to match the CSVs exactly (upsert + purge). FR first, then EN."

    def add_arguments(self, parser):
        parser.add_argument('--targets', default=str(DEFAULT_DATA_DIR / 'targets.csv'))
        parser.add_argument('--aliases', default=str(DEFAULT_DATA_DIR / 'aliases.csv'))
        parser.add_argument('--purge', action='store_true')

    def _read_csv(self, path):
        p = Path(path)
        if not p.exists():
            raise CommandError(f"CSV not found: {p} (use --targets/--aliases to point to your files)")
        with open(p, newline='', encoding='utf-8') as f:
            return list(csv.DictReader(f))

    @transaction.atomic
    def handle(self, *args, **opts):
        rows_t = self._read_csv(opts['targets'])
        if not rows_t:
            raise CommandError("targets.csv empty")

        # Validate and upsert targets
        keep_codes = set()
        for i, r in enumerate(rows_t, start=2):
            code = (r.get('code') or '').strip()
            title_fr = (r.get('title_fr') or '').strip()
            categorie = (r.get('categorie') or '').strip()
            level = (r.get('level') or '').strip()

            if not code or code.startswith('#'):
                continue
            if categorie not in ALLOWED_CATS:
                raise CommandError(f"Line {i}: invalid categorie={categorie}")
            if not title_fr:
                raise CommandError(f"Line {i}: title_fr missing for code={code}")

            level_val = int(level) if level else None
            obj, _ = SubjectTarget.objects.update_or_create(
                code=code,
                defaults=dict(
                    title_fr=title_fr,
                    title_en=None,
                    categorie=categorie,
                    level=level_val,
                    norm_label=normalize_label(title_fr),
                    is_active=True,
                    version=1,
                )
            )
            keep_codes.add(code)

        # Clear aliases for kept targets only, then insert FR first, then EN
        SubjectAlias.objects.filter(target__code__in=keep_codes).delete()
        rows_a = self._read_csv(opts['aliases'])

        # Insert FR then EN deterministically
        for lang in (Lang.FR, Lang.EN):
            for r in rows_a:
                code = (r.get('code') or '').strip()
                if code not in keep_codes or (r.get('language') or '').strip() != lang:
                    continue
                label = (r.get('label') or '').strip()
                if not label:
                    continue
                t = SubjectTarget.objects.get(code=code)
                SubjectAlias.objects.create(
                    target=t,
                    label=label,
                    norm_label=normalize_label(label),
                    language=lang,
                )

        # Guarantee at least one FR alias per target (using title_fr if needed)
        missing_fr = SubjectTarget.objects.filter(code__in=keep_codes).exclude(
            aliases__language=Lang.FR
        )
        for t in missing_fr:
            SubjectAlias.objects.create(
                target=t,
                label=t.title_fr,
                norm_label=normalize_label(t.title_fr),
                language=Lang.FR,
            )

        if opts['purge']:
            deleted, _ = SubjectTarget.objects.exclude(code__in=keep_codes).delete()
            self.stdout.write(f"PURGED targets not in CSV: {deleted}")

        self.stdout.write(self.style.SUCCESS(
            f"Synced {len(keep_codes)} targets, {SubjectAlias.objects.filter(target__code__in=keep_codes).count()} aliases"
        ))
