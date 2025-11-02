from django.core.management.base import BaseCommand
import json, sys
from matcher.match_service import match_subjects

class Command(BaseCommand):
    help = "Test matcher on a JSON list of subject names."

    def add_arguments(self, parser):
        parser.add_argument('--json', help='Path to JSON file with {"subjects":[...]}')

    def handle(self, *args, **opts):
        data = {"subjects": []}
        if opts["json"]:
            with open(opts["json"], encoding="utf-8") as f:
                data = json.load(f)
        else:
            # read subjects from stdin, one per line
            data["subjects"] = [line.strip() for line in sys.stdin if line.strip()]

        out = match_subjects(data["subjects"])
        self.stdout.write(json.dumps(out, ensure_ascii=False, indent=2))