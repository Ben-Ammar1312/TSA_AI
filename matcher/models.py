# matcher/models.py
from django.db import models
import uuid

class Categorie(models.TextChoices):
    MATH = 'math', 'Mathématiques'
    INFO = 'info', 'Informatique'
    RESEAUX = 'reseaux', 'Réseaux'
    SYSTEMES = 'systemes', 'Systèmes'
    LANGUES = 'langues', 'Langues'
    AUTRE = 'autre', 'Autre'

class Lang(models.TextChoices):
    FR = 'fr', 'Français'
    EN = 'en', 'English'
    UND = 'und', 'Indéterminé'

class SubjectTarget(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    code = models.SlugField(unique=True, max_length=64)          # ex: "reseaux.ip.1"
    title_fr = models.CharField(max_length=160)                  # canonique FR
    title_en = models.CharField(max_length=160, blank=True, null=True)
    categorie = models.CharField(max_length=16, choices=Categorie.choices)
    level = models.PositiveSmallIntegerField(null=True, blank=True)  # 5 ou 6
    norm_label = models.CharField(max_length=180, db_index=True)     # normalisé à partir de title_fr
    is_active = models.BooleanField(default=True)
    version = models.PositiveIntegerField(default=1)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        indexes = [
            models.Index(fields=['norm_label']),
            models.Index(fields=['code']),
            models.Index(fields=['categorie', 'level']),
        ]

    @property
    def display_title(self) -> str:
        return self.title_fr or self.title_en or self.code

class SubjectAlias(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    target = models.ForeignKey(SubjectTarget, related_name='aliases', on_delete=models.CASCADE)
    label = models.CharField(max_length=180)
    norm_label = models.CharField(max_length=180, db_index=True)
    language = models.CharField(max_length=8, choices=Lang.choices, default=Lang.FR)

    class Meta:
        indexes = [
            models.Index(fields=['language', 'norm_label']),
            models.Index(fields=['norm_label']),
        ]
        unique_together = [('target', 'norm_label', 'language')]