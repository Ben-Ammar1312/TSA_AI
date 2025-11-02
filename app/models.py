from django.db import models

# core/models.py
from django.db import models
from pgvector.django import VectorField
class Document(models.Model):
    title = models.CharField(max_length=255)
    embedding = VectorField(dimensions=1536)
    created_at = models.DateTimeField(auto_now_add=True)