# matcher/admin.py
from django.contrib import admin
from .models import SubjectTarget, SubjectAlias

@admin.register(SubjectTarget)
class SubjectTargetAdmin(admin.ModelAdmin):
    list_display = ('code','title_fr','categorie','level','is_active','updated_at')
    search_fields = ('code','title_fr','norm_label')
    list_filter = ('categorie','level','is_active')

@admin.register(SubjectAlias)
class SubjectAliasAdmin(admin.ModelAdmin):
    list_display = ('target','label','language')
    search_fields = ('label','norm_label','target__code','target__title_fr')
    list_filter = ('language',)