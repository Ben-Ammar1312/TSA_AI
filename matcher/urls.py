from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import SubjectTargetViewSet, SubjectAliasViewSet, match_view, summarize_audio_view

router = DefaultRouter()
router.register(r'targets', SubjectTargetViewSet)
router.register(r'aliases', SubjectAliasViewSet)

urlpatterns = [
    path('', include(router.urls)),
    path('match/', match_view, name='match-subjects'),
    path('summarize/audio', summarize_audio_view, name='summarize-audio'),
]
