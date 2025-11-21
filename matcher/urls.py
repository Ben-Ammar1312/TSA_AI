from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import SubjectTargetViewSet, SubjectAliasViewSet

router = DefaultRouter()
router.register(r'targets', SubjectTargetViewSet)
router.register(r'aliases', SubjectAliasViewSet)

urlpatterns = [
    path('', include(router.urls)),
]