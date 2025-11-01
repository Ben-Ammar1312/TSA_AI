from django.urls import path
from .views import extract_view

urlpatterns = [
    path("extract/", extract_view, name="courses"),
]
