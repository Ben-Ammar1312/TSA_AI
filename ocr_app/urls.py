from django.urls import path
from ocr_app.views import ocr_extract_courses_view

urlpatterns = [
    path("", ocr_extract_courses_view, name="ocr_view"),
]