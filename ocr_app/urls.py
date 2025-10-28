from django.urls import path
from .views import ocr_upload

urlpatterns = [
    path('', ocr_upload, name='upload'),
]
