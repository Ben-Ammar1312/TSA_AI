from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from .llama_service import extract_courses
import json


@csrf_exempt
def extract_view(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body.decode("utf-8"))
            text = data.get("text", "")
        except Exception:
            text = request.POST.get("text", "")

        print("📩 Received text:", text)
        courses = extract_courses(text)
        print("✅ Extracted courses:", courses)
        return JsonResponse({"courses": courses})
    return JsonResponse({"error": "POST required"}, status=400)
