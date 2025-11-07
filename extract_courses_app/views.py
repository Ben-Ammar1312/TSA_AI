from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from .llama_service import extract_courses
from .serializers import ExtractRequestSerializer

@api_view(["POST"])
def extract_view(request):
    serializer = ExtractRequestSerializer(data=request.data)
    if serializer.is_valid():
        text = serializer.validated_data["text"]
        print("📩 Received text:", text)
        courses = extract_courses(text)
        print("✅ Extracted courses:", courses)
        return Response({"courses": courses}, status=status.HTTP_200_OK)
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
