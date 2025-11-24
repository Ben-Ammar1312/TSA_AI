# ocr_app/views.py

import os
import shutil

from django.conf import settings
import cv2
import numpy as np
import pytesseract
from PIL import Image

from rest_framework.decorators import api_view, parser_classes, permission_classes
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework import status

from .serializers import OCRUploadSerializer
from extract_courses_app.llama_service import extract_courses

# Configure tesseract binary (Homebrew default) unless overridden by env
TESSERACT_CMD = os.getenv("TESSERACT_CMD") or shutil.which("tesseract") or "/opt/homebrew/bin/tesseract"
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

# Point Tesseract to bundled tessdata if user hasn't set one
DEFAULT_TESSDATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tessdata")
if not os.getenv("TESSDATA_PREFIX"):
    os.environ["TESSDATA_PREFIX"] = DEFAULT_TESSDATA

# === Metrics ===
def estimate_noise(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, stddev = cv2.meanStdDev(img)
    return stddev[0][0] ** 2

def laplacian_var(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def salt_pepper_ratio(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    median = cv2.medianBlur(gray, 3)
    diff = cv2.absdiff(gray, median)
    threshold = diff > 30
    ratio = np.sum(threshold) / (gray.shape[0] * gray.shape[1])
    return ratio

def signal_to_noise(img):
    mean = np.mean(img)
    std = np.std(img)
    return 0 if std == 0 else mean / std

# === Preprocessing with quality check ===
def adaptive_preprocess(image_path, save_dir):
    img = cv2.imread(image_path)
    noise_var = estimate_noise(img)
    lap_var = laplacian_var(img)
    sp_ratio = salt_pepper_ratio(img)
    snr = signal_to_noise(img)

    # --- Flexible thresholds ---
    NOISE_VAR_LIMIT = 4900      # Gaussian noise variance above this → very noisy
    SNR_LIMIT = 2               # Signal-to-noise ratio below this → very noisy
    BLUR_LIMIT = 30             # Laplacian variance below this → blurry
    SP_RATIO_LIMIT = 0.1        # Salt & pepper noise above this → noticeable

    # Evaluate conditions
    TOO_NOISY = noise_var > NOISE_VAR_LIMIT or snr < SNR_LIMIT
    TOO_BLURRY = lap_var < BLUR_LIMIT
    TOO_SP = sp_ratio > SP_RATIO_LIMIT

    # --- Decision logic ---
    fail_count = sum([TOO_NOISY, TOO_BLURRY, TOO_SP])
    if fail_count >= 2:
        reason = []
        if TOO_NOISY:
            reason.append("too noisy")
        if TOO_BLURRY:
            reason.append("too blurry")
        if TOO_SP:
            reason.append("too much salt & pepper noise")
        return None, f"Image rejected: {' and '.join(reason)}. Please upload a clearer picture."

    # --- Enhancement ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    processed = clahe.apply(gray)

    if 25 < lap_var < BLUR_LIMIT:
        kernel = np.array(
            [
                [-1, -1, -1],
                [-1,  9, -1],
                [-1, -1, -1],
            ]
        )
        processed = cv2.filter2D(processed, -1, kernel)

    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, os.path.basename(image_path))
    cv2.imwrite(out_path, processed)

    return Image.fromarray(processed), None

# === Combined OCR + Course Extraction API ===
@api_view(["POST"])
@parser_classes([MultiPartParser, FormParser])
@permission_classes([AllowAny])
def ocr_extract_courses_view(request):
    # DRF-level info
    print("DEBUG /ocr/: method =", request.method)
    print("DEBUG /ocr/: content_type =", request.content_type)
    try:
        print("DEBUG /ocr/: data keys =", list(request.data.keys()))
    except Exception as e:
        print("DEBUG /ocr/: error reading data keys:", e)

    # Raw Django request info
    django_req = request._request  # underlying HttpRequest
    print("DEBUG /ocr/: META CONTENT_TYPE =", django_req.META.get("CONTENT_TYPE"))
    print("DEBUG /ocr/: META CONTENT_LENGTH =", django_req.META.get("CONTENT_LENGTH"))
    print("DEBUG /ocr/: POST keys =", list(django_req.POST.keys()))
    print("DEBUG /ocr/: FILES keys =", list(django_req.FILES.keys()))


    serializer = OCRUploadSerializer(data=request.data)
    if not serializer.is_valid():
        print("DEBUG /ocr/: serializer errors =", serializer.errors)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    image_file = serializer.validated_data["image"]
    print("DEBUG /ocr/: received file =", image_file.name, "size =", image_file.size)

    # Save uploaded image
    upload_dir = os.path.join(settings.MEDIA_ROOT, "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    image_path = os.path.join(upload_dir, image_file.name)

    with open(image_path, "wb+") as destination:
        for chunk in image_file.chunks():
            destination.write(chunk)

    # Preprocess image
    processed_dir = os.path.join(settings.MEDIA_ROOT, "processed")
    image, error_message = adaptive_preprocess(image_path, processed_dir)
    if error_message:
        print("DEBUG /ocr/: preprocessing error:", error_message)
        return Response({"error": error_message}, status=status.HTTP_400_BAD_REQUEST)
    if image is None:
        print("DEBUG /ocr/: image is None after preprocessing")
        return Response({"error": "Image processing failed."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    # Perform OCR with Tesseract
    img_np = np.array(image.convert("RGB"))
    # pytesseract returns a full string; split into lines for downstream logic
    result_text = pytesseract.image_to_string(img_np, lang="fra")

    texts = [line for line in result_text.splitlines() if line.strip()]

    print("DEBUG /ocr/: OCR lines_count =", len(texts))

    # Pass OCR result to course extraction (pure function)
    courses = extract_courses(result_text)
    print("DEBUG /ocr/: extracted courses =", courses)

    return Response(
        {
            "filename": image_file.name,
            "ocr_text": result_text,
            "lines_count": len(texts),
            "courses": courses,
        },
        status=status.HTTP_200_OK,
    )
