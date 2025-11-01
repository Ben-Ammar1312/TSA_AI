import os
import cv2
import numpy as np
from PIL import Image
from django.shortcuts import render
from django.conf import settings
from paddleocr import PaddleOCR
from .forms import OCRUploadForm

# Initialize OCR model once
ocr = PaddleOCR(lang='fr', use_textline_orientation=True)

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
    SP_RATIO_LIMIT = 0.1       # Salt & pepper noise above this → noticeable

    # Evaluate conditions
    TOO_NOISY = noise_var > NOISE_VAR_LIMIT or snr < SNR_LIMIT
    TOO_BLURRY = lap_var < BLUR_LIMIT
    TOO_SP = sp_ratio > SP_RATIO_LIMIT

    # --- Decision logic ---
    # Only reject if *two or more* metrics fail significantly
    fail_count = sum([TOO_NOISY, TOO_BLURRY, TOO_SP])
    if fail_count >= 2:
        reason = []
        if TOO_NOISY: reason.append("too noisy")
        if TOO_BLURRY: reason.append("too blurry")
        if TOO_SP: reason.append("too much salt & pepper noise")
        return None, f"Image rejected: {' and '.join(reason)}. Please upload a clearer picture."

    # --- Enhancement ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    processed = clahe.apply(gray)

    # If slightly blurry, add a gentle sharpening filter
    if 25 < lap_var < BLUR_LIMIT:
        kernel = np.array([
            [-1, -1, -1],
            [-1, 9, -1],
            [-1, -1, -1]
        ])
        processed = cv2.filter2D(processed, -1, kernel)

    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, os.path.basename(image_path))
    cv2.imwrite(out_path, processed)

    return Image.fromarray(processed), None

# === Main view ===
def ocr_view(request):
    result_text = None
    error_message = None

    if request.method == "POST":
        form = OCRUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image_file = form.cleaned_data['image']
            upload_dir = os.path.join(settings.MEDIA_ROOT, "uploads")
            os.makedirs(upload_dir, exist_ok=True)
            image_path = os.path.join(upload_dir, image_file.name)

            with open(image_path, 'wb+') as destination:
                for chunk in image_file.chunks():
                    destination.write(chunk)

            processed_dir = os.path.join(settings.MEDIA_ROOT, "processed")
            image, error_message = adaptive_preprocess(image_path, processed_dir)

            if image is not None:
                img_np = np.array(image.convert("RGB"))
                ocr_result = ocr.predict(img_np)
                texts = []
                for res in ocr_result:
                    texts.extend(res.get('rec_texts', []))
                result_text = "\n".join(texts)

    else:
        form = OCRUploadForm()

    return render(request, "upload.html", {
        "form": form,
        "result": result_text,
        "error": error_message,
    })
