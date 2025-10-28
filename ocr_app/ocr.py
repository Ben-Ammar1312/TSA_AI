from paddleocr import PaddleOCR
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import os
import cv2


# === Noise metrics ===
def estimate_noise(img):
    """Estimate overall noise variance."""
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, stddev = cv2.meanStdDev(img)
    return stddev[0][0] ** 2


def laplacian_var(img):
    """Measure sharpness (low = blur, high = noise/edges)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def salt_pepper_ratio(img):
    """Detect salt & pepper noise percentage."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    median = cv2.medianBlur(gray, 3)
    diff = cv2.absdiff(gray, median)
    threshold = diff > 30
    ratio = np.sum(threshold) / (gray.shape[0] * gray.shape[1])
    return ratio


def signal_to_noise(img):
    """Signal-to-noise ratio (higher is better)."""
    mean = np.mean(img)
    std = np.std(img)
    return 0 if std == 0 else mean / std


# === Adaptive Preprocessing ===
def adaptive_preprocess(image_path, save_dir):
    img = cv2.imread(image_path)
    noise_var = estimate_noise(img)
    lap_var = laplacian_var(img)
    sp_ratio = salt_pepper_ratio(img)
    snr = signal_to_noise(img)

    print(f"📊 Noise Report for {os.path.basename(image_path)}:")
    print(f"   • Noise Variance: {noise_var:.2f}")
    print(f"   • Laplacian Var : {lap_var:.2f}")
    print(f"   • S&P Ratio     : {sp_ratio:.4f}")
    print(f"   • SNR           : {snr:.2f}")

    # --- Determine preprocessing strategy ---
    if noise_var > 1000 or snr < 10:
        print("🧹 Detected strong Gaussian noise → Applying Non-local Means denoising")
        processed = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    elif sp_ratio > 0.02:
        print("⚫ Detected salt & pepper noise → Applying median blur")
        processed = cv2.medianBlur(img, 3)
    elif lap_var < 50:
        print("💡 Low detail detected → Possibly blurred, applying sharpening")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        processed = cv2.filter2D(gray, -1, sharpen_kernel)
    else:
        print("✅ Clean image → Light enhancement only")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        processed = clahe.apply(gray)

    # --- Convert and save processed image ---
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, os.path.basename(image_path))
    cv2.imwrite(out_path, processed)
    print(f"🖼️ Processed image saved to: {out_path}\n")

    # Convert to PIL for PaddleOCR
    pil_image = Image.fromarray(processed)
    return pil_image


# === Setup paths ===
base_dir = os.path.dirname(__file__)
variants_folder = os.path.join(base_dir, "variants")
output_folder = os.path.join(base_dir, "ocr_results")
processed_folder = os.path.join(base_dir, "processed")
os.makedirs(output_folder, exist_ok=True)
os.makedirs(processed_folder, exist_ok=True)

# === Initialize PaddleOCR ===
ocr = PaddleOCR(lang='fr', use_textline_orientation=True)

# === Process all variants ===
for filename in os.listdir(variants_folder):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        file_path = os.path.join(variants_folder, filename)
        print(f"\n🔍 Processing: {filename}")

        image = adaptive_preprocess(file_path, processed_folder)
        image_np = np.array(image.convert("RGB"))

        # Perform OCR
        result = ocr.predict(image_np)
        texts = []
        for res in result:
            texts.extend(res.get('rec_texts', []))

        full_text = "\n".join(texts)

        # Save OCR result
        output_file = os.path.join(
            output_folder,
            os.path.splitext(filename)[0] + ".txt"
        )
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(full_text)

        print(f"✅ OCR result saved to: {output_file}")

print("\n🎉 All variant images processed with adaptive noise handling.")
