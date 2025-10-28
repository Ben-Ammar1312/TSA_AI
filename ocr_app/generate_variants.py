import cv2
import numpy as np
from PIL import Image
import os

def add_gaussian_noise(image, mean=0, sigma=25):
    gauss = np.random.normal(mean, sigma, image.shape).astype('uint8')
    noisy = cv2.add(image, gauss)
    return noisy

def add_salt_pepper_noise(image, salt_prob=0.02, pepper_prob=0.02):
    noisy = np.copy(image)
    total_pixels = image.size // 3
    num_salt = int(total_pixels * salt_prob)
    num_pepper = int(total_pixels * pepper_prob)

    # Salt
    coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape[:2]]
    noisy[coords[0], coords[1]] = 255

    # Pepper
    coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape[:2]]
    noisy[coords[0], coords[1]] = 0

    return noisy

def adjust_brightness(image, factor):
    return cv2.convertScaleAbs(image, alpha=factor, beta=0)

def adjust_contrast(image, alpha):
    return cv2.convertScaleAbs(image, alpha=alpha, beta=0)

def apply_blur(image, kernel=(5,5)):
    return cv2.GaussianBlur(image, kernel, 0)

def generate_variants(image_path, output_folder="variants"):
    os.makedirs(output_folder, exist_ok=True)
    img = cv2.imread(image_path)

    variants = {
        "gaussian_noise": add_gaussian_noise(img),
        "salt_pepper": add_salt_pepper_noise(img),
        "dark": adjust_brightness(img, 0.5),
        "bright": adjust_brightness(img, 1.5),
        "low_contrast": adjust_contrast(img, 0.5),
        "blurred": apply_blur(img)
    }

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    for name, variant in variants.items():
        out_path = os.path.join(output_folder, f"{base_name}_{name}.jpg")
        cv2.imwrite(out_path, variant)
        print(f"Saved: {out_path}")

file_path = os.path.join(os.path.dirname(__file__), "page_0.jpg")

if __name__ == "__main__":
    # Example usage
    generate_variants(file_path)
