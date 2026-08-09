import cv2
import numpy as np

def intensity_normalization(image):
    """Normalize image pixel intensities to [0,1]."""
    img_float = image.astype(np.float32)
    min_val = np.min(img_float)
    max_val = np.max(img_float)
    if max_val - min_val == 0:
        return np.zeros_like(img_float)
    return (img_float - min_val) / (max_val - min_val)

def noise_reduction(image, sigma=1.0):
    """Apply Gaussian filter to remove sensor noise."""
    # GaussianBlur supports float32 arrays
    return cv2.GaussianBlur(image, (5, 5), sigmaX=sigma, sigmaY=sigma)

def contrast_enhancement(image):
    """Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)."""
    # OpenCV's CLAHE requires 8-bit or 16-bit integer inputs.
    # Convert from float32 [0,1] to uint8 [0, 255]
    img_uint8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img_uint8)
    # Convert back to float32 in range [0, 1]
    return enhanced.astype(np.float32) / 255.0

def image_resizing(image, height=512, width=512):
    """Normalize dimensions before feature extraction."""
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)

def standardize_image(image):
    """Standardize image to zero mean and unit variance."""
    mean_val = np.mean(image)
    std_val = np.std(image)
    if std_val == 0:
        return np.zeros_like(image)
    return (image - mean_val) / std_val

def preprocess_frame(image):
    """Run the full preprocessing pipeline on a single frame."""
    norm_img = intensity_normalization(image)
    denoised_img = noise_reduction(norm_img)
    enhanced_img = contrast_enhancement(denoised_img)
    resized_img = image_resizing(enhanced_img)
    standardized_img = standardize_image(resized_img)
    return standardized_img
