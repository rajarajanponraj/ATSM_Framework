import cv2
import numpy as np

def temporal_motion_estimation(prev_frame, curr_frame, activity_coefficient=1.0):
    """
    Calculate temporal activity as an absolute difference in intensity of adjacent frames.
    M(x,y,t) = lambda * |I(x,y,t) - I(x,y,t-1)|
    """
    diff = np.abs(curr_frame.astype(np.float32) - prev_frame.astype(np.float32))
    temporal_activity = activity_coefficient * diff
    return np.clip(temporal_activity, 0, 1)

def spatial_feature_extraction(frame):
    """
    Compute gradient magnitudes using Sobel operators to identify 
    texture and edge-rich areas suitable for watermarking.
    G(x,y) = sqrt(Gx^2 + Gy^2)
    """
    img_float = frame.astype(np.float32)
    
    # Compute gradients along the X and Y axis
    gx = cv2.Sobel(img_float, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_float, cv2.CV_32F, 0, 1, ksize=3)
    
    # Gradient magnitude
    magnitude = cv2.magnitude(gx, gy)
    
    # Normalize magnitude to [0, 1] to serve as a spatial mask
    max_val = np.max(magnitude)
    if max_val > 0:
        magnitude = magnitude / max_val
        
    return magnitude
