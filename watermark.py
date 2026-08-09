import numpy as np

def generate_temporal_sequence(num_frames, frequency=30.0, fps=1000.0):
    """
    Generate a high-frequency temporal modulation sequence S(t).
    S(t) is designed to alias with typical rolling shutter row rates.
    """
    t = np.arange(num_frames) / fps
    S_t = np.sin(2 * np.pi * frequency * t)
    return S_t

def generate_computational_watermark(spatial_mask, temporal_value, alpha=0.05):
    """
    Generate the spatio-temporal watermark for a single frame.
    W(x,y,t) = alpha * M(x,y) * S(t)
    """
    watermark = alpha * spatial_mask * temporal_value
    return watermark

def embed_watermark(frame, watermark):
    """
    Integrate the generated watermark into the video sequence.
    I_watermarked = I_original + W
    """
    watermarked_frame = frame.astype(np.float32) + watermark
    # Assuming frames are preprocessed to [0, 1] range
    return np.clip(watermarked_frame, 0, 1)
