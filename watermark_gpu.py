import torch

def generate_computational_watermark(spatial_mask, temporal_value, alpha=0.05):
    watermark = alpha * spatial_mask * temporal_value
    return watermark

def embed_watermark(frame, watermark):
    watermarked_frame = frame + watermark
    return torch.clamp(watermarked_frame, 0.0, 1.0)
