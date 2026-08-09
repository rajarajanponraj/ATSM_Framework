import torch
import math

def capture_simulation(original_frame, spatial_mask, row_readout_time=0.00002, modulation_freq=120.0, alpha=0.05):
    height, width = original_frame.shape
    device = original_frame.device
    
    rows = torch.arange(height, device=device, dtype=torch.float32)
    t_exposure = rows * row_readout_time
    
    S_t_camcorder = torch.sin(2 * math.pi * modulation_freq * t_exposure).view(height, 1)
    
    W_aliased = alpha * spatial_mask * S_t_camcorder
    
    camcorder_frame = original_frame + W_aliased
    return torch.clamp(camcorder_frame, 0.0, 1.0), W_aliased

def extract_watermark(camcorder_frame, original_frame):
    return camcorder_frame - original_frame
