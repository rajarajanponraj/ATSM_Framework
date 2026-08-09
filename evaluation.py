import numpy as np

def capture_simulation(original_frame, spatial_mask, row_readout_time=0.00002, modulation_freq=120.0, alpha=0.05):
    """
    Simulate the camcorder capture using the rolling shutter model.
    A rolling shutter camera exposes each row at a slightly different time.
    """
    height, width = original_frame.shape
    
    # Time at which each row is exposed by the rolling shutter camcorder
    rows = np.arange(height)
    t_exposure = rows * row_readout_time
    
    # The camcorder sees the temporal modulation S(t) at time t_exposure
    # Reshape for broadcasting to (height, width)
    S_t_camcorder = np.sin(2 * np.pi * modulation_freq * t_exposure).reshape(height, 1)
    
    # The aliased watermark that appears on the camcorder due to rolling shutter
    W_aliased = alpha * spatial_mask * S_t_camcorder
    
    camcorder_frame = original_frame + W_aliased
    return np.clip(camcorder_frame, 0, 1), W_aliased

def extract_watermark(camcorder_frame, original_frame):
    """
    Isolate the watermark by computing the difference.
    W_extracted = I_camcorder - I_original
    """
    return camcorder_frame - original_frame

def compute_mse(original, processed):
    return np.mean((original - processed) ** 2)

def compute_psnr(original, processed):
    mse = compute_mse(original, processed)
    if mse == 0:
        return float('inf')
    # Assuming pixel values are in [0, 1]
    return 20 * np.log10(1.0 / np.sqrt(mse))

def compute_ber(original_watermark, extracted_watermark):
    # Binarize watermarks for Bit Error Rate calculation
    orig_bin = (original_watermark > 0).astype(int)
    ext_bin = (extracted_watermark > 0).astype(int)
    errors = np.sum(orig_bin != ext_bin)
    return errors / orig_bin.size

def run_evaluation(original_video, processed_video, original_watermark, extracted_watermark):
    """Run all evaluation metrics and compile results."""
    pass
