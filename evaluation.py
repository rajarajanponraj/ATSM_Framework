import numpy as np
from skimage.metrics import structural_similarity as ssim

def capture_simulation(original_frame, spatial_mask, row_readout_time=0.00002, modulation_freq=120.0, alpha=0.05):
    height, width = original_frame.shape
    rows = np.arange(height)
    t_exposure = rows * row_readout_time
    S_t_camcorder = np.sin(2 * np.pi * modulation_freq * t_exposure).reshape(height, 1)
    W_aliased = alpha * spatial_mask * S_t_camcorder
    camcorder_frame = original_frame + W_aliased
    return np.clip(camcorder_frame, 0, 1), W_aliased

def extract_watermark(camcorder_frame, original_frame):
    return camcorder_frame - original_frame

def compute_mse(original, processed):
    return np.mean((original - processed) ** 2)

def compute_psnr(original, processed):
    mse = compute_mse(original, processed)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(1.0 / np.sqrt(mse))

def compute_ssim(original, processed):
    return ssim(original, processed, data_range=1.0)

def compute_ber(original_watermark, extracted_watermark):
    orig_bin = (original_watermark > 0).astype(int)
    ext_bin = (extracted_watermark > 0).astype(int)
    errors = np.sum(orig_bin != ext_bin)
    return errors / orig_bin.size

def compute_accuracy(ber):
    return 1.0 - ber

def compute_event_sync_error(temporal_activity):
    return np.mean(temporal_activity) * 0.01

def run_evaluation(original_frame, watermarked_frame, original_watermark, extracted_watermark, temporal_activity):
    metrics = {}
    metrics['MSE'] = compute_mse(original_frame, watermarked_frame)
    metrics['PSNR'] = compute_psnr(original_frame, watermarked_frame)
    metrics['SSIM'] = compute_ssim(original_frame, watermarked_frame)
    metrics['BER'] = compute_ber(original_watermark, extracted_watermark)
    metrics['Accuracy'] = compute_accuracy(metrics['BER'])
    metrics['Event_Sync_Error'] = compute_event_sync_error(temporal_activity)
    return metrics
