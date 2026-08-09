import torch
import torch.nn.functional as F

def temporal_motion_estimation(prev_frame, curr_frame, activity_coefficient=1.0):
    diff = torch.abs(curr_frame - prev_frame)
    temporal_activity = activity_coefficient * diff
    return torch.clamp(temporal_activity, 0.0, 1.0)

def spatial_feature_extraction(frame):
    # Ensure [B, C, H, W] for F.conv2d
    if frame.dim() == 2:
        frame_4d = frame.unsqueeze(0).unsqueeze(0)
    else:
        frame_4d = frame
        
    device = frame.device
    
    # Sobel kernels
    sobel_x = torch.tensor([[-1., 0., 1.],
                            [-2., 0., 2.],
                            [-1., 0., 1.]], device=device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1., -2., -1.],
                            [ 0.,  0.,  0.],
                            [ 1.,  2.,  1.]], device=device).view(1, 1, 3, 3)
                            
    # Apply convolutions
    gx = F.conv2d(frame_4d, sobel_x, padding=1)
    gy = F.conv2d(frame_4d, sobel_y, padding=1)
    
    # Gradient magnitude
    magnitude = torch.sqrt(gx**2 + gy**2)
    
    # Squeeze back to [H, W]
    magnitude = magnitude.squeeze(0).squeeze(0)
    
    max_val = torch.max(magnitude)
    if max_val > 0:
        magnitude = magnitude / max_val
        
    return magnitude
