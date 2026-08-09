import os
import argparse
import numpy as np
from data_loader import frame_generator
from preprocessing import preprocess_frame
from feature_extraction import temporal_motion_estimation, spatial_feature_extraction
from watermark import generate_computational_watermark, embed_watermark

def main(dataset_dir):
    print("Starting Adaptive Temporal-Spatial Modulation (ATSM) Framework...")
    
    # 1. Initialize data generator (prevents RAM overload)
    frames = frame_generator(dataset_dir)
    
    prev_processed_frame = None
    
    # Process frames one-by-one
    for idx, frame in enumerate(frames):
        if idx % 50 == 0:
            print(f"Processing frame {idx}...")
            
        # 2. Run Preprocessing
        curr_processed = preprocess_frame(frame)
        
        # 3. Run Feature Extraction
        spatial_mask = spatial_feature_extraction(curr_processed)
        
        if prev_processed_frame is not None:
            # Estimate temporal motion compared to previous frame
            temporal_activity = temporal_motion_estimation(prev_processed_frame, curr_processed)
            
            # 4. Generate and Embed Watermark
            # Adaptive strength: weaker where there is high motion to preserve visual quality
            adaptive_alpha = 0.05 * (1.0 - np.mean(temporal_activity))
            
            # Temporal sequence value S(t) for this frame
            fps_assumption = 1000.0
            modulation_freq = 120.0 # High frequency to alias with rolling shutter
            S_t = np.sin(2 * np.pi * modulation_freq * (idx / fps_assumption))
            
            # Generate W(x,y,t)
            watermark = generate_computational_watermark(spatial_mask, S_t, alpha=adaptive_alpha)
            
            # Embed watermark I_watermarked = I_original + W
            watermarked_frame = embed_watermark(curr_processed, watermark)
            
            # TODO: Simulate Camcorder Capture
            # TODO: Extract Watermark
            
        prev_processed_frame = curr_processed
        
    # TODO: Evaluate Performance
    print("ATSM Execution Completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ATSM Framework")
    parser.add_argument("--dataset", type=str, default="dataset/", help="Path to the dataset sequence directory")
    args = parser.parse_args()
    main(args.dataset)
