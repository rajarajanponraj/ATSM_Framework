import os
import argparse
import numpy as np
from data_loader import frame_generator
from preprocessing import preprocess_frame
from feature_extraction import temporal_motion_estimation, spatial_feature_extraction
from watermark import generate_computational_watermark, embed_watermark
from evaluation import capture_simulation, extract_watermark

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
            adaptive_alpha = 0.05 * (1.0 - np.mean(temporal_activity))
            fps_assumption = 1000.0
            modulation_freq = 120.0
            S_t = np.sin(2 * np.pi * modulation_freq * (idx / fps_assumption))
            watermark = generate_computational_watermark(spatial_mask, S_t, alpha=adaptive_alpha)
            watermarked_frame = embed_watermark(curr_processed, watermark)
            
            # 5. Simulate Camcorder Capture & Extract Watermark
            # Camcorder sees aliased watermark due to rolling shutter
            camcorder_frame, actual_aliased_watermark = capture_simulation(
                curr_processed, spatial_mask, row_readout_time=0.00002, 
                modulation_freq=modulation_freq, alpha=adaptive_alpha
            )
            
            # The cinema owner extracts the watermark from the pirated camcorder recording
            extracted_watermark = extract_watermark(camcorder_frame, curr_processed)
            
            # For demonstration, verify if extraction was successful
            if idx == 50:
                diff = np.mean(np.abs(extracted_watermark - actual_aliased_watermark))
                print(f"   -> Frame 50 Extraction check (Mean Absolute Error): {diff:.6f}")
            
        prev_processed_frame = curr_processed
        
    # TODO: Evaluate Performance (Phase 6)
    print("ATSM Execution Completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ATSM Framework")
    parser.add_argument("--dataset", type=str, default="dataset/", help="Path to the dataset sequence directory")
    args = parser.parse_args()
    main(args.dataset)
