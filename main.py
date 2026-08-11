import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from data_loader import frame_generator
from preprocessing import preprocess_frame
from feature_extraction import temporal_motion_estimation, spatial_feature_extraction
from watermark import generate_computational_watermark, embed_watermark
from evaluation import capture_simulation, extract_watermark, run_evaluation
from metrics_plotter import generate_graphs

def main(dataset_dir):
    print("Starting Adaptive Temporal-Spatial Modulation (ATSM) Framework...")
    
    frames = frame_generator(dataset_dir)
    prev_processed_frame = None
    all_metrics = []
    
    for idx, frame in enumerate(frames):
        if idx > 0 and idx % 20 == 0:
            print(f"Processed {idx} frames...")
            
        curr_processed = preprocess_frame(frame)
        spatial_mask = spatial_feature_extraction(curr_processed)
        
        if prev_processed_frame is not None:
            temporal_activity = temporal_motion_estimation(prev_processed_frame, curr_processed)
            
            adaptive_alpha = 0.05 * (1.0 - np.mean(temporal_activity))
            fps_assumption = 1000.0
            modulation_freq = 120.0
            S_t = np.sin(2 * np.pi * modulation_freq * (idx / fps_assumption))
            
            watermark = generate_computational_watermark(spatial_mask, S_t, alpha=adaptive_alpha)
            watermarked_frame = embed_watermark(curr_processed, watermark)
            
            camcorder_frame, actual_aliased_watermark = capture_simulation(
                curr_processed, spatial_mask, row_readout_time=0.00002, 
                modulation_freq=modulation_freq, alpha=adaptive_alpha
            )
            
            extracted_watermark = extract_watermark(camcorder_frame, curr_processed)
            
            # Save visual proofs for the research paper (on frame 50)
            if idx == 50:
                print("Saving visual proof images for paper Figures...")
                import cv2
                
                def save_proof(img, filename):
                    img_out = np.clip(img * 255.0, 0, 255).astype(np.uint8)
                    cv2.imwrite(filename, img_out)
                
                # Normalize mask and extraction for clear visibility in the paper
                mask_viz = spatial_mask / (np.max(spatial_mask) + 1e-5)
                extracted_viz = np.abs(extracted_watermark)
                extracted_viz = extracted_viz / (np.max(extracted_viz) + 1e-5)
                
                save_proof(curr_processed, 'proof_1_original.png')
                save_proof(mask_viz, 'proof_2_spatial_mask.png')
                save_proof(watermarked_frame, 'proof_3_watermarked.png')
                save_proof(camcorder_frame, 'proof_4_camcorder.png')
                save_proof(extracted_viz, 'proof_5_extracted.png')
                print("Visual proofs saved successfully!")
            
            # Phase 6: Performance Evaluation per frame
            metrics = run_evaluation(
                original_frame=curr_processed,
                watermarked_frame=watermarked_frame,
                original_watermark=actual_aliased_watermark,
                extracted_watermark=extracted_watermark,
                temporal_activity=temporal_activity
            )
            all_metrics.append(metrics)
            
        prev_processed_frame = curr_processed
        
    print("\n--- Final ATSM Performance Evaluation ---")
    if all_metrics:
        for key in all_metrics[0].keys():
            avg_val = np.mean([m[key] for m in all_metrics])
            print(f"Average {key}: {avg_val:.6f}")
            
        # Generate Graphs for the research paper
        print("Generating performance graphs...")
        generate_graphs(all_metrics)
        
    else:
        print("No frames were processed.")
        
    print("ATSM Execution Completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ATSM Framework")
    parser.add_argument("--dataset", type=str, default="dataset/", help="Path to the dataset sequence directory")
    args = parser.parse_args()
    main(args.dataset)
