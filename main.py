import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from data_loader import frame_generator
from preprocessing import preprocess_frame
from feature_extraction import temporal_motion_estimation, spatial_feature_extraction
from watermark import generate_computational_watermark, embed_watermark
from evaluation import capture_simulation, extract_watermark, run_evaluation

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
        frames_x = np.arange(len(all_metrics))
        plt.figure(figsize=(12, 8))
        
        # Plot 1: Accuracy over time
        plt.subplot(2, 2, 1)
        plt.plot(frames_x, [m['Accuracy'] for m in all_metrics], color='blue')
        plt.title('Extraction Accuracy over Time')
        plt.xlabel('Frame Number')
        plt.ylabel('Accuracy')
        plt.grid(True)
        
        # Plot 2: PSNR over time
        plt.subplot(2, 2, 2)
        psnr_vals = [p if p != float('inf') else 50.0 for p in [m['PSNR'] for m in all_metrics]]
        plt.plot(frames_x, psnr_vals, color='green')
        plt.title('Camcorder PSNR over Time')
        plt.xlabel('Frame Number')
        plt.ylabel('PSNR (dB)')
        plt.grid(True)
        
        # Plot 3: SSIM over time
        plt.subplot(2, 2, 3)
        plt.plot(frames_x, [m['SSIM'] for m in all_metrics], color='orange')
        plt.title('Structural Similarity (SSIM)')
        plt.xlabel('Frame Number')
        plt.ylabel('SSIM')
        plt.grid(True)
        
        # Plot 4: Bit Error Rate (BER)
        plt.subplot(2, 2, 4)
        plt.plot(frames_x, [m['BER'] for m in all_metrics], color='red')
        plt.title('Bit Error Rate (BER)')
        plt.xlabel('Frame Number')
        plt.ylabel('BER')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('atsm_metrics_graph.png', dpi=300)
        print("Graph saved successfully as 'atsm_metrics_graph.png' in your current directory!")
        
    else:
        print("No frames were processed.")
        
    print("ATSM Execution Completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ATSM Framework")
    parser.add_argument("--dataset", type=str, default="dataset/", help="Path to the dataset sequence directory")
    args = parser.parse_args()
    main(args.dataset)
