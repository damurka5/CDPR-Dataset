#!/usr/bin/env python3
"""
Script to downsample teleop demonstrations from 20 Hz to 10 Hz.
This includes:
1. Downsampling videos from 20 fps to 10 fps (taking every other frame)
2. Downsampling trajectory data from 20 Hz to 10 Hz
3. Recomputing delta actions from the downsampled absolute actions
"""

import os
import numpy as np
import cv2
from pathlib import Path
import shutil
from tqdm import tqdm

# Base directories
INPUT_DIR = Path("/Users/damirnurtdinov/Desktop/My Courses/Диплом/CDPR-Dataset/cdpr_dataset/datasets/cdpr_synth_trimmed/videos")
OUTPUT_DIR = Path("/Users/damirnurtdinov/Desktop/My Courses/Диплом/CDPR-Dataset/cdpr_dataset/datasets/cdpr_synth_10hz/videos")

# Scaling constants (same as in HeadlessCDPRSimulation.save_trajectory_data)
K_XYZ = 0.05   # meters per normalized unit
K_YAW = 0.25   # radians per normalized unit
K_GRIP = 1.0   # grip assumed [-1,1]
SCALES = np.array([K_XYZ, K_XYZ, K_XYZ, K_YAW, K_GRIP], dtype=np.float32)

def read_summary(summary_path):
    """Read summary.txt file and extract relevant information"""
    info = {}
    with open(summary_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("Total frames captured:"):
                info['total_frames'] = int(line.split(":")[1].strip())
            elif line.startswith("Simulation time:"):
                info['sim_time'] = float(line.split(":")[1].strip().split()[0])
            elif line.startswith("language_instruction:"):
                info['language'] = line.split(":", 1)[1].strip()
    return info

def downsample_video(input_path, output_path):
    """Downsample video from 20 fps to 10 fps by taking every other frame"""
    cap = cv2.VideoCapture(str(input_path))
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"  Original video: {frame_count} frames, {fps} fps")
    print(f"  Downsampling to {fps/2:.1f} fps")
    
    # Create video writer with half the fps
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps/2, (width, height))
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Take every other frame (0, 2, 4, ...)
        if frame_idx % 2 == 0:
            out.write(frame)
        
        frame_idx += 1
    
    cap.release()
    out.release()
    
    # Get actual frame count of downsampled video
    downsampled_frame_count = frame_idx // 2
    print(f"  Downsampled video: {downsampled_frame_count} frames")
    
    return downsampled_frame_count

def recompute_delta_actions(actions_abs):
    """
    Recompute delta actions from absolute actions using the same scaling as in
    HeadlessCDPRSimulation.save_trajectory_data
    
    Args:
        actions_abs: (T, 5) array of absolute actions [x, y, z, yaw, grip]
    
    Returns:
        actions_delta_norm: (T-1, 5) array of normalized delta actions
    """
    # Compute deltas between consecutive absolute actions
    # actions_abs[1:] - actions_abs[:-1] gives (T-1, 5)
    deltas = actions_abs[1:] - actions_abs[:-1]
    
    # Normalize using the same scales
    deltas_norm = deltas / SCALES[None, :]
    
    # Clip to [-1, 1]
    deltas_norm = np.clip(deltas_norm, -1.0, 1.0).astype(np.float32)
    
    return deltas_norm

def downsample_trajectory_data(input_path, output_path, orig_summary):
    """
    Downsample trajectory data from 20 Hz to 10 Hz and recompute delta actions
    
    Args:
        input_path: Path to original trajectory_data.npz
        output_path: Path to save downsampled trajectory_data.npz
        orig_summary: Dictionary with original summary info
    
    Returns:
        new_total_frames: Number of frames after downsampling
        new_sim_time: Simulation time after downsampling
    """
    # Load trajectory data
    data = np.load(input_path, allow_pickle=True)
    
    # Create new dictionary for downsampled data
    downsampled_data = {}
    
    # Calculate downsampling factor
    orig_frames = orig_summary['total_frames']
    downsampled_frames = orig_frames // 2
    print(f"  Original trajectory: {orig_frames} frames")
    print(f"  Downsampled trajectory: {downsampled_frames} frames")
    
    # Downsample each array
    for key in data:
        array = data[key]
        
        # Handle different array types
        if key == 'actions_delta_norm':
            # We'll recompute delta actions from downsampled absolute actions
            continue
        elif key == 'task_description':
            # Downsample task_description array
            if len(array) > 0:
                # Take every other element (0, 2, 4, ...)
                downsampled_array = array[::2]
                # For delta actions, we need T-1 task descriptions
                if len(downsampled_array) > downsampled_frames:
                    downsampled_array = downsampled_array[:downsampled_frames]
                downsampled_data[key] = downsampled_array
        elif isinstance(array, np.ndarray) and array.ndim > 0:
            # For trajectory arrays, take every other element
            if len(array) == orig_frames:
                # Full trajectory arrays (observations at each step)
                downsampled_array = array[::2]
                downsampled_data[key] = downsampled_array
            elif len(array) == orig_frames - 1:
                # Delta action arrays (needs special handling)
                # We'll compute new delta actions from downsampled absolute actions
                continue
            else:
                # Other arrays (keep as is)
                downsampled_data[key] = array
        else:
            # Scalar or other data (keep as is)
            downsampled_data[key] = array
    
    # Recompute absolute actions if present
    if 'actions_abs' in downsampled_data:
        actions_abs = downsampled_data['actions_abs']
        
        # Ensure we have the correct number of frames
        if len(actions_abs) > downsampled_frames:
            actions_abs = actions_abs[:downsampled_frames]
        
        # Recompute delta actions from downsampled absolute actions
        if len(actions_abs) > 1:
            actions_delta_norm = recompute_delta_actions(actions_abs)
            downsampled_data['actions_delta_norm'] = actions_delta_norm
            
            # Align other arrays to match delta action length (T-1)
            # This matches the original code's behavior
            delta_len = len(actions_delta_norm)
            for key in list(downsampled_data.keys()):
                array = downsampled_data[key]
                if (isinstance(array, np.ndarray) and array.ndim > 0 and 
                    len(array) == downsampled_frames and key != 'actions_abs'):
                    # Take first delta_len elements to align with delta actions
                    downsampled_data[key] = array[:delta_len]
    elif 'control_signals' in downsampled_data:
        # If we have control_signals instead of actions_abs
        control = downsampled_data['control_signals']
        if len(control) > 0 and control.shape[1] >= 5:
            # Extract absolute actions from first 5 dims of control_signals
            actions_abs = control[:, :5]
            downsampled_data['actions_abs'] = actions_abs
            
            # Recompute delta actions
            if len(actions_abs) > 1:
                actions_delta_norm = recompute_delta_actions(actions_abs)
                downsampled_data['actions_delta_norm'] = actions_delta_norm
                
                # Align arrays
                delta_len = len(actions_delta_norm)
                for key in list(downsampled_data.keys()):
                    array = downsampled_data[key]
                    if (isinstance(array, np.ndarray) and array.ndim > 0 and 
                        len(array) == downsampled_frames and key not in ['actions_abs', 'actions_delta_norm']):
                        downsampled_data[key] = array[:delta_len]
    
    # Ensure task_description is aligned with delta actions
    if 'task_description' in downsampled_data and 'actions_delta_norm' in downsampled_data:
        delta_len = len(downsampled_data['actions_delta_norm'])
        task_desc = downsampled_data['task_description']
        if len(task_desc) > delta_len:
            downsampled_data['task_description'] = task_desc[:delta_len]
        elif len(task_desc) < delta_len and len(task_desc) > 0:
            # Repeat last task description if needed
            last_desc = task_desc[-1]
            additional = [last_desc] * (delta_len - len(task_desc))
            downsampled_data['task_description'] = np.concatenate([task_desc, additional])
    
    # Save downsampled data
    np.savez_compressed(output_path, **downsampled_data)
    
    # Calculate new simulation time (half of original since we're halving the frame rate)
    new_sim_time = orig_summary['sim_time'] * (downsampled_frames / orig_frames)
    
    return downsampled_frames, new_sim_time

def update_summary_for_downsampling(input_summary_path, output_summary_path, 
                                   new_total_frames, new_sim_time, orig_fps, new_fps):
    """Update summary.txt with downsampled frame count and simulation time"""
    with open(input_summary_path, 'r') as f:
        lines = f.readlines()
    
    with open(output_summary_path, 'w') as f:
        for line in lines:
            if line.startswith("Total frames captured:"):
                f.write(f"Total frames captured: {new_total_frames}\n")
            elif line.startswith("Total simulation steps:"):
                f.write(f"Total simulation steps: {new_total_frames}\n")
            elif line.startswith("Simulation time:"):
                f.write(f"Simulation time: {new_sim_time:.2f} seconds\n")
            elif "fps" in line.lower():
                # Update any FPS information if present
                f.write(f"Note: Downsampled from {orig_fps} Hz to {new_fps} Hz\n")
            else:
                f.write(line)

def process_demonstration(demo_dir, orig_fps=20, target_fps=10):
    """Process a single demonstration directory for downsampling"""
    print(f"\nProcessing: {demo_dir.name}")
    
    # Create output directory
    output_dir = OUTPUT_DIR / demo_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read summary
    summary_path = demo_dir / "summary.txt"
    if not summary_path.exists():
        print(f"  ERROR: summary.txt not found in {demo_dir}")
        return False
    
    summary_info = read_summary(summary_path)
    
    try:
        # Process overview video
        overview_path = demo_dir / "overview_video.mp4"
        if overview_path.exists():
            print(f"  Downsampling overview video...")
            downsampled_frame_count = downsample_video(
                overview_path,
                output_dir / "overview_video.mp4"
            )
        else:
            print(f"  WARNING: overview_video.mp4 not found")
            
        # Process ee camera video
        ee_camera_path = demo_dir / "ee_camera_video.mp4"
        if ee_camera_path.exists():
            print(f"  Downsampling ee camera video...")
            downsample_video(
                ee_camera_path,
                output_dir / "ee_camera_video.mp4"
            )
        else:
            print(f"  WARNING: ee_camera_video.mp4 not found")
        
        # Process trajectory data
        traj_path = demo_dir / "trajectory_data.npz"
        if traj_path.exists():
            print(f"  Downsampling trajectory data...")
            new_total_frames, new_sim_time = downsample_trajectory_data(
                traj_path,
                output_dir / "trajectory_data.npz",
                summary_info
            )
            
            # Update summary
            print(f"  Updating summary...")
            update_summary_for_downsampling(
                summary_path,
                output_dir / "summary.txt",
                new_total_frames,
                new_sim_time,
                orig_fps,
                target_fps
            )
        else:
            print(f"  ERROR: trajectory_data.npz not found")
            return False
        
        # Copy any other files
        for file in demo_dir.iterdir():
            if file.name not in ["overview_video.mp4", "ee_camera_video.mp4", 
                               "trajectory_data.npz", "summary.txt"]:
                shutil.copy2(file, output_dir / file.name)
        
        print(f"  ✓ Successfully downsampled from {orig_fps}Hz to {target_fps}Hz")
        return True
        
    except Exception as e:
        print(f"  ERROR processing {demo_dir.name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def verify_downsampling(demo_dir):
    """Verify that downsampling was done correctly"""
    print(f"\nVerifying: {demo_dir.name}")
    
    # Load original and downsampled data
    orig_path = demo_dir / "trajectory_data.npz"
    down_path = OUTPUT_DIR / demo_dir.name / "trajectory_data.npz"
    
    if not orig_path.exists() or not down_path.exists():
        print(f"  ERROR: Missing files")
        return
    
    orig_data = np.load(orig_path, allow_pickle=True)
    down_data = np.load(down_path, allow_pickle=True)
    
    print(f"  Original frames: {len(orig_data['actions_abs'])}")
    print(f"  Downsampled frames: {len(down_data['actions_abs'])}")
    
    # Check delta actions
    if 'actions_delta_norm' in orig_data and 'actions_delta_norm' in down_data:
        print(f"  Original deltas shape: {orig_data['actions_delta_norm'].shape}")
        print(f"  Downsampled deltas shape: {down_data['actions_delta_norm'].shape}")
        
        # Check if delta actions are properly normalized
        down_deltas = down_data['actions_delta_norm']
        if np.min(down_deltas) >= -1.0 and np.max(down_deltas) <= 1.0:
            print(f"  ✓ Delta actions properly normalized to [-1, 1]")
        else:
            print(f"  ✗ Delta actions out of bounds: min={np.min(down_deltas):.3f}, max={np.max(down_deltas):.3f}")
    
    # Check consistency between absolute and delta actions
    if 'actions_abs' in down_data and 'actions_delta_norm' in down_data:
        actions_abs = down_data['actions_abs']
        actions_delta_norm = down_data['actions_delta_norm']
        
        # Recompute deltas from absolute actions
        recomputed_deltas = actions_abs[1:] - actions_abs[:-1]
        recomputed_deltas_norm = recomputed_deltas / SCALES[None, :]
        recomputed_deltas_norm = np.clip(recomputed_deltas_norm, -1.0, 1.0)
        
        # Compare with saved deltas
        if np.allclose(recomputed_deltas_norm, actions_delta_norm, atol=1e-5):
            print(f"  ✓ Delta actions correctly recomputed from absolute actions")
        else:
            print(f"  ✗ Delta actions don't match recomputed values")
            max_diff = np.max(np.abs(recomputed_deltas_norm - actions_delta_norm))
            print(f"    Maximum difference: {max_diff:.6f}")

def main():
    # Create output dataset directory
    OUTPUT_DIR.parent.mkdir(parents=True, exist_ok=True)
    
    # Get all demonstration directories
    demo_dirs = [d for d in INPUT_DIR.iterdir() if d.is_dir() and "HUMAN_CONTROL" in d.name]
    
    print(f"Found {len(demo_dirs)} demonstrations to downsample")
    print(f"Input dataset (20 Hz): {INPUT_DIR}")
    print(f"Output dataset (10 Hz): {OUTPUT_DIR}")
    
    # Process each demonstration
    success_count = 0
    for demo_dir in tqdm(demo_dirs, desc="Downsampling demonstrations"):
        if process_demonstration(demo_dir, orig_fps=20, target_fps=10):
            success_count += 1
    
    print(f"\nDownsampling complete!")
    print(f"Successfully processed: {success_count}/{len(demo_dirs)} demonstrations")
    print(f"Downsampled dataset saved to: {OUTPUT_DIR}")
    
    # Verify a few demonstrations
    print(f"\n=== Verification ===")
    for demo_dir in demo_dirs[:3]:  # Verify first 3
        verify_downsampling(demo_dir)

if __name__ == "__main__":
    main()