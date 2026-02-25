#!/usr/bin/env python3
"""
Script to trim teleop demonstrations by removing:
1. First 5-8 seconds (5s for videos <40s, 8s for videos >50s)
2. Last 12% of the demonstration
"""

import os
import numpy as np
import cv2
from pathlib import Path
import shutil
import json
from tqdm import tqdm

# Base directories
ORIGINAL_DIR = Path("/Users/damirnurtdinov/Desktop/My Courses/Диплом/CDPR-Dataset/cdpr_dataset/datasets/cdpr_synth/videos/HUMAN_CONTROL")
TRIMMED_DIR = Path("/Users/damirnurtdinov/Desktop/My Courses/Диплом/CDPR-Dataset/cdpr_dataset/datasets/cdpr_synth_trimmed/videos")

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

def calculate_trim_indices(video_path, summary_info):
    """
    Calculate trim indices for both video and trajectory data
    Returns: (video_start_frame, video_end_frame, traj_start_idx, traj_end_idx)
    """
    # Get video duration from video file
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = frame_count / fps
    cap.release()
    
    print(f"  Video: {frame_count} frames, {fps} fps, {video_duration:.2f}s")
    print(f"  Trajectory: {summary_info['total_frames']} frames, {summary_info['sim_time']:.2f}s sim time")
    
    # Calculate conversion ratio between video time and trajectory time
    time_ratio = summary_info['sim_time'] / video_duration
    print(f"  Time ratio (traj/video): {time_ratio:.3f}")
    
    # Determine trim start time based on video duration
    if video_duration < 40:
        trim_start_video_sec = 5.0
    elif video_duration > 50:
        trim_start_video_sec = 8.0
    else:
        trim_start_video_sec = 6.0  # middle ground for 40-50s videos
    
    # Calculate trim end (12% from the end)
    trim_end_video_sec = video_duration * 0.12
    if video_duration > 60:
        trim_end_video_sec = 3
    
    # Convert video times to frames
    video_start_frame = int(trim_start_video_sec * fps)
    video_end_frame = int(frame_count - trim_end_video_sec * fps)
    
    # Convert video times to trajectory indices using the ratio
    # The trajectory indices correspond to simulation steps, not video frames
    traj_start_idx = int(trim_start_video_sec * time_ratio * (summary_info['total_frames'] / summary_info['sim_time']))
    traj_end_idx = int((video_duration - trim_end_video_sec) * time_ratio * (summary_info['total_frames'] / summary_info['sim_time']))
    
    # Ensure we don't go out of bounds
    video_start_frame = max(0, min(video_start_frame, frame_count - 1))
    video_end_frame = max(video_start_frame + 1, min(video_end_frame, frame_count))
    traj_start_idx = max(0, min(traj_start_idx, summary_info['total_frames'] - 1))
    traj_end_idx = max(traj_start_idx + 1, min(traj_end_idx, summary_info['total_frames']))
    
    print(f"  Trimming video: frames {video_start_frame} to {video_end_frame} (of {frame_count})")
    print(f"  Trimming trajectory: indices {traj_start_idx} to {traj_end_idx} (of {summary_info['total_frames']})")
    
    return video_start_frame, video_end_frame, traj_start_idx, traj_end_idx, fps

def trim_video(input_path, output_path, start_frame, end_frame, fps):
    """Trim video file and save to output path"""
    cap = cv2.VideoCapture(str(input_path))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    # Skip to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    for frame_num in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
    
    cap.release()
    out.release()

def trim_trajectory_data(input_path, output_path, start_idx, end_idx, orig_summary):
    """Trim trajectory data and update summary"""
    # Load trajectory data
    data = np.load(input_path, allow_pickle=True)
    
    # Create new dictionary for trimmed data
    trimmed_data = {}
    
    for key in data:
        array = data[key]
        # Only trim arrays that have the same length as total_frames
        if len(array) == orig_summary['total_frames']:
            trimmed_data[key] = array[start_idx:end_idx]
        else:
            trimmed_data[key] = array
    
    # Save trimmed data
    np.savez_compressed(output_path, **trimmed_data)
    
    # Update summary information
    new_total_frames = end_idx - start_idx
    new_sim_time = orig_summary['sim_time'] * (new_total_frames / orig_summary['total_frames'])
    
    return new_total_frames, new_sim_time

def update_summary(input_summary_path, output_summary_path, new_total_frames, new_sim_time):
    """Update summary.txt with new frame count and simulation time"""
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
            else:
                f.write(line)

def process_demonstration(demo_dir):
    """Process a single demonstration directory"""
    print(f"\nProcessing: {demo_dir.name}")
    
    # Create output directory
    output_dir = TRIMMED_DIR / demo_dir.name
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
            # Calculate trim indices based on overview video
            v_start, v_end, t_start, t_end, fps = calculate_trim_indices(
                overview_path, summary_info
            )
            
            # Trim overview video
            print(f"  Trimming overview video...")
            trim_video(
                overview_path,
                output_dir / "overview_video.mp4",
                v_start, v_end, fps
            )
        else:
            print(f"  WARNING: overview_video.mp4 not found")
            
        # Process ee camera video (use same trim indices)
        ee_camera_path = demo_dir / "ee_camera_video.mp4"
        if ee_camera_path.exists():
            print(f"  Trimming ee camera video...")
            trim_video(
                ee_camera_path,
                output_dir / "ee_camera_video.mp4",
                v_start, v_end, fps
            )
        else:
            print(f"  WARNING: ee_camera_video.mp4 not found")
        
        # Process trajectory data
        traj_path = demo_dir / "trajectory_data.npz"
        if traj_path.exists():
            print(f"  Trimming trajectory data...")
            new_total_frames, new_sim_time = trim_trajectory_data(
                traj_path,
                output_dir / "trajectory_data.npz",
                t_start, t_end,
                summary_info
            )
            
            # Update summary
            print(f"  Updating summary...")
            update_summary(
                summary_path,
                output_dir / "summary.txt",
                new_total_frames,
                new_sim_time
            )
        else:
            print(f"  ERROR: trajectory_data.npz not found")
            return False
        
        # Copy any other files
        for file in demo_dir.iterdir():
            if file.name not in ["overview_video.mp4", "ee_camera_video.mp4", 
                               "trajectory_data.npz", "summary.txt"]:
                shutil.copy2(file, output_dir / file.name)
        
        print(f"  ✓ Successfully processed")
        return True
        
    except Exception as e:
        print(f"  ERROR processing {demo_dir.name}: {e}")
        return False

def main():
    # Create trimmed dataset directory
    TRIMMED_DIR.parent.mkdir(parents=True, exist_ok=True)
    
    # Get all demonstration directories
    demo_dirs = [d for d in ORIGINAL_DIR.iterdir() if d.is_dir() and "HUMAN_CONTROL" in d.name]
    
    print(f"Found {len(demo_dirs)} demonstrations to process")
    print(f"Original dataset: {ORIGINAL_DIR}")
    print(f"Trimmed dataset: {TRIMMED_DIR}")
    
    # Process each demonstration
    success_count = 0
    for demo_dir in tqdm(demo_dirs, desc="Processing demonstrations"):
        if process_demonstration(demo_dir):
            success_count += 1
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {success_count}/{len(demo_dirs)} demonstrations")
    print(f"Trimmed dataset saved to: {TRIMMED_DIR}")

if __name__ == "__main__":
    main()