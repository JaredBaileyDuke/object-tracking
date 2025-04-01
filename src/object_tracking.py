"""
OBJECT TRACKING WITH OPENCV

- This script tracks a specified object in a video using OpenCV's object tracking algorithms.
- It initializes a tracker, processes video frames, and saves the output with the tracked object highlighted.
"""

import cv2
import sys
import os

# Input video file name
video_input_file_name = "plane_short.mp4"

# Create output file name by appending '_tracked' before the file extension and using .mp4 extension
base_name, _ = os.path.splitext(video_input_file_name)
video_output_file_name = f"{base_name}_tracked.mp4"

# Initialize video capture
video_cap = cv2.VideoCapture(video_input_file_name)
if not video_cap.isOpened():
    print("Error opening video file")
    sys.exit()

# Get video properties for saving the output video
frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video_cap.get(cv2.CAP_PROP_FPS)

# Try using H264 codec, if available. If not, fallback to MP4V.
fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H264 codec
video_writer = cv2.VideoWriter(video_output_file_name, fourcc, fps, (frame_width, frame_height))
if not video_writer.isOpened():
    print("H264 codec not available, falling back to MP4V codec.")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_output_file_name, fourcc, fps, (frame_width, frame_height))
    
# Read the first frame
ok, frame = video_cap.read()
if not ok:
    print("Failed to read video")
    sys.exit()

def initialize_tracker():
    """
    Initializes an object tracker.
    Returns:
        tracker: Initialized OpenCV tracker.
    """
    # Choose tracker type (using CSRT for accuracy)
    tracker_types = ['BOOSTING', 'CSRT', 'KCF', 'MEDIANFLOW', 'MIL', 'MOSSE', 'TLD']
    tracker_type = tracker_types[1]
    
    if tracker_type == 'BOOSTING':
        tracker = cv2.legacy.TrackerBoosting_create()
    elif tracker_type == 'CSRT':
        tracker = cv2.legacy.TrackerCSRT_create()
    elif tracker_type == 'KCF':
        tracker = cv2.legacy.TrackerKCF_create()
    elif tracker_type == 'MEDIANFLOW':
        tracker = cv2.legacy.TrackerMedianFlow_create()
    elif tracker_type == 'MIL':
        tracker = cv2.legacy.TrackerMIL_create()
    elif tracker_type == 'MOSSE':
        tracker = cv2.legacy.TrackerMOSSE_create()
    elif tracker_type == 'TLD':
        tracker = cv2.legacy.TrackerTLD_create()
    
    return tracker

def initialize_object(frame, tracker, x, y, w, h):
    """
    Initializes the tracker with a predefined bounding box.
    Args:
        frame: First frame of the video.
        tracker: Tracker to initialize.
        x, y, w, h: Coordinates and size of the bounding box.
    Returns:
        bbox: The bounding box used for initialization.
    """
    bbox = (x, y, w, h)
    tracker.init(frame, bbox)
    return bbox

def track_object(video_cap, tracker, bbox, video_writer):
    """
    Processes video frames, tracks the selected object, and writes the results to an output video.
    Args:
        video_cap: Video capture object.
        tracker: Initialized object tracker.
        bbox: Initial bounding box.
        video_writer: VideoWriter object to save the video.
    """
    while True:
        ok, frame = video_cap.read()
        if not ok:
            break 
        
        success, bbox = tracker.update(frame)
        if success:
            x, y, w, h = [int(v) for v in bbox]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        else:
            cv2.putText(frame, "Tracking failure detected", (80, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        
        # Write the frame to the output video file
        video_writer.write(frame)
    
    # Ensure all resources are released properly
    video_cap.release()
    video_writer.release()

def test_functions():
    """
    Tests the object tracking functionality using predefined bounding box coordinates.
    """
    try:
        print("Initializing tracker...")
        tracker = initialize_tracker()
        
        # Define the initial bounding box (x, y, width, height)
        # Adjust these values based on your video and the object you want to track.
        x = 300    # Example x coordinate
        y = 200    # Example y coordinate
        w = 100    # Example width
        h = 100    # Example height
        
        print(f"Initializing object with bounding box: (x={x}, y={y}, w={w}, h={h})")
        bbox = initialize_object(frame, tracker, x, y, w, h)
        
        print("Tracking object and saving video...")
        track_object(video_cap, tracker, bbox, video_writer)
        
        print(f"Tracking complete. Output saved as '{video_output_file_name}'.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Start tracking without any popup windows
test_functions()
