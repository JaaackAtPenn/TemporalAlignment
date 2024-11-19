import cv2
import numpy as np

def video_to_frames(video_path, target_size=(160, 160), downsample_factor=10):
    """
    Convert a video file to a numpy array of frames with resized dimensions.
    
    Args:
        video_path (str): Path to the video file
        target_size (tuple): Target dimensions for resizing (width, height)
        downsample_factor (int): Take every nth frame
        
    Returns:
        numpy.ndarray: Array of frames with shape (num_frames, height, width, channels)
    """
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    frames = []
    frame_count = 0
    
    # Read frames until video is completed
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Only keep every nth frame
        if frame_count % downsample_factor == 0:
            # Resize frame to target dimensions
            frame = cv2.resize(frame, target_size)
            frames.append(frame)
            
        frame_count += 1
    
    # Release the video capture object
    cap.release()
    
    # Convert list of frames to numpy array
    return np.array(frames)


