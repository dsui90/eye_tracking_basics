import cv2

def load_mp4_file(file_path):
    """
    Load an MP4 video file using OpenCV.
    
    Args:
        file_path (str): Path to the MP4 video file.
    
    Returns:
        cv2.VideoCapture: Video capture object.
    """
    # Create a VideoCapture object
    video_capture = cv2.VideoCapture(file_path)
    
    # Check if the video file was opened successfully
    if not video_capture.isOpened():
        raise ValueError(f"Error opening video file: {file_path}")
    
    return video_capture

def process_video_capture(video_capture):
    """
    Process the video capture frame by frame.
    
    Args:
        video_capture (cv2.VideoCapture): Video capture object.
    
    Returns:
        list: List of frames from the video.
    """
    frames = []
    
    while True:
        # Read a frame from the video capture
        ret, frame = video_capture.read()
        
        # Break the loop if no more frames are available
        if not ret:
            break
        
        # Append the frame to the list
        frames.append(frame)
    
    return frames