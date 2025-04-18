import cv2
from time import time

import numpy as np
from helper_functions.load_video_file import load_mp4_file
from helper_functions.tmp import binarize_image, otsu_threshold
import random


def main():
    """
    Main entry point of the application.
    """
    print("Welcome to the Eye Tracking Basics application!")
    
    # Load the video file
    video_capture = load_mp4_file('data/eye0.mp4')

    # Initialize variables for FPS calculation
    frame_count = 0
    start_time = time()

    # mod2 init outside the loop
    frame = video_capture.read()[1]
    orig_shape = frame.shape
    kernel_size = frame.shape[0] // 16
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # Loop through the video frames
    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("End of video or failed to read frame.")
            break
        # mod 1: resize
        
        frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
        
        # Process the frame (convert to grayscale and rotate by 180 degrees)
        processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        processed_frame = cv2.rotate(processed_frame, cv2.ROTATE_180)
        binary_frame = cv2.threshold(processed_frame, 70, 255, cv2.THRESH_BINARY)[1]
        # use closing to fill small holes
        #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        binary_frame = cv2.morphologyEx(binary_frame, cv2.MORPH_CLOSE, kernel)
        # use opening to remove small noise
        #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        binary_frame = cv2.morphologyEx(binary_frame, cv2.MORPH_OPEN, kernel)
        
        ### get the sklera region
        binary_frame_2 = cv2.threshold(processed_frame, 120, 255, cv2.THRESH_BINARY)[1]
        # use closing to fill small holes
        kernel_2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary_frame_2 = cv2.morphologyEx(binary_frame_2, cv2.MORPH_CLOSE, kernel_2)
        # use opening to remove small noise
        kernel_2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary_frame_2 = cv2.morphologyEx(binary_frame_2, cv2.MORPH_OPEN, kernel_2)
        
        # Detect contours in the binary image
        contours, _ = cv2.findContours(binary_frame, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        detected_ellipses = []
        
        fitting = 'LS'
        if fitting == 'LS':
            for contour in contours:
                if len(contour) >= 20:  # Minimum points required to fit an ellipse
                    ellipse = cv2.fitEllipse(contour)
                    if ellipse[1][0] < binary_frame.shape[0]/2 and ellipse[1][1] < binary_frame.shape[0]/2:
                        detected_ellipses.append(ellipse)  
        
        elif fitting == 'Hough':
            # use hough transform to detect circles in cv2
            circles = cv2.HoughCircles(binary_frame, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                                param1=25, param2=10, minRadius=0, maxRadius=0)
        
            # Convert the output to a list of tuples (x, y, radius)
            if circles is not None:
                circles = np.uint16(np.around(circles))
                detected_ellipses = [((circle[0], circle[1]), (circle[2],circle[2]), 0) for circle in circles[0, :]]
            else:
                detected_ellipses = []
        elif fitting == 'RANSAC_Circle':
            # use ransac to detect circles in cv2
            detected_ellipses = []
            
        
        # Convert processed_frame to BGR for colored ellipses
        processed_frame_bgr = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2BGR)

        # Draw the detected ellipses on the processed frame in random colors
        for ellipse in detected_ellipses:
            # Generate a random color
            color = (0, 255, 0)  # Green color
            try:
                cv2.ellipse(processed_frame_bgr, ellipse, color, 2)
            except Exception as e:
                print(f"Error drawing ellipse: {e}")
                continue


        # Stack the processed frame and binary frame side by side
        side_by_side = cv2.hconcat([processed_frame_bgr, cv2.cvtColor(binary_frame, cv2.COLOR_GRAY2BGR)])

        side_by_side_2 = cv2.hconcat([processed_frame_bgr, cv2.cvtColor(binary_frame_2, cv2.COLOR_GRAY2BGR)])

        # stack both side by side
        side_by_side = cv2.vconcat([side_by_side, side_by_side_2])
        
        side_by_side = cv2.resize(side_by_side, (orig_shape[1] * 2, orig_shape[0] * 2))

        # Update frame count and calculate FPS
        frame_count += 1
        elapsed_time = time() - start_time
        fps = frame_count / elapsed_time

        # Overlay FPS on the image
        cv2.putText(
            side_by_side, 
            f"FPS: {fps:.2f}", 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (255, 255, 255), 
            2, 
            cv2.LINE_AA
        )

        # Display the combined frames using OpenCV
        cv2.imshow('Processed Frame and Binary Frame', side_by_side)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close OpenCV windows
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()