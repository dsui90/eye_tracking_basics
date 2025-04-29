import cv2
from time import time

import numpy as np
from helper_functions.load_video_file import load_mp4_file
from helper_functions.image_processing import *

DIVISOR = 1
    
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

    frame = video_capture.read()[1]
    orig_shape = frame.shape
    
    # Resize for faster computation
    frame = cv2.resize(frame, (frame.shape[1] // DIVISOR, frame.shape[0] // DIVISOR))
    
    # get image size dependent kernel size
    kernel_size = frame.shape[0] // 16 
    region_size = frame.shape[0] // 42 *4
    
    # mean filter
    mean_filter = np.ones((region_size, region_size), np.float32) / (region_size * region_size)
    
    # structuring element
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # Loop through the video frames
    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("End of video or failed to read frame.")
            break

        frame = cv2.resize(frame, (frame.shape[1] // DIVISOR, frame.shape[0] // DIVISOR))
        
        # Process the frame (convert to grayscale and rotate by 180 degrees)
        processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        processed_frame = cv2.rotate(processed_frame, cv2.ROTATE_180)
        
        # 2d conv with mean filter
        mean_image = cv2.filter2D(processed_frame, -1, mean_filter)
        
        # Find the coordinates of the darkest region
        min_val, _, min_loc, _ = cv2.minMaxLoc(mean_image)
       
        binary_frame_0 = cv2.threshold(processed_frame, 1.5*min_val, 255, cv2.THRESH_BINARY)[1]
        
        # use closing to fill small holes
        binary_frame_closed = cv2.morphologyEx(binary_frame_0, cv2.MORPH_CLOSE, kernel)
        # use opening to remove small noise
        #binary_frame_2 = cv2.morphologyEx(binary_frame_1, cv2.MORPH_OPEN, kernel)
        
        # check if is blinking
        is_blink = is_blinking_frame(binary_frame_closed)
        
    
        # Convert processed_frame to BGR for colored ellipses
        processed_frame_bgr = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2BGR)
        mean_image_bgr = cv2.cvtColor(mean_image, cv2.COLOR_GRAY2BGR)
        
        if not is_blink:
            # Detect contours in the binary image ()
            contours, _ = cv2.findContours(binary_frame_closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            detected_ellipses = []
            
            
            fitting = 'LS'
            if fitting == 'LS':
                for contour in contours:
                    if len(contour) >= min(20, max(binary_frame_0.shape[0]//20,5)):  # Minimum points required to fit an ellipse
                        cur_ellipse = cv2.fitEllipse(contour)
                        if cur_ellipse[1][0] < binary_frame_0.shape[0]/2 and cur_ellipse[1][1] < binary_frame_0.shape[0]/2:
                            detected_ellipses.append(cur_ellipse)  
            
            elif fitting == 'Hough':
                # use hough transform to detect circles in cv2
                # canny edge detection
                circles = cv2.HoughCircles(binary_frame_closed, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                                    param1=5, param2=10, minRadius=0, maxRadius=binary_frame_0.shape[0]//2)
            
                # Convert the output to a list of tuples (x, y, radius)
                if circles is not None:
                    circles = np.uint16(np.around(circles))
                    detected_ellipses = [((circle[0], circle[1]), (circle[2],circle[2]), 0) for circle in circles[0, :]]
                else:
                    detected_ellipses = []

            
            # Draw the detected ellipses on the processed frame 
            for cur_ellipse in detected_ellipses:
                # Generate a random color
                color = (0, 255, 0)  # Green color
                try:
                    cv2.ellipse(processed_frame_bgr, cur_ellipse, color, 2)
                    
                    angle_radians = np.deg2rad(cur_ellipse[-1])
                    orthogonal_direction = (-np.cos(angle_radians), -np.sin(angle_radians))
                    scale = 50
                    orthogonal_vector = (int(cur_ellipse[0][0] + scale * orthogonal_direction[0]),
                                int(cur_ellipse[0][1] + scale * orthogonal_direction[1]))
                    cv2.line(processed_frame_bgr, (int(cur_ellipse[0][0]), int(cur_ellipse[0][1])), orthogonal_vector, (255, 0, 0), 2)
                except Exception as e:
                    print(f"Error drawing ellipse: {e}")
                    continue
                
        
        # Draw a rectangle around the darkest region
        x, y = min_loc[0]-region_size//2, min_loc[1]-region_size//2
        w, h = region_size, region_size
        cv2.rectangle(mean_image_bgr, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # Draw a circle at the center of the darkest region
        center = (x + w // 2, y + h // 2)
        cv2.circle(mean_image_bgr, center, 1, (0, 255, 0), -1)

        # Stack the processed frame and binary frame side by side
        side_by_side = cv2.hconcat([processed_frame_bgr, mean_image_bgr])

        side_by_side_2 = cv2.hconcat([
            cv2.cvtColor(binary_frame_0, cv2.COLOR_GRAY2BGR), 
            cv2.cvtColor(binary_frame_closed, cv2.COLOR_GRAY2BGR)])

        # stack both side by side
        side_by_side = cv2.vconcat([side_by_side, side_by_side_2])
        
        side_by_side = cv2.resize(side_by_side, (int(orig_shape[1] * 2), int(orig_shape[0] * 2)))

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
        
        if is_blink:
            cv2.putText(
                side_by_side, 
                "Blink Detected", 
                (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (0, 0, 255), 
                2, 
                cv2.LINE_AA
            )

        cv2.imshow('Processed Frame and Binary Frame', side_by_side)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close OpenCV windows
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()