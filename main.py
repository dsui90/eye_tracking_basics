import cv2
from time import time

import numpy as np
from helper_functions.load_video_file import load_mp4_file
import random

def is_blinking_frame(binary_frame, threshold=None):
    if threshold is None:
        threshold = binary_frame.shape[1] // 20
    if horizontal_std(binary_frame) > threshold:
        return True
    return False
  
def horizontal_std(bin_img):
    """
    Calculate the horizontal variance of black pixels in a binary image.
    
    Args:
        bin_img (numpy.ndarray): Input binary image.
    
    Returns:
        numpy.ndarray: Horizontal variance of black pixels.
    """
    # Convert to grayscale if the image is not already
    if len(bin_img.shape) == 3:
        gray_image = cv2.cvtColor(bin_img, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = bin_img

    # get positions of black pixels
    black_pixels = np.where(gray_image == 0)
    # calculate the horizontal variance of black pixels
    return np.std(black_pixels[1])


def my_thresh_fn_python(img, threshold):
    """
    Custom thresholding function to binarize an image.
    
    Args:
        img (numpy.ndarray): Input grayscale image.
        threshold (int): Threshold value for binarization.
    
    Returns:
        numpy.ndarray: Binarized image.
    """
    # Create a binary mask based on the threshold
    binary_mask = img > threshold
    binary_mask = binary_mask.astype(np.uint8) * 255
    return binary_mask


def get_darkest_region(image, region_size=10):
    """
    Get the region with the darkest area in the image.
    
    Args:
        image (numpy.ndarray): Input grayscale image.
        region_size (int): Size of the region to consider for darkness.
    
    Returns:
        tuple: Coordinates of the darkest region (x, y).
    """
    # Convert to grayscale if the image is not already
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    # apply mean filter to the image
    kernel = np.ones((region_size, region_size), np.float32) / (region_size * region_size)
    mean_image = cv2.filter2D(gray_image, -1, kernel)
    
    # Find the coordinates of the darkest region
    min_val, _, min_loc, _ = cv2.minMaxLoc(mean_image)
    
    # return the coordinates of the darkest region
    return min_loc, min_val
    
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
    
    divisor = 2
    frame = cv2.resize(frame, (frame.shape[1] // divisor, frame.shape[0] // divisor))
    kernel_size = frame.shape[0] // 16 # 42 #// 16
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    region_size = frame.shape[0] // 42 *4
    mean_filter = np.ones((region_size, region_size), np.float32) / (region_size * region_size)
    # Loop through the video frames
    show_all = True
    show_ellipse = True
    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("End of video or failed to read frame.")
            break
        # mod 1: resize
        
        frame = cv2.resize(frame, (frame.shape[1] // divisor, frame.shape[0] // divisor))
        
        # Process the frame (convert to grayscale and rotate by 180 degrees)
        processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        processed_frame = cv2.rotate(processed_frame, cv2.ROTATE_180)
        
        
        mean_image = cv2.filter2D(processed_frame, -1, mean_filter)
        
        # Find the coordinates of the darkest region
        min_val, _, min_loc, _ = cv2.minMaxLoc(mean_image)
        #min_loc, min_val = get_darkest_region(processed_frame, region_size=region_size)
       
        #binary_frame_0 = cv2.threshold(processed_frame, 70, 255, cv2.THRESH_BINARY)[1]
        #binary_frame_0 = my_thresh_fn_python(processed_frame, 1.5*min_val)
        binary_frame_0 = cv2.threshold(processed_frame, 1.5*min_val, 255, cv2.THRESH_BINARY)[1]
        # use closing to fill small holes
        binary_frame_1 = cv2.morphologyEx(binary_frame_0, cv2.MORPH_CLOSE, kernel)
        # use opening to remove small noise
        binary_frame_2 = cv2.morphologyEx(binary_frame_1, cv2.MORPH_OPEN, kernel)
        
        is_blink = is_blinking_frame(binary_frame_2)
        
    
        # Convert processed_frame to BGR for colored ellipses
        processed_frame_bgr = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2BGR)
        mean_image_bgr = cv2.cvtColor(mean_image, cv2.COLOR_GRAY2BGR)
        contour_image = np.ones_like(processed_frame_bgr) * 255
        contour_image_with_ellipse = np.ones_like(processed_frame_bgr) * 255
        if not is_blink and show_ellipse:
            # Detect contours in the binary image
            contours, _ = cv2.findContours(binary_frame_2, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            detected_ellipses = []
            
            # draw contours on white background
            
            cv2.drawContours(contour_image, contours, -1, (0, 0, 0), 1)
            cv2.drawContours(contour_image_with_ellipse, contours, -1, (0, 0, 0), 1)
            
            
            fitting = 'LS'
            if fitting == 'LS':
                for contour in contours:
                    if len(contour) >= min(20, max(binary_frame_0.shape[0]//20,5)):  # Minimum points required to fit an ellipse
                        mellipse = cv2.fitEllipse(contour)
                        if mellipse[1][0] < binary_frame_0.shape[0]/2 and mellipse[1][1] < binary_frame_0.shape[0]/2:
                            detected_ellipses.append(mellipse)  
            
            elif fitting == 'Hough':
                # use hough transform to detect circles in cv2
                # canny edge detection
                circles = cv2.HoughCircles(binary_frame_2, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                                    param1=5, param2=10, minRadius=0, maxRadius=binary_frame_0.shape[0]//2)
            
                # Convert the output to a list of tuples (x, y, radius)
                if circles is not None:
                    circles = np.uint16(np.around(circles))
                    detected_ellipses = [((circle[0], circle[1]), (circle[2],circle[2]), 0) for circle in circles[0, :]]
                else:
                    detected_ellipses = []
            elif fitting == 'RANSAC_Circle':
                # use ransac to detect circles in cv2
                detected_ellipses = []
            
            # Draw the detected ellipses on the processed frame in random colors
            for ellipse in detected_ellipses:
                # Generate a random color
                color = (0, 255, 0)  # Green color
                try:
                    cv2.ellipse(processed_frame_bgr, ellipse, color, 2)
                    cv2.ellipse(contour_image_with_ellipse, ellipse, color, 1)
                    
                    #print(ellipse)
                    angle_radians = np.deg2rad(ellipse[-1])
                    orthogonal_direction = (-np.cos(angle_radians), -np.sin(angle_radians))
                    scale = 50
                    orthogonal_vector = (int(ellipse[0][0] + scale * orthogonal_direction[0]),
                                int(ellipse[0][1] + scale * orthogonal_direction[1]))
                    cv2.line(processed_frame_bgr, (int(ellipse[0][0]), int(ellipse[0][1])), orthogonal_vector, (255, 0, 0), 2)
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
        side_by_side = cv2.hconcat([processed_frame_bgr, mean_image_bgr, contour_image])

        side_by_side_2 = cv2.hconcat([
            cv2.cvtColor(binary_frame_0, cv2.COLOR_GRAY2BGR), 
            cv2.cvtColor(binary_frame_2, cv2.COLOR_GRAY2BGR), 
            contour_image_with_ellipse])

        #side_by_side_3 = cv2.hconcat([contour_image, contour_image])
        # stack both side by side
        side_by_side = cv2.vconcat([side_by_side, side_by_side_2])
        
        side_by_side = cv2.resize(side_by_side, (int(orig_shape[1] * 3), int(orig_shape[0] * 2)))

        # Update frame count and calculate FPS
        frame_count += 1
        elapsed_time = time() - start_time
        fps = frame_count / elapsed_time

        # Overlay FPS on the image
        if show_ellipse and True:
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

        if show_all:
            cv2.imshow('Processed Frame and Binary Frame', side_by_side)
        else:
            processed_frame_bgr = cv2.resize(processed_frame_bgr, (orig_shape[1], orig_shape[0]))
            cv2.imshow('Processed Frame', processed_frame_bgr)
        # Display the combined frames using OpenCV
        #cv2.imshow('Processed Frame and Binary Frame', side_by_side)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close OpenCV windows
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()