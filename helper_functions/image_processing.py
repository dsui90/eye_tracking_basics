import numpy as np

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