import cv2
import numpy as np

def identify_sky_pixels(image_path, target_size=(800, 400)):
    # Load the image
    img = cv2.imread(image_path)

    # Convert the image to the HSV color space
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define lower and upper bounds for sky color in HSV
    lower_sky = np.array([0, 40, 80])
    upper_sky = np.array([180, 255, 255])

    # Threshold the image to get a binary mask of sky pixels
    sky_mask = cv2.inRange(hsv_img, lower_sky, upper_sky)

    # Apply morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    sky_mask = cv2.morphologyEx(sky_mask, cv2.MORPH_OPEN, kernel)
    sky_mask = cv2.morphologyEx(sky_mask, cv2.MORPH_CLOSE, kernel)

    # Bitwise AND the original image with the sky mask
    result = cv2.bitwise_and(img, img, mask=sky_mask)

    # Resize the original image and result
    img_resized = cv2.resize(img, target_size)
    result_resized = cv2.resize(result, target_size)

    # Horizontally stack the resized original image and result
    combined_image = np.hstack((img_resized, result_resized))

    # Display the combined image
    cv2.imshow('Original Image vs. Sky Pixels', combined_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Replace 'your_image_path.jpg' with the actual path to your image
# Set the target_size to the desired size (e.g., (800, 400))
identify_sky_pixels('sample picture/1.jpg', target_size=(500, 300))
