import cv2
import numpy as np

def identify_sky_pixels(image_path, target_size):
    # Load the image
    img = cv2.imread(image_path)

    # Convert the image to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Laplacian operator
    lap = cv2.Laplacian(img_gray, cv2.CV_8U)

    # Create a gradient mask based on Laplacian values
    gradient_mask = (lap < 7).astype(np.uint8)

    # Further refine the mask using morphological operations
    kernel_gradient = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    mask = cv2.morphologyEx(gradient_mask, cv2.MORPH_ERODE, kernel_gradient)

    # Bitwise AND the original image with the refined mask
    result = cv2.bitwise_and(img, img, mask=mask)

    # Resize the original image and result
    img_resized = cv2.resize(img, target_size)
    result_resized = cv2.resize(result, target_size)

    # Horizontally stack the resized original image and result
    combined_image = np.hstack((img_resized, result_resized))

    # Display the combined image
    cv2.imshow('Original Image vs. Sky Pixels', combined_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

identify_sky_pixels('sample picture/13.jpg', target_size=(500, 300))
