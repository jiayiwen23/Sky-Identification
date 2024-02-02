import cv2
import numpy as np

def identify_sky_pixels(image_path, target_size):
    # Load the image
    img = cv2.imread(image_path)

    # Convert the image to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Median Blur
    img_blurred = cv2.medianBlur(img_gray, 5)

    # Apply Laplacian operator
    lap = cv2.Laplacian(img_blurred, cv2.CV_8U)

    # Adaptive Thresholding
    lap_threshold = cv2.adaptiveThreshold(lap, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize=9, C=3)

    # Dynamic kernel size for morphological operations
    kernel_size = (max(3, int(img.shape[0] / 50)), max(3, int(img.shape[1] / 50)))
    kernel_gradient = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)

    # Further refine the mask using morphological operations (erosion and dilation)
    mask = cv2.morphologyEx(lap_threshold, cv2.MORPH_ERODE, kernel_gradient)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel_gradient)

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

# Example usage
identify_sky_pixels('sample picture/2.jpg', target_size=(500, 300))
