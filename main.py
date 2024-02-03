import cv2
import numpy as np
import gradio as gr


def identify_sky_pixels(image):
    """
    Identifies and highlights sky pixels in the input image.

    Parameters:
    - image (numpy.ndarray): The input image as a NumPy array.

    Returns:
    - numpy.ndarray or None: The processed image with highlighted sky pixels.
      If the input image is empty or has incorrect shape, returns None.
    """
    
    # Check if the input image is empty or has incorrect shape
    if image is None or image.shape[0] == 0 or image.shape[1] == 0:
        return None
    
    # Convert the Gradio image to a NumPy array
    img_np = np.array(image)

    # Convert the image to grayscale using a simple average method
    img_gray = np.mean(img_np, axis=-1).astype(np.uint8)

    # Apply Median Blur
    img_blurred = cv2.medianBlur(img_gray, 5)

    # Apply Laplacian operator
    lap = cv2.Laplacian(img_blurred, cv2.CV_8U)
    
    # Adaptive Thresholding
    lap_threshold = cv2.adaptiveThreshold(lap, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize=9, C=3)

    # Further refine the mask using morphological operations with a variable kernel size
    kernel_size = (25, 19)
    kernel_gradient = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    mask = cv2.morphologyEx(lap_threshold, cv2.MORPH_ERODE, kernel_gradient)

    # Bitwise AND the original image with the refined mask
    result = cv2.bitwise_and(img_np, img_np, mask=mask)

    # Return the processed image
    return result


# Create a Gradio interface
iface = gr.Interface(
    fn=identify_sky_pixels,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Image(type="numpy"),
    live=True,
    title="Sky Detection App",
    description="Upload an image to identify sky pixels."
)

iface.launch(share=True)
