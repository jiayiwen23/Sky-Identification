# Sky Identification
Allow user to upload an image and highlight sky pixels.\n
Try out the demo: https://huggingface.co/spaces/Galii/sky
<img width="416" alt="Picture1" src="https://github.com/jiayiwen23/Sky-Identification/assets/133088295/b9911fb1-d650-452f-b8af-12e7a303dc0e">

## Chosen Techniques and Rationale
1. Grayscale Conversion
Implementation: The input image is converted to grayscale using a simple average method.
Rationale: It simplifies the image by removing color information, making subsequent processing steps more effective. Prepare for the Laplacian operator in the next step, as it is usually applied to greyscale images.
3. Median Blur
Implementation: A median blur with a kernel size of 5 is applied to reduce noise and smooth the image.
Rationale: It is employed to reduce noise and prepare the image for edge detection.
4. Laplacian Operator
Implementation: The Laplacian operator is employed to enhance edges and features in the image.
Rationale: The Laplacian filter is a spatial filter used in image processing and computer vision for edge detection and image sharpening. It calculates the second spatial derivative of an image, which highlights regions of rapid intensity change. It enhances edges, aiding in the identification of sky pixels.
5. Adaptive Thresholding
Implementation: Adaptive thresholding using a Gaussian mean adaptive thresholding method is applied to segment the image.
Rationale: It adapts to local variations in the image, improving the accuracy of pixel segmentation.
6. Morphological Operations
Implementation: Morphological operations, specifically erosion, are used to further refine the mask and highlight distinct features.
Rationale: It helps fine-tune the segmentation mask for better results.


