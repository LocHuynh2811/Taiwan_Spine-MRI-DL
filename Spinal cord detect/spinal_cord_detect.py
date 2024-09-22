import cv2
import numpy as np
import matplotlib.pyplot as plt
from functions import visualize_results, RoundHPF, process_images_in_folders, crop_images_in_folders, extract_center
import os

input_folder = r'Dataset\Original'  # Replace with the path to your main folder
output_folder = r'Dataset\Cropped'  # Replace with the path where you want to save the DFT images

# Process the images
crop_images_in_folders(input_folder, output_folder)

# # Apply Thresholding
# img_back = cv2.convertScaleAbs(img_back)
# _, binary_image = cv2.threshold(img_back, 100, 200, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# # Canny Edge Detection
# edges_1 = cv2.Canny(np.uint8(img_back), 100, 200)

# # Morphological operation for cleaning up the edges
# kernel = np.ones((5, 5), np.uint8)
# edges_2 = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

# # ---------------------------------------
# # Visualization

# titles = ['Original Image', 'Magnitude Spectrum Before', 'High-Pass Mask', 'Smoothed Mask',
#           'Magnitude Spectrum After', 'Image Back', 'Threshold Image', 'Edge Image (Canny)']
# images = [img, magnitude_spectrum_before, mask, mask_smoothed[:, :, 0],
#           magnitude_spectrum_after, img_back, binary_image, edges_2]

# visualize_results(images, titles, cols=4)




# Continue with segmentation (uncomment when ready)
# Uncomment below if you want to proceed with spinal cord segmentation

# # Use edge detection to identify potential boundaries of the spinal cord
# edges_1 = cv2.Canny(np.uint8(img_back), 100, 190)

# # Use morphological operations to clean up the edges and close gaps
# kernel = np.ones((5, 5), np.uint8)
# edges_2 = cv2.morphologyEx(edges_1, cv2.MORPH_CLOSE, kernel)

# # Find contours
# contours, _ = cv2.findContours(edges_2, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
# largest_contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]

# # Create a mask and draw the largest contour
# mask = np.zeros(img.shape, np.uint8)
# cv2.drawContours(mask, [largest_contour], -1, 255, thickness=-1)

# # Refine the mask with morphological operations
# mask = cv2.bitwise_not(mask)

# # Extract the spinal cord using the mask
# extracted_spinal_cord = cv2.bitwise_and(img, mask)

# # Thresholding to create a binary image
# _, binary_image = cv2.threshold(extracted_spinal_cord, 40, 60, cv2.THRESH_BINARY)

# # Find the largest contour in the binary image
# contours, _ = cv2.findContours(binary_image, cv2.RETR_LIST, cv2.CHAIN