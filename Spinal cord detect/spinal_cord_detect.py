import cv2

import numpy as np
import matplotlib.pyplot as plt


# Load the JPG Image
image_path = r"D:\Document\University\Scientific research\Taiwan\Deep learning\CLAHE\Original\Patient\Patient (44).jpg"
image = cv2.imread(image_path)

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Use edge detection to identify potential boundaries of the spinal cord
edges_1 = cv2.Canny(gray_image, 100, 190)

# Use morphological operations to clean up the edges and close gaps
kernel = np.ones((5, 5), np.uint8)
edges_2 = cv2.morphologyEx(edges_1, cv2.MORPH_CLOSE, kernel)

# Find contours
contours, hierarchy = cv2.findContours(edges_2, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
largest_contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]

# Create a mask and draw the largest contour
mask = np.zeros(gray_image.shape, np.uint8)
cv2.drawContours(mask, [largest_contour], -1, 255, thickness=-1)

# Refine the mask with morphological operations
mask = cv2.bitwise_not(mask)

# Extract the spinal cord using the mask
extracted_spinal_cord = cv2.bitwise_and(gray_image, mask)

# Thresholding to create a binary image
_, binary_image = cv2.threshold(extracted_spinal_cord, 40, 60, cv2.THRESH_BINARY)

# Find the largest contour in the binary image
contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
largest_contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]

# Create a refined spinal cord mask
spinal_cord_mask = np.zeros_like(mask)
cv2.drawContours(spinal_cord_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

# Refine the spinal cord mask further with morphological closing
kernel = np.ones((3, 3), np.uint8)
spinal_cord_mask = cv2.morphologyEx(spinal_cord_mask, cv2.MORPH_CLOSE, kernel)

# Convert the mask to a 3-channel image (BGR) so we can overlay it
spinal_cord_mask_colored = cv2.cvtColor(spinal_cord_mask, cv2.COLOR_GRAY2BGR)

# Create a red color where the mask is white
spinal_cord_mask_colored[spinal_cord_mask == 255] = [0, 0, 255]

# Overlay the red mask on the original image
overlay_image = cv2.addWeighted(image, 1, spinal_cord_mask_colored, 0.5, 0)

# Show all the result images
titles = ['Original Image', 'Edges 1', 'Edges 2', 'Largest Contour Mask', 'Extracted Spinal Cord', 'Binary Image', 'Spinal Cord Mask', 'Overlay Image']
images = [image, edges_1, edges_2, mask, extracted_spinal_cord, binary_image, spinal_cord_mask, overlay_image]

plt.figure(figsize=(15, 10))
for i in range(len(images)):
    plt.subplot(2, 4, i+1)
    if len(images[i].shape) == 2:  # Grayscale image
        plt.imshow(images[i], cmap='gray')
    else:  # BGR image
        plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()
