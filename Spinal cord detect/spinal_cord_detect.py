import cv2
import numpy as np
import matplotlib.pyplot as plt
from functions import visualize_results, RoundHPF

# Load the grayscale MRI image
# Dataset\Original\Healthy\Healthy (1).jpg
# Dataset\Original\Patient\Patient (16).jpg
image_path = r"Dataset\Original\Patient\Patient (16).jpg"
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# ---------------------------------------
# Fourier Transform

# Output is a 2D complex array. 1st channel real and 2nd imaginary
dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)

# Shift zero-frequency to the center
dft_shift = np.fft.fftshift(dft)

# ---------------------------------------
# Create a High-Pass Filter Mask
mask = RoundHPF(img, r_out=20)
# ---------------------------------------
# Convert the mask to a 3-channel for convolution (if needed for complex DFT)
mask_smoothed = cv2.GaussianBlur(mask, (61,61), sigmaX=10)

# Expand the smoothed mask to apply it to both real and imaginary parts of the DFT
mask_smoothed = np.repeat(mask_smoothed[:, :, np.newaxis], 2, axis=2)

# Apply the softened mask to the frequency domain
fshift = dft_shift * mask_smoothed

# Magnitude spectrum after applying the high-pass filter (optional)
magnitude_spectrum = 2000 * np.log(cv2.magnitude(fshift[:, :, 0], fshift[:, :, 1]) + 1)

# ---------------------------------------
# Inverse DFT to get the edge-detected image
f_ishift = np.fft.ifftshift(fshift)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

# Normalize the result for better visualization
img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)



# kernel = np.ones((5,5), np.uint8)
# edges_1 = cv2.morphologyEx(img_back, cv2.MORPH_CLOSE, kernel)

# # Step 1: Apply Thresholding to isolate potential regions of the spinal cord
img_back = cv2.convertScaleAbs(img_back)
_,binary_image = cv2.threshold(img_back,100,200,cv2.THRESH_BINARY+cv2.THRESH_OTSU)


edges_2 = cv2.Canny(np.uint8(img_back), 100, 200)

kernel = np.ones((5, 5), np.uint8)
edges_2 = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

# ---------------------------------------
# Visualization

titles = ['Original Image', 'High-Pass Mask (Sharp)', 'Smoothed Mask','Image Back' ,'Binary Image', 'Edge Image (Canny)']
images = [img, mask, mask_smoothed[:, :, 0], img_back ,binary_image,  edges_2]

visualize_results(images, titles, cols=4)



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
