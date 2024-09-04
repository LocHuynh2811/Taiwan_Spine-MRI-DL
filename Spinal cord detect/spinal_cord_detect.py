import cv2

import numpy as np
import matplotlib.pyplot as plt
import pydicom

image_path = r"Original\Healthy\Healthy (5).jpg"
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

#Output is a 2D complex array. 1st channel real and 2nd imaginary
#For fft in opencv input image needs to be converted to float32
dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)

#Rearranges a Fourier transform X by shifting the zero-frequency 
#component to the center of the array.
#Otherwise it starts at the tope left corenr of the image (array)
dft_shift = np.fft.fftshift(dft)

##Magnitude of the function is 20.log(abs(f))
#For values that are 0 we may end up with indeterminate values for log. 
#So we can add 1 to the array to avoid seeing a warning. 
magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))


# Circular HPF mask, center circle is 0, remaining all ones
#Can be used for edge detection because low frequencies at center are blocked
#and only high frequencies are allowed. Edges are high frequency components.
#Amplifies noise.

rows, cols = img.shape
crow, ccol = int(rows / 2), int(cols / 2)

mask = np.zeros((rows, cols, 2), np.uint8)
r_out = 70
r_in = 10
center = [crow, ccol]
x, y = np.ogrid[:rows, :cols]
mask_area = np.logical_and(((x - center[0]) ** 2 + (y - center[1]) ** 2 >= r_in ** 2),
                           ((x - center[0]) ** 2 + (y - center[1]) ** 2 <= r_out ** 2))
mask[mask_area] = 1


# Circular LPF mask, center circle is 1, remaining all zeros
# Only allows low frequency components - smooth regions
#Can smooth out noise but blurs edges.
#
"""
rows, cols = img.shape
crow, ccol = int(rows / 2), int(cols / 2)

mask = np.zeros((rows, cols, 2), np.uint8)
r = 100
center = [crow, ccol]
x, y = np.ogrid[:rows, :cols]
mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
mask[mask_area] = 1

# Band Pass Filter - Concentric circle mask, only the points living in concentric circle are ones
rows, cols = img.shape
crow, ccol = int(rows / 2), int(cols / 2)

mask = np.zeros((rows, cols, 2), np.uint8)
r_out = 80
r_in = 10
center = [crow, ccol]
x, y = np.ogrid[:rows, :cols]
mask_area = np.logical_and(((x - center[0]) ** 2 + (y - center[1]) ** 2 >= r_in ** 2),
                           ((x - center[0]) ** 2 + (y - center[1]) ** 2 <= r_out ** 2))
mask[mask_area] = 1
"""


# apply mask and inverse DFT
fshift = dft_shift * mask

fshift_mask_mag = 2000 * np.log(cv2.magnitude(fshift[:, :, 0], fshift[:, :, 1]))

f_ishift = np.fft.ifftshift(fshift)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])



fig = plt.figure(figsize=(12, 12))
ax1 = fig.add_subplot(2,2,1)
ax1.imshow(img, cmap='gray')
ax1.title.set_text('Input Image')
ax2 = fig.add_subplot(2,2,2)
ax2.imshow(magnitude_spectrum, cmap='gray')
ax2.title.set_text('FFT of image')
ax3 = fig.add_subplot(2,2,3)
ax3.imshow(fshift_mask_mag, cmap='gray')
ax3.title.set_text('FFT + Mask')
ax4 = fig.add_subplot(2,2,4)
ax4.imshow(img_back, cmap='gray')
ax4.title.set_text('After inverse FFT')
plt.show()



# # Apply Fourier Transform
# f_transform = np.fft.fft2(image)
# f_shift = np.fft.fftshift(f_transform)
# magnitude_spectrum = 20 * np.log(cv2.magnitude(f_shift.real, f_shift.imag))


# # Create a mask for the frequency domain (high-pass filter)
# rows, cols, _ = image.shape
# crow, ccol = rows // 2, cols // 2
# mask = np.ones((rows, cols), np.uint8)
# r = 30  # Radius for the filter
# center = [crow, ccol]
# x, y = np.ogrid[:rows, :cols]
# mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
# mask[mask_area] = 0

# # Apply mask and inverse DFT
# mask = np.stack([mask] * 3, axis=-1)  # Expand mask to 3 channels
# f_shift_filtered = f_shift * mask
# f_ishift = np.fft.ifftshift(f_shift_filtered)
# img_back = np.fft.ifft2(f_ishift)
# img_back = np.abs(img_back)

# # Normalize the image to enhance visibility
# img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)

# plt.imshow(img_back, cmap='gray')
# plt.show()

# Optionally, print all metadata



# # Load the JPG Image
# image_path = r"D:\Document\University\Scientific research\Taiwan\Deep learning\CLAHE\Original\Patient\Patient (44).jpg"
# image = cv2.imread(image_path)

# # Convert the image to grayscale
# gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # Use edge detection to identify potential boundaries of the spinal cord
# edges_1 = cv2.Canny(gray_image, 100, 190)

# # Use morphological operations to clean up the edges and close gaps
# kernel = np.ones((5, 5), np.uint8)
# edges_2 = cv2.morphologyEx(edges_1, cv2.MORPH_CLOSE, kernel)

# # Find contours
# contours, hierarchy = cv2.findContours(edges_2, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
# largest_contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]

# # Create a mask and draw the largest contour
# mask = np.zeros(gray_image.shape, np.uint8)
# cv2.drawContours(mask, [largest_contour], -1, 255, thickness=-1)

# # Refine the mask with morphological operations
# mask = cv2.bitwise_not(mask)

# # Extract the spinal cord using the mask
# extracted_spinal_cord = cv2.bitwise_and(gray_image, mask)

# # Thresholding to create a binary image
# _, binary_image = cv2.threshold(extracted_spinal_cord, 40, 60, cv2.THRESH_BINARY)

# # Find the largest contour in the binary image
# contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
# largest_contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]

# # Create a refined spinal cord mask
# spinal_cord_mask = np.zeros_like(mask)
# cv2.drawContours(spinal_cord_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

# # Refine the spinal cord mask further with morphological closing
# kernel = np.ones((3, 3), np.uint8)
# spinal_cord_mask = cv2.morphologyEx(spinal_cord_mask, cv2.MORPH_CLOSE, kernel)

# # Convert the mask to a 3-channel image (BGR) so we can overlay it
# spinal_cord_mask_colored = cv2.cvtColor(spinal_cord_mask, cv2.COLOR_GRAY2BGR)

# # Create a red color where the mask is white
# spinal_cord_mask_colored[spinal_cord_mask == 255] = [0, 0, 255]

# # Overlay the red mask on the original image
# overlay_image = cv2.addWeighted(image, 1, spinal_cord_mask_colored, 0.5, 0)

# # Show all the result images
# titles = ['Original Image', 'Edges 1', 'Edges 2', 'Largest Contour Mask', 'Extracted Spinal Cord', 'Binary Image', 'Spinal Cord Mask', 'Overlay Image']
# images = [image, edges_1, edges_2, mask, extracted_spinal_cord, binary_image, spinal_cord_mask, overlay_image]

# plt.figure(figsize=(15, 10))
# for i in range(len(images)):
#     plt.subplot(2, 4, i+1)
#     if len(images[i].shape) == 2:  # Grayscale image
#         plt.imshow(images[i], cmap='gray')
#     else:  # BGR image
#         plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
#     plt.title(titles[i])
#     plt.xticks([]), plt.yticks([])

# plt.show()
