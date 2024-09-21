import cv2
import numpy as np
import matplotlib.pyplot as plt
from functions import visualize_results, RoundHPF
import os

# DFT and High-Pass Filter function
def apply_dft_filter(image):
    dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    # Create a High-Pass Filter Mask
    mask = RoundHPF(image, r_out=20)

    # Soften the mask using Gaussian smoothing
    mask_smoothed = cv2.GaussianBlur(mask, (61, 61), sigmaX=10)
    mask_smoothed = np.repeat(mask_smoothed[:, :, np.newaxis], 2, axis=2)

    # Apply mask to the frequency domain
    fshift = dft_shift * mask_smoothed

    # Inverse DFT
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    # Normalize the result for visualization
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
    img_back = np.uint8(img_back)
    
    return img_back

# Function to create the DFT folder structure and save processed images
def process_images_in_folders(input_folder, output_folder):
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith(('.jpg', '.png', '.jpeg', '.bmp', '.tiff')):  # Add more formats if needed
                file_path = os.path.join(root, file)
                
                # Load the image
                img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"Failed to load image {file_path}")
                    continue
                
                # Apply the DFT filter
                processed_img = apply_dft_filter(img)
                
                # Create corresponding output path
                relative_path = os.path.relpath(root, input_folder)
                output_dir = os.path.join(output_folder, relative_path)
                os.makedirs(output_dir, exist_ok=True)

                # Save the processed image
                output_file_path = os.path.join(output_dir, file)
                cv2.imwrite(output_file_path, processed_img)
                print(f"Processed and saved: {output_file_path}")

input_folder = r'Dataset\Original'  # Replace with the path to your main folder
output_folder = r'Dataset\DFT'  # Replace with the path where you want to save the DFT images

# Process the images
process_images_in_folders(input_folder, output_folder)

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