import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

def extract_center(image, blur_size=21):
    # Get image dimensions
    h, w = image.shape[:2]
    
    # Create a binary mask with a sharp center region
    mask = np.zeros_like(image, dtype=np.uint8)
    
    # Define the center region (cropped part)
    start_x = w // 4  # Start from one-fourth of the width
    end_x = 3 * w // 4  # End at three-fourths of the width
    start_y = 0  # Full height of the image
    end_y = h
    
    # Set the central part of the mask to white (255)
    mask[start_y:end_y, start_x:end_x] = 255
    
    # Blur only the mask to create smooth edges
    mask_blurred = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)
    
    # Create a result image by combining the original image and the mask
    # The mask defines how much of the original image is visible
    mask_normalized = mask_blurred / 255.0  # Normalize the mask to range [0, 1]
    
    # Multiply the original image by the mask to apply it
    result = (image * mask_normalized).astype(np.uint8)

    return result

def crop_images_in_folders(input_folder, output_folder):
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
                crop_img = extract_center(img)
                
                # Create corresponding output path
                relative_path = os.path.relpath(root, input_folder)
                output_dir = os.path.join(output_folder, relative_path)
                os.makedirs(output_dir, exist_ok=True)

                # Save the processed image
                output_file_path = os.path.join(output_dir, file)
                cv2.imwrite(output_file_path, crop_img)
                print(f"Processed and saved: {output_file_path}")

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

def RoundHPF(img, r_out):
    rows, cols = img.shape
    crow, ccol = int(rows / 2), int(cols / 2)

    # Create a binary mask (standard high-pass filter)
    mask = np.ones((rows, cols), np.float32)
    r_out = r_out  # Radius for the circular mask (determines the high-pass filter cut-off)
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - crow) ** 2 + (y - ccol) ** 2 <= r_out ** 2
    mask[mask_area] = 0  # Low frequencies are zeroed out, keeping only high frequencies
    return mask

def RoundBPF(img, r_in, r_out):
    rows, cols = img.shape
    crow, ccol = int(rows / 2), int(cols / 2)
    # Create a binary mask initialized to zeros (for band-pass filtering)
    mask = np.zeros((rows, cols), np.float32)
    # Create the coordinate grid
    x, y = np.ogrid[:rows, :cols]
    # Define the band-pass filter area (frequencies between r_in and r_out)
    mask_area = np.logical_and(
        (x - crow) ** 2 + (y - ccol) ** 2 >= r_in ** 2,  # Frequencies higher than r_in
        (x - crow) ** 2 + (y - ccol) ** 2 <= r_out ** 2  # Frequencies lower than r_out
    )
    # Set the mask to 1 in the band-pass region
    mask[mask_area] = 1
    return mask

def visualize_results(images, titles, cols=4):
    """
    Visualize the images with their corresponding titles.

    Parameters:
    - images: List of images to display.
    - titles: List of titles corresponding to the images.
    - cols: Number of columns to display (default is 4).
    """
    # Calculate rows needed for the grid
    rows = (len(images) + cols - 1) // cols  # Compute rows based on number of images and cols

    # Create the figure
    plt.figure(figsize=(15, 10))

    # Loop through each image and its title
    for i in range(len(images)):
        plt.subplot(rows, cols, i + 1)  # Create subplot at position (row, col)
        plt.imshow(images[i], cmap='gray')  # Display the image in grayscale
        plt.title(titles[i])  # Set the title
        plt.xticks([]), plt.yticks([])  # Hide the axis ticks

    # Show the figure
    # plt.tight_layout()
    plt.show()