import matplotlib.pyplot as plt
import numpy as np

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