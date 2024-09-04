import cv2
import numpy as np
import os
import random

def adjust_brightness(img, brightness):
  """Adjusts the brightness of an image.

  Args:
    img: The input image.
    brightness: The brightness factor. Positive values increase brightness, negative values decrease brightness.

  Returns:
    The image with adjusted brightness.
  """

  # Convert image to float32 to prevent clipping
  img = img.astype(np.float32)

  # Adjust brightness
  adjusted = cv2.add(img, brightness)

  # Clip values to ensure they are within the valid range (0-255)
  adjusted = np.clip(adjusted, 0, 255)

  # Convert back to uint8
  adjusted = adjusted.astype(np.uint8)

  return adjusted

def process_folder(input_folder, output_folder):
  """Processes images in a folder by applying random brightness adjustments.

  Args:
    input_folder: The path to the input folder.
    output_folder: The path to the output folder.
  """

  if not os.path.exists(output_folder):
    os.makedirs(output_folder)

  for filename in os.listdir(input_folder):
    if filename.endswith('.jpg') or filename.endswith('.png'):
      img_path = os.path.join(input_folder, filename)
      img = cv2.imread(img_path)

      # Generate random brightness factor
      brightness_factor = random.randint(10, 50)

      adjusted_img = adjust_brightness(img, brightness_factor)

      output_path = os.path.join(output_folder, filename)
      cv2.imwrite(output_path, adjusted_img)

# Example usage
input_folder = 'Patient'
output_folder = 'Patient_Random_Brightness'
process_folder(input_folder, output_folder)
