import os
import pydicom
import nibabel as nib
import numpy as np
import subprocess
import matplotlib.pyplot as plt


def get_affine_from_dicom(ds):
    """
    Construct an affine matrix from DICOM metadata.

    Parameters:
    - ds: pydicom Dataset object.

    Returns:
    - affine: 4x4 numpy array representing the affine matrix.
    """
    try:
        pixel_spacing = ds.PixelSpacing  # [row_spacing, col_spacing]
        slice_thickness = ds.SliceThickness
        image_orientation = ds.ImageOrientationPatient  # [row_cos_x, row_cos_y, row_cos_z, col_cos_x, col_cos_y, col_cos_z]
        image_position = ds.ImagePositionPatient  # [x, y, z]

        # Direction cosines
        row_cos = np.array(image_orientation[:3])
        col_cos = np.array(image_orientation[3:])
        # Compute the slice direction (cross product)
        slice_cos = np.cross(row_cos, col_cos)

        # Construct the affine matrix
        affine = np.eye(4)
        affine[0, 0:3] = row_cos * pixel_spacing[0]
        affine[1, 0:3] = col_cos * pixel_spacing[1]
        affine[2, 0:3] = slice_cos * slice_thickness
        affine[0:3, 3] = image_position

        return affine
    except AttributeError as e:
        print(f"Missing DICOM metadata for affine construction: {e}")
        return np.eye(4)


def convert_dicom_to_nifti(dicom_file, output_nii):
    """
    Convert a single DICOM file to NIfTI format with accurate affine.

    Parameters:
    - dicom_file: Path to the input DICOM file (.dcm).
    - output_nii: Path to save the output NIfTI file (.nii or .nii.gz).
    """
    try:
        # Read the DICOM file
        ds = pydicom.dcmread(dicom_file)

        # Extract the pixel array data from the DICOM file
        image_data = ds.pixel_array

        # Ensure the pixel data is in the correct shape and type
        image_data = image_data.astype(np.float32)

        # Create an affine transformation matrix from DICOM metadata
        affine = get_affine_from_dicom(ds)

        # Create a NIfTI1Image object from the pixel data and affine
        nifti_image = nib.Nifti1Image(image_data, affine)

        # Save the NIfTI image with correct .nii or .nii.gz extension
        nib.save(nifti_image, output_nii)

        print(f"Converted DICOM to NIfTI: {output_nii}")

    except Exception as e:
        print(f"Error during DICOM to NIfTI conversion: {e}")


def spinal_cord_segmentation(input_nii, contrast='t2'):
    """
    Perform spinal cord segmentation using SCT's sct_deepseg_sc tool.

    Parameters:
    - input_nii: Path to the input NIfTI file (.nii or .nii.gz).
    - output_seg: Path where the segmented output NIfTI file will be saved.
    - contrast: MRI contrast type (default is 't2'). Options: 't1', 't2', 't2s', 'dmri'.
    """
    try:
        # Construct the command
        command = f'sct_deepseg_sc -i "{input_nii}" -c {contrast}'

        # Run the command
        subprocess.run(command, shell=True, check=True)

        print(f"Spinal cord segmentation completed and saved at: {output_seg}")

    except subprocess.CalledProcessError as e:
        print(f"Error during spinal cord segmentation: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def visualize_segmentation(nii_image_path, nii_seg_path):
    """
    Visualize the spinal cord segmentation on a single slice from NIfTI files.

    Parameters:
    - nii_image_path: Path to the original NIfTI image file (.nii or .nii.gz).
    - nii_seg_path: Path to the segmented NIfTI file (spinal cord segmentation, .nii or .nii.gz).

    Returns:
    - None (displays the image with segmentation mask overlayed).
    """
    try:
        # Load the original NIFTI image and the segmentation mask
        nii_image = nib.load(nii_image_path)
        nii_seg = nib.load(nii_seg_path)

        # Get the image and mask data
        image_data = nii_image.get_fdata()
        seg_data = nii_seg.get_fdata()

        # If data is 3D with 1 slice, extract 2D slice
        if image_data.ndim == 3 and image_data.shape[2] == 1:
            image_data = image_data[:, :, 0]
        if seg_data.ndim == 3 and seg_data.shape[2] == 1:
            seg_data = seg_data[:, :, 0]

        # Determine a central slice if data is 3D
        if image_data.ndim == 3:
            central_slice = image_data.shape[2] // 2
            image_slice = image_data[:, :, central_slice]
            seg_slice = seg_data[:, :, central_slice]
        else:
            image_slice = image_data
            seg_slice = seg_data

        # Plot the original image
        plt.imshow(image_slice, cmap='gray')  # Show the original image (single slice)

        # Overlay the segmentation mask in red using contour
        plt.contour(seg_slice, colors='r')  # Red contour for spinal cord segmentation

        # Add title and show the plot
        plt.title("Spinal Cord Segmentation (Central Slice)")
        plt.axis('off')  # Turn off the axis labels
        plt.show()

    except Exception as e:
        print(f"Error during visualization: {e}")


dicom_file = r"D:\Document\University\Scientific research\Taiwan\Spine MRI DL\Code_repo\SCT\DCOM\28210 CN.dcm"
output_nii = r"28210_CN.nii.gz"
output_seg = r"28210_CN_seg.nii.gz"

# Step 1: Convert DICOM to NIfTI
convert_dicom_to_nifti(dicom_file, output_nii)

# Step 2: Perform spinal cord segmentation
spinal_cord_segmentation(output_nii, contrast = 't2s')

# Step 3: Visualize the segmentation
visualize_segmentation(output_nii, output_seg)

