"""
This script preprocesses MRI scans and segmentations by cropping and padding them to a specified bounding box shape.
The preprocessed scans and segmentations are then saved in a new directory for region-of-interest analysis.
The script reads configuration from 'config.ini' and follows a specific file naming convention.
"""

import numpy as np
import os
import configparser
import nibabel as nib

def get_file_number(path):
    """
    Extracts the file number from the given file path.

    This function assumes a specific naming convention used when saving files by MRIProcessor or SegmentationProcessor.
    The convention is typically like "path/to/file/file000x.nii.gz" where 'x' represents the file number.

    Args:
        path (str): The file path from which to extract the file number.

    Returns:
        str: The extracted file number.
    """
    return path.split('/')[-1][-11: -7]

def find_bounding_box(segmentation_data):
    """
    Finds the bounding box coordinates for a given segmentation mask.

    Args:
        segmentation_data (numpy.ndarray): 3D array representing the segmentation mask.

    Returns:
        tuple: A tuple containing the minimum and maximum coordinates along each axis
               (e.g., ((min_x, min_y, min_z), (max_x, max_y, max_z))).
    """
    
    # Find indices where the segmentation mask is non-zero
    non_zero_indices = np.nonzero(segmentation_data)
    
    # Get minimum and maximum coordinates along each axis
    min_coords = np.min(non_zero_indices, axis=1)
    max_coords = np.max(non_zero_indices, axis=1)
    
    return min_coords, max_coords

if __name__ == '__main__':
    # Read configuration from config.ini
    config = configparser.ConfigParser()
    config.read('config.ini')

    # Set paths for preprocessed scans and segmentations, and ROI scans and segmentations
    data_dir = config['Paths']['preprocessed_scans_dir']
    seg_dir = config['Paths']['preprocessed_segmentations_dir']
    ROI_scans_dir = config['Paths']['ROI_scans_dir']
    ROI_seg_dir = config['Paths']['ROI_segmentations_dir']

    # Define the desired bounding box shape for cropping
    bounding_box_shape = (96, 128, 256)

    # Iterate through each pair of preprocessed scan and segmentation
    for img_file, seg_file in zip(sorted(os.listdir(data_dir)), sorted(os.listdir(seg_dir))):
        print('*')
        # Load the preprocessed scan and segmentation
        img = nib.load(os.path.join(data_dir, img_file)).get_fdata()
        seg = nib.load(os.path.join(seg_dir, seg_file)).get_fdata()

        # Ensure shape consistency and matching file numbers
        assert (img.shape == seg.shape and get_file_number(os.path.join(data_dir, img_file)) == get_file_number(os.path.join(seg_dir, seg_file)))

        # Get the file number for identification
        f_num = get_file_number(os.path.join(data_dir, img_file))

        # Find the bounding box coordinates for segmentation
        minc, maxc = find_bounding_box(seg)

        # Calculate current shape and the difference with the desired bounding box shape
        curr_shape = maxc - minc
        diff = bounding_box_shape - curr_shape

        # Calculate left and right offsets for cropping
        l_offset = diff // 2
        r_offset = diff - l_offset

        # Update bounding box coordinates with offsets
        minc -= l_offset
        maxc += r_offset

        # Calculate left and right indices for cropping
        l_idx = np.maximum(minc, 0)
        r_idx = maxc

        # Crop the images
        cropped_img = img[l_idx[0]:r_idx[0], l_idx[1]:r_idx[1], l_idx[2]:r_idx[2]]
        cropped_seg = seg[l_idx[0]:r_idx[0], l_idx[1]:r_idx[1], l_idx[2]:r_idx[2]]

        # Handle the case where the shape is less than bounding_box_shape
        img_shape = cropped_img.shape
        pad_x = max(bounding_box_shape[0] - img_shape[0], 0)
        pad_y = max(bounding_box_shape[1] - img_shape[1], 0)
        pad_z = max(bounding_box_shape[2] - img_shape[2], 0)

        # Calculate padding on each axis
        pad_x_before = pad_x // 2
        pad_x_after = pad_x - pad_x_before
        pad_y_before = pad_y // 2
        pad_y_after = pad_y - pad_y_before
        pad_z_before = pad_z // 2
        pad_z_after = pad_z - pad_z_before

        # Pad the images
        final_img = np.pad(cropped_img, ((pad_x_before, pad_x_after), (pad_y_before, pad_y_after), (pad_z_before, pad_z_after)), mode='constant', constant_values=0)
        final_seg = np.pad(cropped_seg, ((pad_x_before, pad_x_after), (pad_y_before, pad_y_after), (pad_z_before, pad_z_after)), mode='constant', constant_values=0)

        # Ensure the final shapes match the desired bounding box shape
        assert final_img.shape == final_seg.shape == bounding_box_shape

        # Save the final cropped and padded images
        final_img = nib.Nifti1Image(final_img, affine=np.eye(4))
        final_seg = nib.Nifti1Image(final_seg, affine=np.eye(4))
        nib.save(final_img, os.path.join(ROI_scans_dir, img_file))
        nib.save(final_seg, os.path.join(ROI_seg_dir, seg_file))
