import numpy as np
import os
from BIDS import *
import configparser
import nibabel as nib

def get_file_number(path):
    return path.split('/')[-1][-11, -7] #workes with the naming convention used when saving files by MRIProcessor or SegmentationProcessor (path/to/file/file000x.nii.gz)

def find_bounding_box(segmentation_data):
    # Find indices where the segmentation mask is non-zero
    non_zero_indices = np.nonzero(segmentation_data)
    
    # Get minimum and maximum coordinates along each axis
    min_coords = np.min(non_zero_indices, axis=1)
    max_coords = np.max(non_zero_indices, axis=1)
    
    return min_coords, max_coords

if __name__ == '__main__':
    
    config = configparser.ConfigParser()
    config.read('config.ini')
                    
    data_dir = config['Paths']['preprocessed_scans_dir']
    seg_dir = config['Paths']['preprocessed_segmentations_dir']
    
    ROI_scans_dir = config['Paths']['ROI_scans_dir']
    ROI_seg_dir = config['Paths']['ROI_segmentations_dir']
    
    #Should be calculated by calculating the largest minimum bounding box among all images
    bounding_box_shape = (96, 128, 256)
    
    for img_file, seg_file in zip(sorted(os.listdir(data_dir)), sorted(os.listdir(seg_dir))):
        img = nib.load(os.path.join(data_dir, img_file)).get_fdata()
        seg = nib.load(os.path.join(seg_dir, seg_file)).get_fdata()
        
        assert (img.shape == seg.shape and get_file_number(os.path.join(data_dir, img_file)) == get_file_number(os.path.join(seg_dir, seg_file)))
        
        f_num = get_file_number(os.path.join(data_dir, img_file))
        
        minc, maxc = find_bounding_box(seg)
                
        curr_shape = maxc - minc
        
        diff = bounding_box_shape - curr_shape
        l_offset = diff//2
        r_offset = diff - l_offset
        
        minc -= l_offset
        maxc += r_offset
        
        l_idx = np.maximum(minc, 0)
        r_idx = maxc
        
        cropped_img = img[l_idx[0]:r_idx[0], l_idx[1]:r_idx[1], l_idx[2]:r_idx[2]]
        cropped_seg = seg[l_idx[0]:r_idx[0], l_idx[1]:r_idx[1], l_idx[2]:r_idx[2]]
        
        # Handle the case where the shape is less than bounding_box_shape
        img_shape = cropped_img.shape
        pad_x = max(bounding_box_shape[0] - img_shape[0], 0)
        pad_y = max(bounding_box_shape[1] - img_shape[1], 0)
        pad_z = max(bounding_box_shape[2] - img_shape[2], 0)
        
        # Calculate the total amount of padding required on each axis
        pad_x_before = pad_x // 2
        pad_x_after = pad_x - pad_x_before
        pad_y_before = pad_y // 2
        pad_y_after = pad_y - pad_y_before
        pad_z_before = pad_z // 2
        pad_z_after = pad_z - pad_z_before

        # Pad the images equally on both sides
        final_img = np.pad(cropped_img, ((pad_x_before, pad_x_after), (pad_y_before, pad_y_after), (pad_z_before, pad_z_after)), mode='constant', constant_values=0)
        final_seg = np.pad(cropped_seg, ((pad_x_before, pad_x_after), (pad_y_before, pad_y_after), (pad_z_before, pad_z_after)), mode='constant', constant_values=0)
        
        assert final_img.shape == final_seg.shape == bounding_box_shape
        
        final_img = nib.Nifti1Image(final_img, affine = np.eye(4))
        final_seg = nib.Nifti1Image(final_seg, affine = np.eye(4))
        
        nib.save(final_img, os.path.join(ROI_scans_dir, img_file))
        nib.save(final_seg, os.path.join(ROI_seg_dir, seg_file))      