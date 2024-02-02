import numpy as np
import os
from BIDS import *

from helpers import get_fnumber

def find_bounding_box(segmentation_data):
    # Find indices where the segmentation mask is non-zero
    non_zero_indices = np.nonzero(segmentation_data)
    
    # Get minimum and maximum coordinates along each axis
    min_coords = np.min(non_zero_indices, axis=1)
    max_coords = np.max(non_zero_indices, axis=1)
    
    return min_coords, max_coords

if __name__ == '__main__':
                    
    data_dir = '/local_ssd/practical_wise24/vertebra_labeling/data'
    bounding_box_shape = (96, 128, 256)
    
    images = []
    seg_images = []
    i = 0
    
    # for i in range(258):
    #     if os.path.exists(f'/u/home/iba/practice/Data/NewPreprocessed/file{i:04d}.nii.gz'):
            
    
    for subject_dir in os.listdir(os.path.join(data_dir, 'spider_raw')):
        subject_path = os.path.join(data_dir, 'spider_raw', subject_dir, 'T1w')
        
        if os.path.exists(subject_path):
            nii_files = [file for file in os.listdir(subject_path) if file.endswith('.nii.gz')]
            
            if len(nii_files) > 0:
                img = preprocess_img(f'/u/home/iba/practice/Data/Preprocessed/file{get_fnumber(os.path.join(subject_path, nii_files[0]))}.nii.gz')
                seg = preprocess_seg(f'/local_ssd/practical_wise24/vertebra_labeling/data/dataset-spider/derivatives/sub-{get_fnumber(os.path.join(subject_path, nii_files[0]))}/T1w/sub-{get_fnumber(os.path.join(subject_path, nii_files[0]))}_mod-T1w_seg-vert_msk.nii.gz')
                assert img.shape == seg.shape
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
                
                images.append(final_img)
                seg_images.append(final_seg)
                i+=1
                
    images = np.array(images)
    seg_images = np.array(seg_images)
    
    np.save('/u/home/iba/practice/3D/images1.npy', images)
    np.save('/u/home/iba/practice/3D/segmentations1.npy', seg_images)
                
            
                
                

                