"""
Script: convert_nifti_to_npy.py

Description:
This script converts NIfTI files (scans and segmentations) to NumPy arrays
and saves them in a specified directory. It reads file paths and configurations
from the 'config.ini' file.

Dependencies:
- numpy
- nibabel
- configparser

Usage:
Run this script to convert NIfTI files to NumPy arrays as specified in the
'config.ini' file. It saves the resulting arrays in the specified 'npy_dir'.
"""

import numpy as np
import nibabel as nib
import configparser
import os

if __name__ == '__main__':
    # Read configuration file
    config_path = 'config.ini'
    config = configparser.ConfigParser()
    config.read(config_path)
    
    # Define directories from configuration
    scans_dir = config['Paths']['ROI_scans_dir']
    seg_dir = config['Paths']['ROI_segmentations_dir']
    npy_dir = config['Paths']['npy_dir']
    
    # Initialize lists to store scans and segmentations
    scans = []
    segs = []
    
    # Loop through pairs of scan and segmentation files
    for scan_file, seg_file in zip(sorted(os.listdir(scans_dir)), sorted(os.listdir(seg_dir))):
        print(scan_file, seg_file)
        
        # Load NIfTI files and convert to NumPy arrays
        scan = nib.load(os.path.join(scans_dir, scan_file)).get_fdata()
        seg = nib.load(os.path.join(seg_dir, seg_file)).get_fdata()
        
        # Append to lists
        scans.append(scan)
        segs.append(seg)
        
    # Convert lists to NumPy arrays
    scans = np.array(scans)
    segs = np.array(segs)
    
    # Save NumPy arrays as .npy files
    np.save(os.path.join(npy_dir, "scans.npy"), scans)
    np.save(os.path.join(npy_dir, "segmentations.npy"), segs)
