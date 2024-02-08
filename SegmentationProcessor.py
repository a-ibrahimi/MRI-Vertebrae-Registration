import nibabel as nib
import numpy as np
import configparser
import os
from BIDS import *
from utils.helpers import get_fnumber

class SegmentationProcessor:
    def __init__(self, config_path='config.ini'):
        """
        Initializes the SegmentationProcessor with configuration parameters.

        Args:
            config_path (str, optional): Path to the configuration file. Defaults to 'config.ini'.
        """
        # Load configuration from file
        self.config = configparser.ConfigParser()
        self.config.read(config_path)
        
        self.seg_directory = self.config['Paths']['seg_dir']
        self.desired_orientation = tuple(self.config['PreprocessingParameters']['desired_orientation'].split(','))
        self.preprocessed_dir = self.config['Paths']['preprocessed_segmentations_dir']
        
    def preprocess_seg(self):
        """
        Preprocesses segmentation masks by reorienting and rescaling.

        Saves the preprocessed segmentation masks.

        """
        for subject_dir in os.listdir(os.path.join(self.seg_directory, 'derivatives')):
            subject_path = os.path.join(self.seg_directory, 'derivatives', subject_dir, 'T1w')
            
            if os.path.exists(subject_path):
                nii_files = [file for file in os.listdir(subject_path) if file.endswith('vert_msk.nii.gz')]
                
                if len(nii_files) > 0:

                    img = nib.load(os.path.join(subject_path, nii_files[0]))
                    nii_img = to_nii(img, seg=True)
                    nii_img = nii_img.rescale_and_reorient_(axcodes_to=self.desired_orientation, voxel_spacing=(1,1,1), verbose=True)
                    nib.save(nii_img.nii, f'{self.preprocessed_dir}/segmentation{get_fnumber(os.path.join(subject_path, nii_files[0]))}.nii.gz')
        
