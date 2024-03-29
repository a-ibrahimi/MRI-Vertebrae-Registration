import os
import SimpleITK as sitk
from utils import get_fnumber
import nibabel as nib
from BIDS import to_nii
import numpy as np
import configparser


class MRIProcessor:
    def __init__(self, config_path='config.ini'):
        """
        Initializes the MRIProcessor with configuration parameters.

        Args:
            config_path (str, optional): Path to the configuration file. Defaults to 'config.ini'.
        """
        # Load configuration from file
        self.config = configparser.ConfigParser()
        self.config.read(config_path)

        # Set parameters from the configuration
        self.data_dir = self.config['Paths']['data_dir']
        self.preprocessed_dir = self.config['Paths']['preprocessed_scans_dir']
        self.desired_orientation = tuple(self.config['PreprocessingParameters']['desired_orientation'].split(','))
        self.shrink_factor = int(self.config['PreprocessingParameters']['shrink_factor'])
        self.template_img_path = self.config['PreprocessingParameters']['template_img_path']

    def preprocess_data(self):
        """
        Preprocesses MRI data by correcting bias, normalizing, and saving the preprocessed images.
        """
        for subject_dir in os.listdir(os.path.join(self.data_dir, 'spider_raw')):
            subject_path = os.path.join(self.data_dir, 'spider_raw', subject_dir, 'T1w')

            if os.path.exists(subject_path):
                nii_files = [file for file in os.listdir(subject_path) if file.endswith('.nii.gz')]

                if len(nii_files) > 0:
                    target_file = f'{self.preprocessed_dir}/image{get_fnumber(os.path.join(subject_path, nii_files[0]))}.nii.gz'
                    img = self.correct_bias(os.path.join(subject_path, nii_files[0]))
                    normalized = self.normalize(img)
                    sitk.WriteImage(normalized, target_file)

                    img = nib.load(target_file)
                    nii_img = to_nii(img, seg=False).rescale_and_reorient(axcodes_to=self.desired_orientation, voxel_spacing=(1, 1, 1), verbose=True)

                    nib.save(nii_img.nii, target_file)

    def correct_bias(self, raw_img_path):
        """
        Corrects bias in the given MRI image.

        Args:
            raw_img_path (str): Path to the raw MRI image.

        Returns:
            sitk.Image: Corrected MRI image.
        """
        print('Reading Image...')
        raw_img_sitk = sitk.ReadImage(raw_img_path, sitk.sitkFloat32)
        raw_img_sitk = sitk.DICOMOrient(raw_img_sitk, 'LPS')
        print('Rescaling Image...')
        transformed = sitk.RescaleIntensity(raw_img_sitk, 0, 255)

        transformed = sitk.LiThreshold(transformed, 0, 1)

        head_mask = transformed
        inputImage = raw_img_sitk

        inputImage = sitk.Shrink(raw_img_sitk, [self.shrink_factor] * inputImage.GetDimension())
        maskImage = sitk.Shrink(head_mask, [self.shrink_factor] * inputImage.GetDimension())

        bias_corrector = sitk.N4BiasFieldCorrectionImageFilter()

        print('Correcting Bias...')
        corrected = bias_corrector.Execute(inputImage, maskImage)
        log_bias_field = bias_corrector.GetLogBiasFieldAsImage(raw_img_sitk)
        corrected_image_full_resolution = raw_img_sitk / sitk.Exp(log_bias_field)

        return corrected_image_full_resolution

    def normalize(self, img):
        """
        Normalizes the given MRI image using histogram matching.

        Args:
            img (sitk.Image): MRI image to be normalized.

        Returns:
            sitk.Image: Normalized MRI image.
        """
        template_img_sitk = sitk.ReadImage(self.template_img_path, sitk.sitkFloat64)
        template_img_sitk = sitk.DICOMOrient(template_img_sitk, 'LPS')

        transformed = sitk.HistogramMatching(img, template_img_sitk)
        return transformed
