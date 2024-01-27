# MRI-Vertebrae-Registration

## Project Description

This repository is dedicated to the exploration and implementation of deep learning models for image registration of vertebrae, as part of the practical class "Advanced Deep Learning in Medicine." The project focuses initially on the application of Voxelmorph to spine MRI and CT scans.

The primary emphasis lies in training models for intra-modality registration within the same modality (MR). As the project evolves, the scope will extend to cover inter-patient intra-modality registration, specifically targeting MRI sequences such as T1w and T2w.

## Project Structure

The development of this project was divided into several key phases:

### Literature Review

The initial phase involved an extensive literature review to understand the current landscape of image registration techniques, specifically within the context of MRI scans.

### Baseline Model with VoxelMorph

We began by implementing a baseline model using VoxelMorph for intra T1w MRI scans registration. This served as the foundation for further enhancements.

### Preprocessing Pipeline

Prior to model development, a robust preprocessing pipeline was crucial. This included techniques to prepare and preprocess MRI scans, ensuring optimal input for subsequent registration models.

### Model Progression

The baseline model was progressively refined and expanded to increase complexity. Different strategies were employed to enhance registration accuracy and overall performance.

#### Semisupervised Learning

We explored semisupervised learning by incorporating segmentation masks of the vertebrae. This approach leverages additional information to improve the model's understanding of the registration task.

#### Affine Transformations

To further improve results, we implemented affine transformations. This involved aligning moving and fixed images using transformation matrices, enhancing the overall registration process.

#### Region of Interest (ROI) Experimentation

In an experimental phase, we focused on using only the Region of Interest (ROI) by identifying the maximum minimum bounding box. This exploration aimed to understand the impact of limiting the registration process to specific regions.

### Still a Work in Progress... 

Currently expermenting with spatially-variant and adaptive regularization.

## Requirements

To run the project, follow these steps:

1. **Create a Conda Environment**:

    ```bash
    conda create --name your_environment_name python=3.8
    conda activate your_environment_name
    ```

2. **Install Dependencies**:

    ```bash
    conda install -c simpleitk -c anaconda -c conda-forge nibabel jupyter simpleitk pillow pyparsing matplotlib
    ```

3. **Install Python Requirements**:

    ```bash
    pip install -r requirements.txt
    ```

## Folder Structure

```txt
.
├── BIDS
├── README.md
├── data
│   └── labels.npy
├── MRIProcessor.py
├── requirements.txt
├── utils
│   ├── __init__.py
│   └── helpers.py
└── voxelmorph
    ├── __init__.py
    ├── generators.py
    ├── model_weights
    ├── scripts
    │   └── train_semisupervised_3d.py
    └── tensorboard
```

### BIDS 📁

The `BIDS` directory is a versatile package designed to handle BIDS-conform datasets, including CT, MRI, etc. It provides functions for finding, filtering, and searching BIDS families and subjects. For more details, refer to the [BIDS README](BIDS/README.md).

### Voxelmorph 📁

The `voxelmorph` folder encapsulates tools and subfolders related to working with the Voxelmorph model.

- **`generators.py`**: This module provides a list of data generators that can generate data in the correct format for Voxelmorph models.

- **`scripts`**: Contains script files for training and testing various Voxelmorph models.

- **`tensorboard`**: A directory to store TensorBoard logs of losses and evaluation metrics, which can be visualized later using TensorBoard commands.

- **`model_weights`**: A directory to store model weights.

### Utils 📁

The `utils` folder contains functionalities that are useful when working on this project.

- **`helpers.py`**: Defines a set of methods for tasks such as visualizing scans, file handling, etc.

### Data 📁

The `data` directory is intended to store dataset-related files.

### MRIProcessor.py

Defines a class `MRIProcessor` for preprocessing `.nii.gz` files.

## Project Reference

## Project Team

## Project License

## Project Acknowledgement

## Project Contact
