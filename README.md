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

To be able to run the project, you need first to create a conda environment and run the following commands

```bash
conda install -c simpleitk -c anaconda -c conda-forge nibabel jupyter simpleitk pillow pyparsing matplotlib
pip install requirements.txt
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

### BIDS

A multi-functional package to handle any sort of bids-conform dataset (CT, MRI, ...) It can find, filter, search any BIDS_Family and subjects, and has many functionalities. For more info, check [BIDS README](BIDS/README.md).

### Voxelmorph

The voxelmoprh folder encapsulates tools and subfolders related to working with the voxelmorph model.

- `generators.py`: Provides a list of data generators that can be used to generate data in the right format for voxelmorph models

- `scripts`: Provides a list of script files to train and test different voxelmorph models

- `tensorboard`: A folder to store tensorboard logs of losses and evaluation metrics which you can later visualize using tensorboard commands

- `modelweights`: A folder to store model weights

### utils

The utils folder is for functionalities that are helpful when working in this project

- `helpers.py`: defines a list of methods which can be used for visualizing scans, file handling, etc...

#### data

A folder to store data.

### `MRIProcessor.py`

Defines a class `MRIProcessor` for preprocessing `.nii.gz` files

## Project Reference

## Project Team

## Project License

## Project Acknowledgement

## Project Contact
