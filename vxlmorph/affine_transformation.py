from scipy.ndimage import center_of_mass, affine_transform
import cv2
import numpy as np

def calculate_centroids(seg):
    """
    Calculate the centroids of the input segmentation.
        
    Parameters:
    - seg (numpy.ndarray): Input segmentation.
        
    Returns:
    - dict: A dictionary containing the centroids of the input segmentation.
    """
    centroids = {}  # Dictionary to store centroids for current mask
    unique_labels = np.unique(seg) # Get unique labels in the mask
    unique_labels = unique_labels[unique_labels != 0] # Remove background label

    for label in unique_labels:
        label_mask = seg == label # Create a mask for the current label
        centroid = center_of_mass(label_mask) # Calculate the centroid of the current label
        centroids[label] = centroid # Store the centroid in the dictionary

    return centroids
    
def get_corresponding_points(
    moving_centroids, fixed_centroids):
    """
    Get corresponding points from two sets of centroids.
        
    Parameters:
    - moving_centroids (dict): A dictionary containing the centroids of the moving mask.
    - fixed_centroids (dict): A dictionary containing the centroids of the fixed mask.
        
    Returns:
    - tuple: A tuple containing the following elements:
    """

    # Ensure that both sets have the same labels
    common_labels = set(moving_centroids.keys()).intersection(set(fixed_centroids.keys()))

    # Extract corresponding points 
    moving_points = np.array([moving_centroids[label] for label in common_labels])
    fixed_points = np.array([fixed_centroids[label] for label in common_labels])

    return moving_points, fixed_points
    
def calculate_affine_transformations(moving_segs, fixed_segs, volumetric=False):

    """
    Calculates affine transformation matrices based on the centroids of the input masks.
        
    Parameters:
    - moving_segs (numpy.ndarray): Moving segmentation masks.
    - fixed_segs (numpy.ndarray): Fixed segmentation masks.
    - volumetric (bool): Indicates the given data type, True for 3D. (default is False)
        
    Returns:
    - tuple: A tuple containing the following elements:
        - list: A list of affine transformation matrices.
        - list: A list of moving points.
        - list: A list of fixed points.
    """
    # Remove the channel dimension
    moving_segs = moving_segs[..., 0] 
    fixed_segs = fixed_segs[..., 0] 
        
    batch_size = moving_segs.shape[0]
    affine_matrices = []
    moving_points_list = []
    fixed_points_list = []

    for i in range(batch_size):
        # Get the current moving and fixed masks
        moving_seg = moving_segs[i, ...] 
        fixed_seg = fixed_segs[i, ...] 
        
        # Calculate the centroids of the moving and fixed masks
        moving_centroids = calculate_centroids(moving_seg)
        fixed_centroids = calculate_centroids(fixed_seg)

        moving_points, fixed_points = get_corresponding_points(moving_centroids, fixed_centroids) # Get corresponding points
        
        # Reverse the points to match the input format of the cv2.estimateAffinePartial2D function
        for i in range(len(moving_points)):
            moving_points[i] = moving_points[i][::-1]
            fixed_points[i] = fixed_points[i][::-1]
        
        # Calculate the affine transformation matrix
        if volumetric:
            out = cv2.estimateAffine3D(moving_points, fixed_points, force_rotation=True)
            M = np.vstack((out[0], [0, 0, 0, 1]))
        else:
            M, _ = cv2.estimateAffinePartial2D(moving_points, fixed_points)

        moving_points_list.append(moving_points)
        fixed_points_list.append(fixed_points)
        affine_matrices.append(M)
    return affine_matrices, moving_points_list, fixed_points_list
    
def apply_affine_transformations(moving_images, affine_matrices, volumetric=False):
    """
    Applies affine transformations to the input images.
        
    Parameters:
    - moving_images (numpy.ndarray): Moving images.
    - affine_matrices (list): A list of affine transformation matrices.
        
    Returns:
    - numpy.ndarray: Translated images.
    """

    moving_images = moving_images[...,0] # Remove the channel dimension
    translated_images = np.zeros_like(moving_images)

    # Apply the affine transformations
    for i in range(len(affine_matrices)):
        moving_image = moving_images[i, ...]
        if volumetric:
            affine_matrix_inv = np.linalg.inv(affine_matrices[i])
            translated_images[i, ...] = affine_transform(
                                    moving_image, 
                                    affine_matrix_inv[:3, :3], 
                                    offset=-affine_matrix_inv[:3, 3], 
                                    output_shape=moving_image.shape,  
                                    order=1)
        else:
            translated_images[i, ...] = cv2.warpAffine(moving_image, affine_matrices[i], (moving_image.shape[1], moving_image.shape[0]))

    return np.expand_dims(translated_images, axis=-1)