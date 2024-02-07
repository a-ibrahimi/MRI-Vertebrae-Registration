from scipy.ndimage import center_of_mass
import cv2
import numpy as np

class AffineTransformer:
    def __init__(self):
        pass

    def calculate_centroids(self, masks):
        all_centroids = []
        for mask in masks:
            centroids = {}  # Dictionary to store centroids for current mask
            unique_labels = np.unique(mask)
            unique_labels = unique_labels[unique_labels != 0]

            for label in unique_labels:
                label_mask = mask == label
                centroid = center_of_mass(label_mask)
                centroids[label] = centroid
            all_centroids.append(centroids)
        return all_centroids
        
    def get_corresponding_points(self,centroids):
        centroids_1 = centroids[0]
        centroids_2 = centroids[1]

        # Ensure that both sets have the same labels
        common_labels = set(centroids_1.keys()).intersection(set(centroids_2.keys()))

        # Extract corresponding points 
        points_1 = np.array([centroids_1[label] for label in common_labels])
        points_2 = np.array([centroids_2[label] for label in common_labels])

        return points_1, points_2
        
    def calculate_affine_transformations(self, moving_segs, fixed_segs):
        
        if len(moving_segs.shape)==4:
            moving_segs = moving_segs[...,0]
            fixed_segs = fixed_segs[...,0]
            
        batch_size = moving_segs.shape[0]
        affine_matrices = []
        moving_points_list = []
        fixed_points_list = []

        for i in range(batch_size):
            moving_seg = moving_segs[i, ...] 
            fixed_seg = fixed_segs[i, ...]

            centroids = self.calculate_centroids([moving_seg, fixed_seg])
            moving_points, fixed_points = self.get_corresponding_points(centroids)

            for i in range(len(moving_points)):
                moving_points[i] = moving_points[i][::-1]
                fixed_points[i] = fixed_points[i][::-1]

            M, _ = cv2.estimateAffinePartial2D(moving_points, fixed_points)
            moving_points_list.append(moving_points)
            fixed_points_list.append(fixed_points)
            affine_matrices.append(M)
        return affine_matrices, moving_points_list, fixed_points_list
        
    def apply_affine_transformations(self, moving_images, affine_matrices):
        if len(moving_images.shape) == 4:
            moving_images = moving_images[...,0]
        translated_images = np.zeros_like(moving_images)
        for i in range(len(affine_matrices)):
            moving_image = moving_images[i, ...]
            translated_images[i, ...] = cv2.warpAffine(moving_image, affine_matrices[i], (moving_image.shape[1], moving_image.shape[0]))
        return translated_images