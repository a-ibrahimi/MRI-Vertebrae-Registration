import numpy as np

import vxlmorph.affine_transformation as affine_transformation

def volgen(
    vol_names,
    batch_size=1,
    segs=None,
):
    """
    Volume Data Generator.

    This function generates batches of random volumes and, optionally, their corresponding segmentations.

    Parameters:
    - vol_names (numpy.ndarray): List of volume names, paths, or preloaded volumes.
    - batch_size (int, optional): Number of volumes to generate in each batch. Default is 1.
    - segs (numpy.ndarray or None, optional): List of segmentation names, paths, or preloaded segmentations.
      If provided, corresponding segmentations will be loaded and yielded with volumes. Default is None.

    Yields:
    - tuple: A tuple containing the generated volumes and, if segs is provided, their corresponding segmentations.
      The tuple structure is (vols, seg) where:
      - vols (numpy.ndarray): Batch of volumes with shape (batch_size, *volume_shape, 1).
      - seg (numpy.ndarray or None): Batch of segmentations with shape (batch_size, *segmentation_shape, 1).
        If segs is None, seg is also None.

    Example:
    ```python
    # Generate batches of volumes without segmentations
    generator = volgen(vol_names, batch_size=8)
    volumes = next(generator)

    # Generate batches of volumes with corresponding segmentations
    generator = volgen(vol_names, batch_size=8, segs=seg_names)
    volumes, segmentations = next(generator)
    ```

    Note:
    - If segs is None, the yielded tuple will contain (vols, None).
    - Volumes and segmentations are loaded randomly from the provided list or paths in each batch.
    - The volumes are reshaped to include a singleton channel axis at the end.

    """
    while True:
        # generate [batchsize] random image indices
        indices = np.random.randint(len(vol_names), size=batch_size)

        # load volumes and concatenate
        vols = vol_names[indices, ..., np.newaxis]
        
        if(segs is not None):
            seg = segs[indices, ..., np.newaxis]

        yield vols, seg
        
def semisupervised(vol_names, seg_names, labels, batch_size=16, downsize=1):
    
    # configure base generator
    gen = volgen(vol_names, segs=seg_names, batch_size=batch_size)
    zeros = None

    # internal utility to generate downsampled prob seg from discrete seg
    def split_seg(seg):
        prob_seg = np.zeros((*seg.shape[:4], len(labels)))
        for i, label in enumerate(labels):
            prob_seg[0, ..., i] = (seg[0, ..., 0] == label)
        return prob_seg[:, ::downsize, ::downsize, ::downsize, :]

    while True:
        # load source vol and seg
        src_vol, src_seg = next(gen)
        src_seg = split_seg(src_seg)

        trg_vol, trg_seg = next(gen)
        trg_seg = split_seg(trg_seg)

        # cache zeros
        if zeros is None:
            shape = src_vol.shape[1:-1]
            zeros = np.zeros((batch_size, *shape, len(shape)))

        invols = [src_vol, trg_vol, src_seg]
        outvols = [trg_vol, zeros, trg_seg]
        
        yield (invols, outvols)
    
def semisupervised_affine(vol_names, seg_names, labels, batch_size=16, downsize=1, volumetric=False):
    """
    Semi-supervised Data Generator with Affine Transformations.

    Parameters:
    - vol_names (numpy.ndarray): List of volume names, paths, or preloaded volumes.
    - seg_names (numpy.ndarray): List of segmentation names, paths, or preloaded segmentations.
    - labels (list): List of labels to consider in the segmentation.
    - batch_size (int): Number of volumes to generate in each batch.
    - downsize (int): Downsize factor for the segmentation.
    - volumetric (bool): Indicates the given data type, True for 3D. (default is False)
    
    Yields:
    - tuple: A tuple containing the generated volumes and, if segs is provided, their corresponding segmentations.
    """

    # configure base generator
    gen = volgen(vol_names, batch_size, seg_names)
    zeros = None

    # internal utility to generate downsampled prob seg from discrete seg
    def split_seg(seg, volumetric=False):
        print(seg.shape)
        if volumetric:
            dim = 3
        else:
            dim = 2
        print(dim)
        prob_seg = np.zeros((*seg.shape[:dim+1], len(labels)))
        print(prob_seg.shape)
        for i, label in enumerate(labels):
            prob_seg[0, ..., i] = (seg[0, ..., 0] == label)
        return prob_seg[:, ::downsize, ::downsize]

    while True:
        # load source vol and seg
        src_vol, src_seg = next(gen)
        trg_vol, trg_seg = next(gen)
        
        # calculate affine transformations and transform source vol and seg
        affine_matrices, _, __ = affine_transformation.calculate_affine_transformations(src_seg, trg_seg, volumetric=volumetric)
        src_vol_tr = affine_transformation.apply_affine_transformations(src_vol, affine_matrices, volumetric=volumetric)
        src_seg_tr = affine_transformation.apply_affine_transformations(src_seg, affine_matrices, volumetric=volumetric)
        src_seg_tr = split_seg(src_seg_tr, volumetric=volumetric)
        trg_seg = split_seg(trg_seg, volumetric=volumetric)

        vol_shape = src_vol.shape[1:]
        ndims = len(vol_shape)

        # cache zeros
        if zeros is None:
            zeros = np.zeros((batch_size, *vol_shape, ndims))
                
        invols = [src_vol_tr, trg_vol, src_seg_tr]
        outvols = [trg_vol, zeros, trg_seg]
            
        yield (invols, outvols)
    
def vxm_data_generator_affine(x_data, seg_data=None, batch_size=16):
    """
    Data Generator with Affine Transformations.
    
    Parameters:
    - x_data (numpy.ndarray): Input data.
    - seg_data (numpy.ndarray or None): Segmentation data. Default is None.
    - batch_size (int): Number of volumes to generate in each batch.

    Yields:
    - tuple: A tuple containing the generated volumes.
    """

    affine_transformer = AffineTransformer()

    # preliminary sizing
    vol_shape = x_data.shape[1:] # extract data shape
    ndims = len(vol_shape)
        
    # prepare a zero array the size of the deformation
    # we'll explain this below
    zero_phi = np.zeros([batch_size, *vol_shape, ndims])
        
    while True:
        # prepare inputs:
        # images need to be of the size [batch_size, H, W, 1]
        idx1 = np.random.randint(0, x_data.shape[0], size=batch_size)
        idx2 = np.random.randint(0, x_data.shape[0], size=batch_size)

        moving_images = x_data[idx1, ..., np.newaxis]
        fixed_images = x_data[idx2, ..., np.newaxis] 

        if seg_data is not None:
            moving_segs = seg_data[idx1, ..., np.newaxis]
            fixed_segs = seg_data[idx2, ..., np.newaxis]

            affine_matrices, _, __ = affine_transformation.calculate_affine_transformations(moving_segs, fixed_segs)
            translated_images = affine_transformation.apply_affine_transformations(moving_images, affine_matrices)

            inputs = [translated_images, fixed_images]
        else:
            inputs = [moving_images, fixed_images] 
            
        # prepare outputs (the 'true' moved image):
        # of course, we don't have this, but we know we want to compare 
        # the resulting moved image with the fixed image. 
        # we also wish to penalize the deformation field. 
        outputs = [fixed_images, zero_phi]
            
        yield (inputs, outputs)