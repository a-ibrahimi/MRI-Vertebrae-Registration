import numpy as np

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
        
        if(segs):
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