import numpy as np
import nibabel as nib
import configparser
import os

if __name__ == '__main__':
    config_path = 'config.ini'
    config = configparser.ConfigParser()
    
    config.read(config_path)
    
    scans_dir = config['Paths']['ROI_scans_dir']
    seg_dir = config['Paths']['ROI_segmentations_dir']
    
    npy_dir = config['Paths']['npy_dir']
    
    scans = []
    segs = []
    
    for scan_file, seg_file in zip(sorted(os.listdir(scans_dir)), sorted(os.listdir(seg_dir))):
        
        print(scan_file, seg_file)
        
        scan = nib.load(os.path.join(scans_dir, scan_file)).get_fdata()
        seg = nib.load(os.path.join(seg_dir, seg_file)).get_fdata()
        
        scans.append(scan)
        segs.append(seg)
        
    scans = np.array(scans)
    segs = np.array(segs)
    
    print(scans.shape, segs.shape)
    
    np.save(os.path.join(npy_dir, "scans.npy"), scans)
    np.save(os.path.join(npy_dir, "segmentations.npy"), segs)