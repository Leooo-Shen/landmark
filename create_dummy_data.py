"Create a dummy 3D volume from 2D image"

import numpy as np
import matplotlib.pyplot as plt

def create_dummy_3d(img_path, nslice=10):

    # Create a dummy 3D volume by stacking the 2D image along the third dimension
    image_2d = plt.imread(img_path)
    dummy_volume = np.stack([image_2d] * nslice, axis=0)
    # add 1 as the first dimension of dummy_volume
    # dummy_volume = np.expand_dims(dummy_volume, axis=0)
    
    img_name = img_path.split('/')[-1]
    save_name = img_name.replace('.png', '.nii.gz')
    save_dir = 'data/3D/S_Point/Images_without'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    # save the dummy volume as a nifti file
    import nibabel as nib
    nii_image = nib.Nifti1Image(dummy_volume, affine=np.eye(4)) 
    nib.save(nii_image, os.path.join(save_dir, save_name))
    
    
    
if __name__ == '__main__':
    import os 
    img_dir = 'data/2D/S_Point/Images_without'
    
    
    files = os.listdir(img_dir)
    for file in files:
        img_path = os.path.join(img_dir, file)
        create_dummy_3d(img_path)
