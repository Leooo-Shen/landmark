"Create a dummy 3D volume from 2D image"

import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
from tqdm import tqdm
import os
import nibabel as nib
import csv
import pandas as pd
from sklearn.model_selection import train_test_split
import torchvision
from torchvision.transforms import ColorJitter, GaussianBlur
from PIL import Image


def duplicate_2d_to_3d(src_dir, dst_dir, nslice=16):
    """Duplicate the 2D image to 3D volume"""
    
    print(f'Duplicate the 2D image to 3D volume with {nslice} slices')
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    files = os.listdir(src_dir)
    files = [file for file in files if file.endswith('.png')]
    
    for filepath in tqdm(files):
        img_path = os.path.join(src_dir, filepath)
        image_2d = plt.imread(img_path)
        
        dummy_volume = np.stack([image_2d] * nslice, axis=0)
        # add 1 as the first dimension of dummy_volume
        # dummy_volume = np.expand_dims(dummy_volume, axis=0)
        
        img_name = img_path.split('/')[-1]
        save_name = img_name.replace('.png', '.nii.gz')

        # save the dummy volume as a nifti file
        nii_image = nib.Nifti1Image(dummy_volume, affine=np.eye(4)) 
        nib.save(nii_image, os.path.join(dst_dir, save_name))
        

# def create_dummy_images_2d_interpolate(src_dir, dst_dir, num_images=100):
#     """Create dummy images by linear interpolating two images together"""
#     print('Create dummy images by linear interpolating two images together')
    
#     if not os.path.exists(dst_dir):
#         os.makedirs(dst_dir)
#     files = os.listdir(src_dir)
#     # only keep files ending with .png
#     files = [file for file in files if file.endswith('.png')]
    
#     for n in tqdm(range(6, num_images+1)):
#         # random pick 2 images and load
#         random.shuffle(files)
#         files = files[:2]
    
#         image1 = cv2.imread(os.path.join(src_dir, files[0]), cv2.COLOR_BGR2GRAY)
#         image2 = cv2.imread(os.path.join(src_dir, files[1]), cv2.COLOR_BGR2GRAY)

#         # Ensure both images have the same dimensions
#         if image1.shape != image2.shape:
#             raise ValueError("Both images must have the same dimensions")

#         # random choose alpha between 0-0.3 and 0.7 to 1
#         alpha = random.uniform(0, 0.3)
#         interpolated_image = cv2.addWeighted(image1, 1 - alpha, image2, alpha, 0)
        
#         save_name = 'image' + str(n) + '.png'
#         cv2.imwrite(os.path.join(dst_dir, save_name), interpolated_image)
    

transform = torchvision.transforms.Compose([
    ColorJitter(brightness=0.3, contrast=0.2, saturation=0, hue=0),
    GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
                            ]) 
                    


def create_dummy_data_2d(img_src_dir, img_dst_dir, coor_src, coor_dst, num_samples=100):
    """Create dummy data. Copy paste the original image, and add noise to the coordinates"""
    gt_data = pd.read_csv(coor_src)
    gt_imgs = gt_data['name'].tolist()
    gt_imgs = [img + '.png' for img in gt_imgs]
    
    gt_coors = gt_data[['X', 'Y']].to_numpy(dtype=np.float32)
    num_duplicates = num_samples // len(gt_imgs)
    fake_data = []
    
    if not os.path.exists(img_dst_dir):
        os.makedirs(img_dst_dir)
    
    # augment the original image and save
    for img, coord in zip(gt_imgs, gt_coors):
        for i in range(num_duplicates):
            save_name = img.replace('.png', f'_{i}.png')
            src_path = os.path.join(img_src_dir, img)
            dst_path = os.path.join(img_dst_dir, save_name)
            image = Image.open(src_path)
            image = transform(image)
            image.save(dst_path)
            print(f'image saved to {dst_path}')
            
            # os.system(f'cp {src_path} {dst_path}')
            # print(f'Copied {src_path} to {dst_path}')
            
            # add noise to the coordinates
            noise = np.random.normal(0, 6, size=2)
            coord += noise
            coord = coord.round(2)
            
            new_data_point = {
                'name': save_name.replace('.png', ''),
                'X': coord[0],
                'Y': coord[1],
            }
            fake_data.append(new_data_point)
    
    fake_df = pd.DataFrame(fake_data)
    fake_df.to_csv(coor_dst, index=False)
    print('Done creating dummy data')
    print(f'Fake coors saved to {coor_dst}')


def create_dummy_z_dim(src_csv, dst_csv):
    """Randomly sample z dimension for each point"""
    data = pd.read_csv(src_csv)
    z_values = np.random.normal(8, 0.5, size=len(data)).round(2)
    data['Z'] = z_values
    data.to_csv(dst_csv, index=False)
    print(f'Fake 3d coors saved to {dst_csv}')
        

def train_val_test_split(path, seed=0):
    """
    Split the data into train and validation and test set
    Ratio: 0.7, 0.15, 0.15
    """
    
    data = pd.read_csv(path)
    root = os.path.dirname(path)
    
    train_data, val_test_data = train_test_split(data, test_size=0.3, random_state=seed)
    val_data, test_data = train_test_split(val_test_data, test_size=0.5, random_state=seed)
    train_data.to_csv(os.path.join(root,'train.csv'), index=False)
    val_data.to_csv(os.path.join(root,'val.csv'), index=False)
    test_data.to_csv(os.path.join(root,'test.csv'), index=False)
    print('train data size: ', len(train_data))
    print('val data size: ', len(val_data))
    print('test data size: ', len(test_data))
    


if __name__ == '__main__':
    # img_src_dir = 'data/2D/S_Point/Images_without'
    # img_dst_dir = 'dummydata/images/2d/'
    # coor_src = 'dummydata/S_Point/gt.csv'
    # coor_dst = 'dummydata/S_Point/fake_2d.csv'
    # create_dummy_data_2d(img_src_dir, img_dst_dir, coor_src, coor_dst, num_samples=100)
    
    src_dir = 'dummydata/images/2d/'
    dst_dir = 'dummydata/images/3d/'
    duplicate_2d_to_3d(src_dir, dst_dir, nslice=16)
    
    # create_dummy_z_dim('dummydata/S_Point/fake_2d.csv', 'dummydata/S_Point/fake_3d.csv')
    
    
    # train_val_test_split('dummydata/S_Point/fake_2d.csv')
    # train_val_test_split('dummydata/S_Point/fake_3d.csv')