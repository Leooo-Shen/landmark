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


def duplicate(src_dir, dst_dir, nslice=10):
    """Duplicate the 2D image to 3D volume"""
    
    print('Duplicate the 2D image to 3D volume')
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
        

def create_dummy_images_2d(src_dir, dst_dir, num_images=100):
    """Create dummy images by linear interpolating two images together"""
    print('Create dummy images by linear interpolating two images together')
    
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    files = os.listdir(src_dir)
    # only keep files ending with .png
    files = [file for file in files if file.endswith('.png')]
    
    for n in tqdm(range(6, num_images+1)):
        # random pick 2 images and load
        random.shuffle(files)
        files = files[:2]
    
        image1 = cv2.imread(os.path.join(src_dir, files[0]), cv2.COLOR_BGR2GRAY)
        image2 = cv2.imread(os.path.join(src_dir, files[1]), cv2.COLOR_BGR2GRAY)

        # Ensure both images have the same dimensions
        if image1.shape != image2.shape:
            raise ValueError("Both images must have the same dimensions")

        # random choose alpha between 0-0.3 and 0.7 to 1
        alpha = random.uniform(0, 0.3)
        interpolated_image = cv2.addWeighted(image1, 1 - alpha, image2, alpha, 0)
        
        save_name = 'image' + str(n) + '.png'
        cv2.imwrite(os.path.join(dst_dir, save_name), interpolated_image)
    

def create_dummy_points_3d(src_path, dst_path='dummydata/fake.csv', num_samples=100):
    """Create dummy points by adding noise to the original points"""
    data = pd.read_csv(src_path)
    augmented_data = []

    for i in range(6, num_samples + 1):
        # Randomly select one of the existing data points
        random_index = np.random.randint(0, len(data))
        selected_data_point = data.iloc[random_index]

        # Add noise to the selected data point (adjust noise level as needed)
        noise = np.random.normal(0, 1, size=3).round(2)
        augmented_data_point = selected_data_point[['X', 'Y', 'Z']] + noise
        
        new_data_point = {
            'name': f'image{i}',
            'X': augmented_data_point['X'],
            'Y': augmented_data_point['Y'],
            'Z': augmented_data_point['Z']
        }

        augmented_data.append(new_data_point)

    augmented_df = pd.DataFrame(augmented_data)
    augmented_df.to_csv(dst_path, index=False)


def train_val_split(path):
    """Split the data into train and validation set"""
    data = pd.read_csv(path)
    train = data.sample(frac=0.8, random_state=200)
    val = data.drop(train.index)
    root_path = '/'.join(path.split('/')[:-1])
    
    train.to_csv(os.path.join(root_path, 'train.csv'), index=False)
    val.to_csv(os.path.join(root_path, 'val.csv'), index=False)


if __name__ == '__main__':
    # src_dir = 'data/2D/S_Point/Images_without'
    # dst_dir = 'dummydata/images/2D/'
    # create_dummy_images_2d(src_dir, dst_dir, num_images=100)
    
    src_dir = 'dummydata/images/2d/'
    dst_dir = 'dummydata/images/3d/'
    duplicate(src_dir, dst_dir, nslice=10)
    
    # create_dummy_points_3d(src_path='dummydata/S_Point/test.csv', dst_path='dummydata/S_Point/fake.csv', num_samples=100)
    # train_val_split('dummydata/S_Point/fake.csv')