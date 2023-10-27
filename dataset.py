import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import pytorch_lightning as pl
import nibabel as nib

def load_coordinates(path):
    # read points coordinates from txt file
    f = open(path, 'r')
    x_start_text = 'X:'
    x_end_text = ','
    y_start_text = 'Y:'
    y_end_text = ';'
    all_points = []
    num_line = 0

    for line in f:
        num_line += 1
        if num_line > 2:
            x_coor = line[line.find(x_start_text) + len(x_start_text): line.find(x_end_text)]
            y_coor = line[line.find(y_start_text) + len(y_start_text): line.find(y_end_text)]
            all_points.append([int(x_coor), int(y_coor)])
            
    return all_points


class LandmarkDataset(Dataset):
    """
    Dataset for landmark detection.
    Read data in 2D or 3D, create ROI, and normalize the coordinate.
    Return the image and the normalized coordinate.
    """
    
    def __init__(self, dimension, stage, name_point='S_Point', size_image=520, transform=None):
        assert stage in ['train', 'val', 'test'], 'stage must be one of train, val, test'
        
        self.transform = transform
        # read the mean XY from prepration, to create ROI later
        text_mean_x_y_dir = 'data/Preparation/' + name_point + '/Mean_X_Y.txt'
        f = open(text_mean_x_y_dir, 'r')
        num_line = 0
        x_start_text = 'Mean_X:'
        x_end_text = ','
        y_start_text = 'Mean_Y:'
        y_end_text = ';'

        for line in f:
            num_line += 1
            x_coor = line[line.find(x_start_text) + len(x_start_text): line.find(x_end_text)]
            y_coor = line[line.find(y_start_text) + len(y_start_text): line.find(y_end_text)]
            mean_x = int(x_coor)
            mean_y = int(y_coor)

        self.input_images, self.output = self.read_data(dimension, name_point, mean_x, mean_y, size_image, stage)
        
        
    def __len__(self):
        return len(self.input_images)
    
    def __getitem__(self, index):
        img = self.input_images[index]
        if self.transform is not None:
            img = self.transform(img)
        target = self.output[index]
        target = torch.tensor(target, dtype=torch.float32)    
        return img, target
    
    def read_data(self, dimension, name_point, mean_x, mean_y, size_image, stage):
        "Read the data from the folder, create ROI, and normalize the coordinate"
        
        if dimension == '3d':
            recognized_text_dir = 'data/3D/' + name_point + '/Coordinates_' + stage
            recognized_image_dir = 'data/3D/' + name_point + '/Images_without_' + stage
        elif dimension == '2d':
            recognized_text_dir = 'data/2D/' + name_point + '/Coordinates_' + stage
            recognized_image_dir = 'data/2D/' + name_point + '/Images_without_' + stage
            
        half_size = int(size_image/2)
        roi_x_start = mean_x - half_size
        roi_x_end = mean_x + half_size
        roi_y_start = mean_y - half_size
        roi_y_end = mean_y + half_size

        # check number of files
        num_images = len([name for name in os.listdir(recognized_image_dir) if os.path.isfile(os.path.join(recognized_image_dir, name))])
        num_texts = len([name for name in os.listdir(recognized_text_dir) if os.path.isfile(os.path.join(recognized_text_dir, name))])
        assert num_images == num_texts, 'Problem occurs. The number of text files does not ' \
                                                'equal the number of images. There must be one text file for each image!'

        # The arrays with the data needed for training
        input_images = []
        output = []
        print(f'{stage} images are being preprocessed...')
        
        # crop the image with ROI, also load the real target points and normalize the coordinate with the size of ROI
        num_img = 0
        for subdir, dirs, files in os.walk(recognized_image_dir):
            for file in files:
                curr_file_name = file.split('.')[0]
                
                if dimension == '3d':
                    name_image = recognized_image_dir+'/'+curr_file_name+'.nii.gz'
                    image = nib.load(name_image)
                    image = np.array(image.dataobj).transpose(2, 1, 0)
                    roi = image[roi_y_start:roi_y_end, roi_x_start:roi_x_end, :]
                
                elif dimension == '2d':
                    name_image = recognized_image_dir+'/'+curr_file_name+'.png'
                    image = cv2.imread(name_image, 0)
                    roi = image[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
                    
                coord_path = recognized_text_dir+'/'+curr_file_name+'.txt'
                all_points = load_coordinates(coord_path)
                print('loading coord: ', coord_path)
                local_x = all_points[0][0] - roi_x_start
                local_y = all_points[0][1] - roi_y_start
                input_images.append(roi)
                output.append([local_x/size_image, local_y/size_image]) # normalize the coordinate for NN output
                num_img += 1
        return input_images, output
    
# test
if __name__ == '__main__':
    
    # test dataset
    dataset = LandmarkDataset(dimension='3d', stage='train')
    print(len(dataset))
    img, coord = dataset[0]
    print(img.shape, coord.shape) 
    
    # import matplotlib.pyplot as plt
    
    # if len(img.shape) == 2:
    #     plt.imshow(img, cmap='gray')
    #     plt.scatter(coord[0]*520, coord[1]*520, c='r', s=10)
    #     plt.show()
        
    # elif len(img.shape) == 3:
    #     # create a figure with subplots, visualiaze each slice of img on the first dim
    #     figure, axes = plt.subplots(nrows=1, ncols=img.shape[0], figsize=(20, 5))
    #     for i in range(img.shape[0]):
    #         axes[i].imshow(img[i], cmap='gray')
    #         axes[i].scatter(coord[0]*520, coord[1]*520, c='r', s=10)
    #     plt.show()
                
