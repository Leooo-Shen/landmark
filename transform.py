import random
import torchvision
import torch
from typing import Any, Tuple

class RandomRotation:
    """
    An extension of torchvision.transforms.RandomRotation for 3D data.
    Consistent rotation across slices.
    """
    def __init__(self, degree):
        self.degree = degree
        self.rotation = torchvision.transforms.RandomRotation(degrees=degree)
        
    def __call__(self, data):
        if len(data.shape) == 4:
            bs, s, h, w = data.shape
            # merge batch and slice dim, then split it back
            data = data.reshape(bs*s, 1, h, w)
            data = self.rotation(data).reshape(bs, s, h, w)
        else:
            data = self.rotation(data)
        return data
    
    
class ColorJitter:
    """
    An extension of torchvision.transforms.ColorJitter for 3D data.
    Consistent color jitter across slices.
    """
    def __init__(self, brightness, contrast, saturation=0, hue=0):
        self.color_jitter = torchvision.transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
        
    def __call__(self, data):
        if len(data.shape) == 4:
            # merge batch and slice dim, then split it back
            bs, s, h, w = data.shape
            data = data.reshape(bs*s, 1, h, w)
            data = self.color_jitter(data).reshape(bs, s, h, w)
        else:
            data = self.color_jitter(data)
        return data
    
    
class GaussianBlur:
    """
    An extension of torchvision.transforms.GaussianBlur for 3D data.
    """
    def __init__(self, kernel_size, sigma=(0.1, 2.0)):
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.gaussian_blur = torchvision.transforms.GaussianBlur(kernel_size, sigma=sigma)
        
    def __call__(self, data):
        if len(data.shape) == 4:
            # merge batch and slice dim, then split it back
            bs, s, h, w = data.shape
            data = data.reshape(bs*s, 1, h, w)
            data = self.gaussian_blur(data).reshape(bs, s, h, w)
        else:
            data = self.gaussian_blur(data)
        return data
    

