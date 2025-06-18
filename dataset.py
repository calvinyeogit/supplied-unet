# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 14:55:38 2019

@author: mbiww
"""

import os

import numpy as np
import math
import imgaug
from imgaug import augmenters as iaa

# from albumentations import (
#     RandomCrop, HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
#     Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, ElasticTransform,
#     IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur,
#     IAASharpen, RandomBrightnessContrast, Flip, OneOf, Compose
# )



# from albumentations import RandomRotate90, Flip, Transpose, OneOf, Blur, MotionBlur, MedianBlur, ShiftScaleRotate, OpticalDistortion, ElasticTransform, GridDistortion, CLAHE, RandomBrightnessContrast, RandomCrop, Compose
# Deepseek explained that newer versions of Albumentations have renamed some augmentation functions.

# from albumentations import (
#     RandomRotate90, HorizontalFlip, Transpose, OneOf, Blur, MotionBlur, MedianBlur,
#     ShiftScaleRotate, OpticalDistortion, ElasticTransform, GridDistortion, CLAHE,
#     RandomBrightnessContrast, RandomCrop, Compose
# )

import albumentations as A


import matplotlib.pyplot as plt

from Unet_Config import Unet_Config

import glob;
import skimage
import skimage.io as skio


class dataset(Unet_Config):
    """
    Base dataset class
    """
    def __init__(self):
        self.classes = []
        self.train_images=[]
        self.train_masks=[]
        super().__init__()
    
    def get_class_id(self, class_name):
        """
        Returns the class id
        Adds class to list if not in list of classes and returns a new class id
        """
        if len(self.classes) == 0:
            self.classes.append({"class": class_name, "id": 0})
            return 0
        
        for class_info in self.classes:
            # if class exist, return class id
            if class_info["class"] == class_name:
                return class_info["id"]
   
        self.classes.append({"class": class_name, "id": len(self.classes)-1})
        return len(self.classes)-1
    
    
    def load_dataset(self, image_dir=None, mask_dir=None):
        """
        Load and match images with masks based on alphabetical order
        """
        print("\nLoading and matching images with masks...")
        print(f"Image directory: {image_dir}")
        print(f"Mask directory: {mask_dir}")
        
        # Load and sort files alphabetically
        image_list = sorted(glob.glob(os.path.join(image_dir,'*.tif')))
        mask_list = sorted(glob.glob(os.path.join(mask_dir,'*.tif')))
        
        print(f"\nFound {len(image_list)} images and {len(mask_list)} masks")
        
        if len(image_list) != len(mask_list):
            raise ValueError(f"Number of images ({len(image_list)}) does not match number of masks ({len(mask_list)})")
        
        # Load images and masks in sorted order
        images = []
        masks = []
        print("\nLoading pairs:")
        for img_path, mask_path in zip(image_list, mask_list):
            img = skio.imread(img_path)
            mask = skio.imread(mask_path)
            
            # Handle resizing if needed
            if (img.shape[0] < self.tile_size[0] & img.shape[0] < self.tile_size[0]):
                [img, shape] = self.pad_image(img, self.tile_size)
            if (mask.shape[0] < self.tile_size[0] & mask.shape[0] < self.tile_size[0]):
                [mask, shape] = self.pad_image(mask, self.tile_size)
            
            images.append(img)
            masks.append(mask)
            print(f"✓ Loaded: {os.path.basename(img_path)} -> {os.path.basename(mask_path)}")
        
        print(f"\nSuccessfully loaded {len(images)} image-mask pairs")
        return images, masks
            

        
      
    def reshape_image(self, image):
        """
        reshape images to correct dimensions for unet
        """
        h, w = image.shape[:2]
        image = np.reshape(image, (h, w, -1))
        return image
    
    
    
    def pad_image(self, image, image_size =(1024,1024)):
        """
        Pads image in order to generate images that are a factor of 2**n
        """
        h, w = image.shape[:2]
        
        top_pad = max( (image_size[0] - h) // 2,0)
        bottom_pad = max(image_size[0] - h - top_pad,0)
            
        left_pad = max((image_size[1] - w) // 2,0)
        right_pad =max( image_size[1] - w - left_pad,0)

        padding = ((top_pad, bottom_pad), (left_pad, right_pad))
        image = np.pad(image, padding, mode='constant', constant_values=0)
        
        return image, padding
    
    def remove_pad_image(self, image, padding = None):
        if padding is None:
            return image
        
        h, w = image.shape[:2]
        
        return image[padding[0][0]:h-padding[0][1], padding[1][0]:w-padding[1][1]]
    
    #######
    # Tiling functions
    #######
    def tile_image(self, image, tile_size):
        """
        Converts an image into a list of tiled images with tile_size
        if dimension smaller than tile_size, return padded images
        """
        image_height, image_width = image.shape[:2]
        tile_height = tile_size[0] 
        tile_width = tile_size[1]
        
        #if image_height <= tile_height and image_width <= tile_width:
         #   return image
        
        num_rows = math.ceil(image_height/tile_height)
        num_cols = math.ceil(image_width/tile_width)
        num_tiles = num_rows*num_cols
        
        # pad image to fit tile size
        image, padding = self.pad_image(image, image_size = (tile_height*num_rows , tile_width*num_cols))
        
        tile_image_list = []
        
        for tile_no in range(num_tiles):
            tile_x_start = (tile_no % num_rows) * tile_width
            tile_x_end = tile_x_start + tile_size[1]
            
            tile_y_start = (tile_no // num_rows) * tile_height
            tile_y_end = tile_y_start + tile_size[0]
            
            tile_image = image[tile_x_start:tile_x_end, tile_y_start: tile_y_end]
            
            # ensure input into unet is of correct shape
            tile_image = self.reshape_image(tile_image)
            
            tile_image_list.append(tile_image)
            
        return tile_image_list, num_rows, num_cols, padding
    
    
    def untile_image(self, tile_list, tile_size, num_rows, num_cols, padding = None): 
       
        
        if num_rows == 1 and num_cols == 1:
            image = tile_list[0]
            
            if padding is not None:
                image = self.remove_pad_image(image, padding = padding)
                
            return image
            
        
        for col in range(num_cols):
            for row in range(num_rows):
                tile_image = tile_list[num_rows*col + row][:,:,0]
                if row == 0:
                    image_col = np.array(tile_image)
                else:
                    image_col = np.vstack((image_col, tile_image))
            
            if col == 0:
                image = image_col
            else:
                image = np.hstack((image, image_col))
        
        
        if padding is not None:
            image = self.remove_pad_image(image, padding = padding)
            
        return image
    
    
    
    
    
    def augment_images(self):
        augmentor = self.augmentations(p=1.0)
        
        # increase number of images
        self.num_augmented_images=100
        self.aug_images = self.train_images*self.num_augmented_images
        self.aug_masks = self.train_masks*self.num_augmented_images
        for i in range(len(self.aug_images)):
            data = {"image": self.aug_images[i], 
                    "mask": self.aug_masks[i]}
            augmented = augmentor(**data)
            self.aug_images[i] = self.reshape_image(np.asarray(augmented["image"]))
            aa=self.aug_images[i]
            print(aa.shape)
            self.aug_masks[i] = self.reshape_image(np.ndarray.astype(augmented["mask"], bool))

    
    # Deepseek suggested replacing this entire method
    # def augmentations(self, p=None):
    #     augmentation_list = []
        
    #     # RandomRotate
    #     if self.random_rotate:
    #         augmentation_list.append(RandomRotate90(p=self.random_rotate_p))
        
    #     # Flip
    #     if self.flip:
    #         augmentation_list.append(HorizontalFlip)
        
    #     # Transpose
    #     if self.transpose:
    #         augmentation_list.append(Transpose())
        
    #     # Blur Group
    #     if self.blur_group:
    #         blur_augmentation = []
    #         if self.motion_blur:
    #             blur_augmentation.append(MotionBlur(p=self.motion_blur_p))
    #         if self.median_blur:
    #             blur_augmentation.append(MedianBlur(blur_limit=self.median_blur_limit, p=self.median_blur_p))
    #         if self.blur:
    #             blur_augmentation.append(Blur(blur_limit=self.blur_limit, p=self.blur_p))
    #         augmentation_list.append(OneOf(blur_augmentation, p=self.blur_group_p))
        
    #     # ShiftScaleRotate
    #     if self.shift_scale_rotate:
    #         augmentation_list.append(ShiftScaleRotate(shift_limit=self.shift_limit,
    #                                                   scale_limit=self.scale_limit,
    #                                                   rotate_limit=self.rotate_limit,
    #                                                   p=self.shift_scale_rotate_p))
        
    #     # Distortion Group
    #     if self.distortion_group:
    #         distortion_augmentation = []
    #         if self.optical_distortion:
    #             distortion_augmentation.append(OpticalDistortion(p=self.optical_distortion_p))
    #         if self.elastic_transform:
    #             distortion_augmentation.append(ElasticTransform(p=self.elastic_transform_p))
    #         if self.grid_distortion:
    #             distortion_augmentation.append(GridDistortion(p=self.grid_distortion_p))
            
    #         augmentation_list.append(OneOf(distortion_augmentation, p=self.distortion_group_p))
        
    #     # Brightness / Contrast Group
    #     if self.brightness_contrast_group:
    #         contrast_augmentation = []
    #         if self.clahe:
    #             contrast_augmentation.append(CLAHE())
    #         if self.random_brightness_contrast:
    #             contrast_augmentation.append(RandomBrightnessContrast())
    #         augmentation_list.append(OneOf(contrast_augmentation, p=self.brightness_contrast_group_p))
        
    #     # RandomCrop
    #     augmentation_list.append(RandomCrop(512, 512, always_apply=True))  # Fixed size for crop
        
    #     # Return the complete augmentation pipeline
    #     return Compose(augmentation_list, p=p)
    
    def augmentations(self, p=None):
        augmentation_list = []
        
        # RandomRotate
        if self.random_rotate:
            augmentation_list.append(A.RandomRotate90(p=self.random_rotate_p))
        
        # Flip
        if self.flip:
            augmentation_list.append(A.HorizontalFlip())
        
        # Transpose
        if self.transpose:
            augmentation_list.append(A.Transpose())
        
        # Blur Group
        if self.blur_group:
            blur_augmentation = []
            if self.motion_blur:
                blur_augmentation.append(A.MotionBlur(p=self.motion_blur_p))
            if self.median_blur:
                blur_augmentation.append(A.MedianBlur(blur_limit=self.median_blur_limit, p=self.median_blur_p))
            if self.blur:
                blur_augmentation.append(A.Blur(blur_limit=self.blur_limit, p=self.blur_p))
            augmentation_list.append(A.OneOf(blur_augmentation, p=self.blur_group_p))
        
        # ShiftScaleRotate
        if self.shift_scale_rotate:
            augmentation_list.append(A.ShiftScaleRotate(
                shift_limit=self.shift_limit,
                scale_limit=self.scale_limit,
                rotate_limit=self.rotate_limit,
                p=self.shift_scale_rotate_p
            ))
        
        # Distortion Group
        if self.distortion_group:
            distortion_augmentation = []
            if self.optical_distortion:
                distortion_augmentation.append(A.OpticalDistortion(p=self.optical_distortion_p))
            if self.elastic_transform:
                distortion_augmentation.append(A.ElasticTransform(p=self.elastic_transform_p))
            if self.grid_distortion:
                distortion_augmentation.append(A.GridDistortion(p=self.grid_distortion_p))
            
            augmentation_list.append(A.OneOf(distortion_augmentation, p=self.distortion_group_p))
        
        # Brightness / Contrast Group
        if self.brightness_contrast_group:
            contrast_augmentation = []
            if self.clahe:
                contrast_augmentation.append(A.CLAHE())
            if self.random_brightness_contrast:
                contrast_augmentation.append(A.RandomBrightnessContrast())
            augmentation_list.append(A.OneOf(contrast_augmentation, p=self.brightness_contrast_group_p))
        
        # RandomCrop
        augmentation_list.append(A.RandomCrop(512, 512, always_apply=True))
        
        # Return the complete augmentation pipeline
        return A.Compose(augmentation_list)
        
 
        
    def verify_image_mask_pairs(self, images, masks):
        """
        Verify that image-mask pairs are valid and correctly matched.
        Returns True if all checks pass, False otherwise.
        """
        if len(images) == 0 or len(masks) == 0:
            print("Error: No images or masks to verify!")
            return False
        
        if len(images) != len(masks):
            print(f"Error: Number of images ({len(images)}) does not match number of masks ({len(masks)})")
            return False
        
        all_valid = True
        print("\nVerifying image-mask pairs...")
        
        for idx, (img, mask) in enumerate(zip(images, masks)):
            print(f"\nChecking pair {idx + 1}/{len(images)}:")
            
            # Check dimensions match
            if img.shape[:2] != mask.shape[:2]:
                print(f"✗ Dimensions mismatch:")
                print(f"  Image shape: {img.shape}")
                print(f"  Mask shape: {mask.shape}")
                all_valid = False
            else:
                print(f"✓ Dimensions match: {img.shape[:2]}")
            
            # Check mask is binary (0 or 255)
            unique_values = np.unique(mask)
            if not np.array_equal(unique_values, np.array([0, 255])) and not np.array_equal(unique_values, np.array([0])) and not np.array_equal(unique_values, np.array([255])):
                print(f"✗ Mask is not binary. Found values: {unique_values}")
                all_valid = False
            else:
                print(f"✓ Mask is binary: values {unique_values}")
            
            # Check image type and range
            if img.dtype != np.uint8:
                print(f"⚠️ Warning: Image is not uint8 type. Found: {img.dtype}")
            else:
                print(f"✓ Image type is uint8")
            
            if mask.dtype != np.uint8:
                print(f"⚠️ Warning: Mask is not uint8 type. Found: {mask.dtype}")
            else:
                print(f"✓ Mask type is uint8")
        
        if all_valid:
            print("\n✓ All image-mask pairs verified successfully!")
        else:
            print("\n✗ Some pairs failed verification. See errors above.")
        
        return all_valid
        
    