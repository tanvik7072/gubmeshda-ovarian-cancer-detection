# -*- coding: utf-8 -*-

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random
from PIL import Image, ImageEnhance

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

import tensorflow as tf
import cv2

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

"""# Step 1: Image Rescaling"""

# function to preprocess images
def preprocess_images(input_folder, output_folder, target_size=(224, 224)):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate through each image in the input folder
    for filename in os.listdir(input_folder):
        # Construct the input path for the current image
        input_path = os.path.join(input_folder, filename)

        # Read the image using OpenCV
        img = cv2.imread(input_path)

        # Resize the image to the target size, using a target size of 224x224
        resized_img = cv2.resize(img, target_size)


        # Construct the output path for the resized image
        output_path = os.path.join(output_folder, filename)

        # Save the resized image to the output folder
        cv2.imwrite(output_path, resized_img)

input_folder = "/kaggle/input/UBC-OCEAN/train_thumbnails"
output_folder = "/kaggle/working/rescaled_images"

# Set the target size for resizing
target_size = (224, 224)

# Preprocess images
preprocess_images(input_folder, output_folder, target_size)

"""# 2. Data Augmentation"""

def adjust_random_brightness(image_path, file_name):
    """
    Adjust the brightness of an image to a random factor between 0.5 and 1.5 and save the result.
    Parameters:
        - image_path: The file path of the input image.
    """
    original_image = Image.open(image_path)

    # Generate a random brightness factor between 0.5 and 1.5
    random_factor = random.uniform(0.5, 1.5)
    print(f"Adjusting brightness by a factor of {random_factor}")

    # Create a brightness enhancer and apply the random factor
    enhancer = ImageEnhance.Brightness(original_image)
    BA_image = enhancer.enhance(random_factor)

    # Create a new file name for the adjusted image
    file_name = os.path.splitext(file_name)[0]  # Remove file extension
    brightness_image_path = os.path.join(folder_path, f"{file_name}_brightness_altered.png")


    # Save the adjusted image
    BA_image.save(brightness_image_path)
    print(f"Brightness adjusted image saved: {brightness_image_path}")


def adjust_contrast(image_path, file_name):
    """
    Adjust the contrast of an image to a random factor between 0.5 and 1.5 and save the result.
    Parameters:
        - image_path: The file path of the input image.
    """
    original_image = Image.open(image_path)

    # Generate a random contrast factor between 0.5 and 1.5
    random_factor = random.uniform(0.5, 1.5)
    print(f"Adjusting contrast by a factor of {random_factor}")

    # Create a contrast enhancer and apply the random factor
    enhancer = ImageEnhance.Contrast(original_image)
    contrast_image = enhancer.enhance(random_factor)

    # Create a new file name for the adjusted image
    file_name = os.path.splitext(file_name)[0]  # Remove file extension
    contrast_image_path = os.path.join(folder_path, f"{file_name}_contrast_altered.png")

    # Save the adjusted image
    contrast_image.save(contrast_image_path)
    print(f"Contrast adjusted image saved: {contrast_image_path}")


def random_crop(image):
    '''
    Cropping the image in the centre from a random margin from the borders
    '''
    margin = 1 / 3.5
    width, height = image.size
    start_x = int(random.uniform(0, width * margin))
    start_y = int(random.uniform(0, height * margin))
    end_x = int(random.uniform(width * (1 - margin), width))
    end_y = int(random.uniform(height * (1 - margin), height))

    cropped_image = image.crop((start_x, start_y, end_x, end_y))
    return cropped_image


def crop_images(image_path, file_name):
    """
    Crop the given image and save the result.
    Parameters:
        - image_path: The file path of the input image.
    """
    original_image = Image.open(image_path)

    # Apply random cropping
    cropped_image = random_crop(original_image)

    # Create a new file name for the cropped image
    file_name = os.path.splitext(file_name)[0]  # Remove file extension
    cropped_image_path = os.path.join(folder_path, f"{file_name}_cropped.png")
    
    # Save the cropped image
    cropped_image.save(cropped_image_path)
    print(f"Cropped image saved: {cropped_image_path}")


def horizontal_flip_images(image_path, file_name):
    """
    Perform horizontal flipping on the given image and save the result.
    Parameters:
        - image_path: The file path of the input image.
    """
    original_image = Image.open(image_path)

    # Perform horizontal flipping
    flipped_image = original_image.transpose(Image.FLIP_LEFT_RIGHT)

    # Create a new file name for the flipped image
    file_name = os.path.splitext(file_name)[0]  # Remove file extension
    hflipped_image_path = os.path.join(folder_path, f"{file_name}_hflipped.png")

    # Save the flipped image
    flipped_image.save(hflipped_image_path)
    print(f"Horizontal flipped image saved: {hflipped_image_path}")


def vertical_flip_images(image_path, file_name):
    """
    Perform vertical flipping on the given image and save the result.
    Parameters:
        - image_path: The file path of the input image.
    """
    original_image = Image.open(image_path)

    # Perform vertical flipping
    flipped_image = original_image.transpose(Image.FLIP_TOP_BOTTOM)

    # Create a new file name for the flipped image
    file_name = os.path.splitext(file_name)[0]  # Remove file extension
    vflipped_image_path = os.path.join(folder_path, f"{file_name}_vflipped.png")

    
    # Save the flipped image
    flipped_image.save(vflipped_image_path)
    print(f"Vertical flipped image saved: {vflipped_image_path}")


def add_random_gaussian_noise(image_path, file_name):
    """
    Add random Gaussian noise to an image and save the result.

    Parameters:
    - image_path: The file path of the input image.
    - output_path: The file path for the output image.
    - min_sigma: Minimum standard deviation of the Gaussian noise.
    - max_sigma: Maximum standard deviation of the Gaussian noise.
    """


    # Open the image using Pillow
    original_image = Image.open(image_path)

    #Adding noise
    image_array = np.array(original_image)
    # Ensure image is in float format
    image_array = image_array.astype(float)

    # Generate random Gaussian noise
    sigma = random.uniform(50, 100)
    noise = np.random.normal(0, sigma, image_array.shape)

    # Add the noise to the image
    noisy_image = image_array + noise

    # Ensure values remain within the valid range [0, 255]
    noisy_image = np.clip(noisy_image, 0, 255)

    # Convert back to an image
    noisy_image = noisy_image.astype(np.uint8)
    noisy_image_pil = Image.fromarray(noisy_image)


    # Create a new file name for the adjusted image
    file_name = os.path.splitext(file_name)[0]  # Remove file extension
    noisy_path = os.path.join(folder_path, f"{file_name}_noisy.png")


    # Save the noisy image
    noisy_image_pil.save(noisy_path)
    print(f"Added Gaussian noise with sigma = {sigma}. Saved to {noisy_path}")

#Combined function

def augment(folder_path):
    """
    Combine all augmentation functions and apply them to images in the specified folder.
    Parameters:
        - folder_path: The path to the folder containing images to be augmented.
    """
    file_list = os.listdir(folder_path)

    # Iterate through each file in the folder
    for file_name in file_list:
        # Check if the file is an image
        if file_name.lower().endswith('.png'):
            # Create the full path to the image file
            image_path = os.path.join(folder_path, file_name)

            # Apply all augmentation functions
            adjust_random_brightness(image_path, file_name)
            adjust_contrast(image_path, file_name)
            crop_images(image_path, file_name)
            horizontal_flip_images(image_path, file_name)
            vertical_flip_images(image_path, file_name)
            add_random_gaussian_noise(image_path, file_name)

#Calling the function
folder_path = '/kaggle/working/rescaled_images' #after scaling, images saved here
augment(folder_path)
