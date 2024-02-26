# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 01:36:23 2024

@author: aryan
"""
from PIL import Image
import numpy as np
import random

def add_random_gaussian_noise(image_path, output_path, min_sigma=5, max_sigma=50):
    """
    Add random Gaussian noise to an image and save the result.

    Parameters:
    - image_path: The file path of the input image.
    - output_path: The file path for the output image.
    - min_sigma: Minimum standard deviation of the Gaussian noise.
    - max_sigma: Maximum standard deviation of the Gaussian noise.
    """
    # Load the image
    image = Image.open(image_path)
    image_array = np.array(image)

    # Ensure image is in float format to avoid overflow or underflow
    image_array = image_array.astype(float)

    # Generate random Gaussian noise
    sigma = random.uniform(min_sigma, max_sigma)
    noise = np.random.normal(0, sigma, image_array.shape)

    # Add the noise to the image
    noisy_image = image_array + noise

    # Ensure values remain within the valid range [0, 255]
    noisy_image = np.clip(noisy_image, 0, 255)

    # Convert back to an image
    noisy_image = noisy_image.astype(np.uint8)
    noisy_image_pil = Image.fromarray(noisy_image)

    # Save the noisy image
    noisy_image_pil.save(output_path)
    print(f"Added Gaussian noise with sigma = {sigma}. Saved to {output_path}")

# Example usage:
image_path = 'path/to/your/image.jpg'  # Update this to your image's file path
output_path = 'path/to/your/noisy_image.jpg'  # Update this to your desired output file path
add_random_gaussian_noise(image_path, output_path)
