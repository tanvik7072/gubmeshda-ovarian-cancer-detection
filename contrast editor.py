# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 01:35:05 2024

@author: aryan
"""

from PIL import Image, ImageEnhance
import random

def adjust_random_contrast(image_path, output_path, min_factor=0.5, max_factor=1.5):
    """
    Adjust the contrast of an image by a random factor and save the result.
   
    Parameters:
    - image_path: The file path of the input image.
    - output_path: The file path for the output image.
    - min_factor: Minimum contrast factor (less than 1 reduces contrast, 1 is original contrast).
    - max_factor: Maximum contrast factor (greater than 1 increases contrast).
    """
    # Load the image
    image = Image.open(image_path)
   
    # Generate a random contrast factor
    factor = random.uniform(min_factor, max_factor)
    print(f"Applying contrast factor: {factor}")
   
    # Apply the contrast enhancement
    enhancer = ImageEnhance.Contrast(image)
    enhanced_image = enhancer.enhance(factor)
   
    # Save the enhanced image
    enhanced_image.save(output_path)
    print(f"Enhanced image saved to {output_path}")

# Example usage:
image_path = 'path/to/your/image.jpg' # Update this to your image's file path
output_path = 'path/to/your/contrast_image.jpg' # Update this to your desired output file path
adjust_random_contrast(image_path, output_path)