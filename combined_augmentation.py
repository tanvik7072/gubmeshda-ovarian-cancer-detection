import os
import random
from PIL import Image, ImageEnhance

def adjust_random_brightness(image_path):
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
    brightness_image_path = os.path.join(folder_path, f"{file_name}_brightness")


    # Save the adjusted image
    BA_image.save(brightness_image_path)
    print(f"Brightness adjusted image saved: {brightness_image_path}")


def adjust_contrast(image_path):
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
    contrast_image_path = os.path.join(folder_path, f"{file_name}_contrast")

    # Save the adjusted image
    contrast_image.save(contrast_image_path)
    print(f"Contrast adjusted image saved: {contrast_image_path}")


def random_crop(image):
    '''
    Cropping the image in the center from a random margin from the borders
    '''
    margin = 1 / 3.5
    width, height = image.size
    start_x = int(random.uniform(0, width * margin))
    start_y = int(random.uniform(0, height * margin))
    end_x = int(random.uniform(width * (1 - margin), width))
    end_y = int(random.uniform(height * (1 - margin), height))

    cropped_image = image.crop((start_x, start_y, end_x, end_y))
    return cropped_image


def crop_images(image_path):
    """
    Crop the given image and save the result.
    Parameters:
        - image_path: The file path of the input image.
    """
    original_image = Image.open(image_path)

    # Apply random cropping
    cropped_image = random_crop(original_image)

    # Create a new file name for the cropped image
    cropped_image_path = os.path.join(folder_path, f"{file_name}_cropped")

    # Save the cropped image
    cropped_image.save(cropped_image_path)
    print(f"Cropped image saved: {cropped_image_path}")


def horizontal_flip_images(image_path):
    """
    Perform horizontal flipping on the given image and save the result.
    Parameters:
        - image_path: The file path of the input image.
    """
    original_image = Image.open(image_path)

    # Perform horizontal flipping
    flipped_image = original_image.transpose(Image.FLIP_LEFT_RIGHT)

    # Create a new file name for the flipped image
    flipped_image_path = os.path.join(folder_path, f"{file_name}_hflipped")

    # Save the flipped image
    flipped_image.save(flipped_image_path)
    print(f"Horizontal flipped image saved: {flipped_image_path}")


def vertical_flip_images(image_path):
    """
    Perform vertical flipping on the given image and save the result.
    Parameters:
        - image_path: The file path of the input image.
    """
    original_image = Image.open(image_path)

    # Perform vertical flipping
    flipped_image = original_image.transpose(Image.FLIP_TOP_BOTTOM)

    # Create a new file name for the flipped image
    flipped_image_path = os.path.join(folder_path, f"{file_name}_vflipped")
    # Save the flipped image
    flipped_image.save(flipped_image_path)
    print(f"Vertical flipped image saved: {flipped_image_path}")


def add_random_gaussian_noise(folder_path):
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
    noisy_path = os.path.join(folder_path, f"{file_name}_noisy")

    # Save the noisy image
    noisy_image_pil.save(noisy_path)
    print(f"Added Gaussian noise with sigma = {sigma}. Saved to {noisy_path}")


##USAGE
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
            adjust_random_brightness(image_path)
            adjust_contrast(image_path)
            crop_images(image_path)
            horizontal_flip_images(image_path)
            vertical_flip_images(image_path)
            add_random_gaussian_noise(image_path)

#Calling the function:
folder_path = '/kaggle/output/train_images' #after scaling, images saved here
augment(folder_path)
