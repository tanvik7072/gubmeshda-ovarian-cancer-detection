from PIL import Image
import os
import random

def randRange(a, b):
    #A function to generate random float values in the desired range
    return random.uniform(a, b)

def random_crop(image):
    #Cropping the image in the centre from a random margin from the borders
    margin = 1 / 3.5
    width, height = image.size
    start_x = int(randRange(0, width * margin))
    start_y = int(randRange(0, height * margin))
    end_x = int(randRange(width * (1 - margin), width))
    end_y = int(randRange(height * (1 - margin), height))

    cropped_image = image.crop((start_x, start_y, end_x, end_y))
    return cropped_image

def crop_images(folder_path):
    # Get a list of all files in the folder
    file_list = os.listdir(folder_path)

    # Iterate through each file in the folder
    for file_name in file_list:
        # Check if the file is an image
        if file_name.lower().endswith('.png')
            # Create the full path to the image file
            image_path = os.path.join(folder_path, file_name)

            # Open the image using Pillow
            original_image = Image.open(image_path)

            # Apply random cropping
            cropped_image = random_crop(original_image)

            # Create a new file name for the cropped images
            cropped_image_path = os.path.join(folder_path, f"cropped_{file_name}")

            # Save the cropped image
            cropped_image.save(cropped_image_path)

            print(f"Cropped image saved: {cropped_image_path}")
