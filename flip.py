import os
from PIL import Image

def horizontal_flip_images(folder_path):
    # Get a list of all files in the folder
    file_list = os.listdir(folder_path)

    # Iterate through each file in the folder
    for file_name in file_list:
        # Check if the file is an image
        if file_name.lower().endswith('.png'):
            # Create the full path to the image file
            image_path = os.path.join(folder_path, file_name)

            # Open the image using Pillow
            original_image = Image.open(image_path)

            # Perform horizontal flipping
            flipped_image = original_image.transpose(Image.FLIP_LEFT_RIGHT)

            # Create a new file name for the flipped image
            flipped_image_path = os.path.join(folder_path, f"flipped_{file_name}")

            # Save the flipped image
            flipped_image.save(flipped_image_path)

            print(f"Flipped image saved: {flipped_image_path}")

def vertical_flip_images(folder_path):
    # Get a list of all files in the folder
    file_list = os.listdir(folder_path)

    # Iterate through each file in the folder
    for file_name in file_list:
        # Check if the file is an image
        if file_name.lower().endswith('.png'):
            # Create the full path to the image file
            image_path = os.path.join(folder_path, file_name)

            # Open the image using Pillow
            original_image = Image.open(image_path)

            # Perform horizontal flipping
            flipped_image = original_image.transpose(Image.FLIP_TOP_BOTTOM)

            # Create a new file name for the flipped image
            flipped_image_path = os.path.join(folder_path, f"flipped_{file_name}")

            # Save the flipped image
            flipped_image.save(flipped_image_path)

            print(f"Flipped image saved: {flipped_image_path}")

