import os
from PIL import Image
from skimage.transform import rotate


def cw_rotate_images(folder_path):
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

            # Perform clockwise rotation
            rot_cw_image = rotate(original_image, angle=90) 
            # Create a new file name for the flipped image
            cw_rotated_image_path = os.path.join(folder_path, f"cw_rotated_{file_name}")

            # Save the flipped image
            rot_cw_image.save(cw_rotated_image_path)

            print(f"Rotated image saved: {cw_rotated_image_path}")

def acw_rotate_images(folder_path):
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

            # Perform clockwise rotation
            rot_acw_image = rotate(original_image, angle=90) 
            # Create a new file name for the flipped image
            acw_rotated_image_path = os.path.join(folder_path, f"acw_rotated_{file_name}")

            # Save the flipped image
            rot_acw_image.save(acw_rotated_image_path)

            print(f"Rotated image saved: {acw_rotated_image_path}")
