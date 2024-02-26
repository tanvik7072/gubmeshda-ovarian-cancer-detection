import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import cv2
import os 

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
        
# calling function to preprocess image 
if __name__ == "__main__":
    # Set the input and output folders
    input_folder = "/kaggle/input/UBC-OCEAN/train_thumbnails"
    output_folder = "/kaggle/working/"

    # Set the target size for resizing
    target_size = (224, 224)

    # Preprocess images
    preprocess_images(input_folder, output_folder, target_size)