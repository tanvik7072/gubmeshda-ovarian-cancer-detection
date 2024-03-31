# -*- coding: utf-8 -*-

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random, shutil
from PIL import Image, ImageEnhance
from torch.utils.data import DataLoader

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
    brightness_image_path = os.path.join(folder_path, f"{file_name}_brightness.png")


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
    contrast_image_path = os.path.join(folder_path, f"{file_name}_contrast.png")

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

# Set the paths
valid_images_path = '/kaggle/working/valid_images'

# Create the validation folder if it doesn't exist
os.makedirs(valid_images_path, exist_ok=True)

# Get the list of all images in the train folder
all_images = os.listdir(folder_path)

# Calculate the number of images to move (25%)
num_images_to_move = int(0.25 * len(all_images))

# Randomly select the images to move
images_to_move = random.sample(all_images, num_images_to_move)

# Move the selected images to the validation folder
for image in images_to_move:
    src_path = os.path.join(folder_path, image)
    dest_path = os.path.join(valid_images_path, image)
    shutil.move(src_path, dest_path)

#After scaling and augmenting the data, we need to make sure the new images have their original unique numerical id + "_flipped"
# The following class makes sure that pytorch can process the dataset during training, including accessing labels

class CustomDataset():
    def __init__(self, root_dir, csv_file):
        self.root_dir = root_dir
        self.df = pd.read_csv(csv_file)
        self.mapping_dict = self.create_mapping_dict()

    def create_mapping_dict(self):
        mapping_dict = {}
        for idx in range(len(self.df)):
            numeric_id = str(self.df.iloc[idx, 0])  # Assuming the numeric ID is in the first column
            for suffix in ['blurred', 'noisy', 'hflipped', 'vflipped', 'cropped']:
                augmented_id = f"{numeric_id}_{suffix}"
                mapping_dict[augmented_id] = numeric_id
        return mapping_dict

    def __len__(self):
        # Count the total number of augmented images (5 times the number of original images)
        return len(self.mapping_dict)

    def __getitem__(self, idx):
        numeric_id = str(self.df.iloc[idx, 0])
        augmented_id = list(self.mapping_dict.keys())[idx]
        

        img_path = os.path.join(self.root_dir, f"{numeric_id}_{suffix}.png")
        image = Image.open(img_path)

        label = self.df.loc[self.df['numeric_id'] == int(numeric_id), 'label'].values[0]

        return image, label


#dataset and dataloaders
training_dataset = CustomDataset(csv_file='/kaggle/input/UBC-OCEAN/train.csv', root_dir='/kaggle/input/UBC-OCEAN/train_images')
train_DL = DataLoader(training_dataset, batch_size=32, shuffle=True)

#repeat for validation
validation_dataset = CustomDataset(csv_file='/kaggle/input/UBC-OCEAN/train.csv', root_dir='/kaggle/input/UBC-OCEAN/valid_images')
validation_DL = DataLoader(validation_dataset, batch_size=32, shuffle=True)


# TODO: Build and train your network
model = models.vgg16(pretrained = True) #Loading pre-trained network

#Freeze parameters to avoid backpropagation
for param in model.parameters():
    param.requires_grad = False


model = model.to('cuda')

#Defining new untrained feed-forward network
classifier = nn.Sequential(nn.Linear(25088,4096),
                          nn.ReLU(), #activation function - ReLU is effective, computationally inexpensive
                                     #and removes the vanishing gradient problem
                          nn.Dropout(0.2),
                          nn.Linear(4096, 256),
                          nn.ReLU(),
                          nn.Dropout(0.2),
                          nn.Linear(256, 64),
                          nn.Dropout(0.2), #Removed 20% of data each time, good place to start
                          nn.Linear(64, 6),  #Must be 6 because 6 = number of classes
                          nn.LogSoftmax(dim=1))


classifier = classifier.to('cuda')
model.classifier = classifier
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr = 0.01)

#Training
epochs = 5 #maybe change to 10 later on? depending on output
train_loss = 0

for epoch in range(epochs):
    #model.train()
    for inputs, labels in train_DL:
      model.train()
      inputs, labels = inputs.to('cuda'), labels.to('cuda')
      optimizer.zero_grad()
      outputs = model.forward(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()

      train_loss += loss.item()


    model.eval()
    with torch.inference_mode():
      validation_loss = 0
      accuracy = 0
      for inputs, labels in validation_DL:
          inputs, labels = inputs.to('cuda'), labels.to('cuda')
          outputs = model.forward(inputs)
          running_valid_loss = criterion(outputs, labels).item()
          validation_loss += running_valid_loss

          ps = torch.exp(outputs)
          top_p, top_class = ps.topk(1, dim = 1)
          equals = top_class == labels.view(*top_class.shape)
          accuracy += torch.mean(equals.type(torch.FloatTensor))

    print(f"Epoch {epoch+1}/{epochs}...",
          f"Train loss: {train_loss/len(train_DL):.3f}..."
        f"Validation loss: {validation_loss/len(validation_DL):.3f}..."
        f"Validation accuracy: {accuracy:.3f}...")
    train_loss = 0


