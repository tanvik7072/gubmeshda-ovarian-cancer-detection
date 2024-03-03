# Set the paths
valid_images_path = '/kaggle/output/valid_images'

# Create the validation folder if it doesn't exist
os.makedirs(valid_images_path, exist_ok=True)

# Get the list of all images in the train folder
all_images = os.listdir(train_images_path)

# Calculate the number of images to move (25%)
num_images_to_move = int(0.25 * len(all_images))

# Randomly select the images to move
images_to_move = random.sample(all_images, num_images_to_move)

# Move the selected images to the validation folder
for image in images_to_move:
    src_path = os.path.join(train_images_path, image)
    dest_path = os.path.join(valid_images_path, image)
    shutil.move(src_path, dest_path)
