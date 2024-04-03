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
            for suffix in ['brightness_altered', 'noisy', 'hflipped', 'vflipped', 'cropped', 'contrast_altered']:
                augmented_id = f"{numeric_id}_{suffix}"
                mapping_dict[augmented_id] = numeric_id
        return mapping_dict

    def __len__(self):
        # Count the total number of augmented images (5 times the number of original images)
        return len(self.mapping_dict)

    def __getitem__(self, idx):
        augmented_id = list(self.mapping_dict.keys())[idx]
        numeric_id = self.mapping_dict[augmented_id]

        img_path = os.path.join(self.root_dir, f"{numeric_id}_{augmented_id.split('_')[1]}.png")
        image = Image.open(img_path)

        label = self.df.loc[self.df['image_id'] == int(numeric_id), 'label'].values[0]

        return image, label


#dataset and dataloaders
training_dataset = CustomDataset(root_dir='/kaggle/working/rescaled_images', csv_file='/kaggle/input/UBC-OCEAN/train.csv')
train_DL = DataLoader(training_dataset, batch_size=32, shuffle=True)

#repeat for validation
validation_dataset = CustomDataset(root_dir='/kaggle/working/valid_images', csv_file='/kaggle/input/UBC-OCEAN/train.csv')
validation_DL = DataLoader(validation_dataset, batch_size=32, shuffle=True)
