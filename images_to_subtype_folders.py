# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import csv
import shutil


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))
# Define the name of the new folder
new_folder_name = "CC"

# Path for the new folder
new_folder_path = f'/kaggle/working/sorted_images/{new_folder_name}'
if not os.path.exists(new_folder_path):
    os.makedirs(new_folder_path)


df = pd.read_csv('/kaggle/input/UBC-OCEAN/train.csv')
print(df.head())

l = df['image_id'].tolist()
print(l[0])

# Create a dictionary where 'key_column' values are keys and 'value_column' values are the corresponding values
mapping_dict = {key: value for key, value in zip(df['image_id'], df['label'])}

print(mapping_dict)

# HGSC, LGSC, MC, EC, CC, other

counter = 0
for dirname, _, filenames in os.walk('/kaggle/input/UBC-OCEAN/train_thumbnails'):
    for filename in filenames:
        s=""
        for x in filename:
            if(x=='_'):
                break
            s+=x
        if(counter<15):
            source_path = f'/kaggle/input/UBC-OCEAN/train_thumbnails/{filename}'
            destination_path = f'/kaggle/working/sorted_images/{mapping_dict[int(s)]}/{filename}'
            shutil.copy(source_path, destination_path)
        counter+=1

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session