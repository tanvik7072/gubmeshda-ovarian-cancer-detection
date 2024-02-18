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

# Path for the new folder
df = pd.read_csv('/kaggle/input/UBC-OCEAN/train.csv')
print(df.head())

l = df['label'].tolist()

HGSC = 0
LGSC = 0
MC = 0
EC = 0
CC = 0
other = 0

for x in l:
    if(x=="HGSC"):
        HGSC=HGSC+1
    elif(x=="LGSC"):
        LGSC+=1
    elif(x=="MC"):
        MC+=1
    elif(x=="EC"):
        EC+=1
    elif(x=="CC"):
        CC+=1
    elif(x=="other"):
        other+=1
    
print(f"HGSC: {HGSC}")
print(f"LGSC: {LGSC}")
print(f"MC: {MC}")
print(f"EC: {EC}")
print(f"CC: {CC}")
print(f"Other: {other}")
# Create a dictionary where 'key_column' values are keys and 'value_column' values are the corresponding values
mapping_dict = {key: value for key, value in zip(df['image_id'], df['label'])}

# print(mapping_dict)

# HGSC, LGSC, MC, EC, CC, other


# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
