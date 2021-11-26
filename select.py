import numpy as np
# import pandas as pd
import xml.etree.ElementTree as ET
import os
from bs4 import BeautifulSoup
import enchant
import csv
import shutil




Root1 = "processed_dataset_v2/annotation"
Dest_path1 = "processed_dataset_v3/annotation"

Root2 = "processed_dataset_v2/image"
Dest_path2 = "processed_dataset_v3/image"

file = 'feature.csv'
selected_images = []
with open(file) as f:
    for row in f:
        selected_images.append(row.split(',')[0])


myList = [value for value in selected_images if value != '']


print(myList)

for name in myList:

    name = name[:-4]

    img_dir = os.path.join(Root2, name+'.jpg')
    ann_dir = os.path.join(Root1, name+'.xml')
    if (os.path.exists(img_dir)) and (os.path.exists(ann_dir)):
        print(name)
        shutil.copyfile(img_dir, os.path.join(Dest_path2, name+'.jpg'))
        shutil.copyfile(ann_dir, os.path.join(Dest_path1, name+'.xml'))

 







