import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
import os
from bs4 import BeautifulSoup
import enchant
import csv
import shutil



# Root = "labelme/Images"
# Dest_path = "dataset/image"
# Root = "labelme/Annotations"
# Dest_path = "dataset/annotation"

Root = "labelme/temp"
Dest_path = "dataset/image"




# rename all files
# count = 0
# for root, dirs, files in os.walk(Root, topdown=True):
 
#     for name in files:
#         print(os.path.join(root, name))
#         curr_name = os.path.join(root, name)
#         new_name = os.path.join(root, str(count)+name)
#         os.rename(curr_name, new_name)
#     count += 1


# move all files in subfolders into one folder
for root, dirs, files in os.walk(Root, topdown=True):

    for name in files:
        print(os.path.join(root, name))
        shutil.copyfile(os.path.join(root, name), os.path.join(Dest_path, name))

    










      
       
      

        

        
