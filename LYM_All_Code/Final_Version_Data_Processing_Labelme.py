
import numpy as np
import nltk
import xml.etree.ElementTree as ET
import os
from bs4 import BeautifulSoup
import enchant
import csv
import inflect
import shutil



def tag_filtering(tag_store_path, image_path):
#   1）Remove all tags that contain non-English words, numerical terms, or symbols
#   2）Remove all tags with less than or equal to 2 letters
#   3）Replace plural tags with equivalent singular tags
#   4）Remove images with less than 4 tags 
#   5）Assign indices to all unique tags  

# Store a list of images that pass all these filters and their corresponding tags


    d = enchant.Dict("en_US")
    f = open(tag_store_path, 'w', encoding='UTF8')
    p_checker = inflect.engine()
    writer = csv.writer(f)
    Root = image_path
    feature_idx = ['Feature Index']


    for root, dirs, files in os.walk(Root, topdown=True):
        for name in files:
            temp = []
            temp.append(name)
            print(os.path.join(root, name))
            infile = open(os.path.join(root, name), "r",encoding="utf8")
            contents = infile.read()
            soup = BeautifulSoup(contents,"html.parser")
            titles = soup.find_all('name')

            for title in titles:
                if title.get_text():
                    if d.check(title.get_text()):
                        a = str(title.get_text())
                        if a.isalpha() and len(a)>2:
                            temp_a = p_checker.singular_noun(a)
                            if temp_a != False:
                                a = temp_a
                            ans = nltk.pos_tag([a])
                            val = ans[0][1]
                            if(val == 'NN' or val == 'NNS' or val == 'NNPS' or val == 'NNP'):
                                    
                                if a not in temp:
                                        temp.append(a)
            infile.close()
            if (len(temp)>4):
                writer.writerow(temp)
                for i in range(1,len(temp)):
                    if temp[i] not in feature_idx:
                        feature_idx.append(temp[i])
    writer.writerow(feature_idx)


    f.close()


def tag_to_binary(tag_idx_path, string_tag_path, binary_tag_store_path):

    # Convert string tags to binary form based on tag index
    
    feature_idx_path = tag_idx_path
    feature_idx = []
    with open(feature_idx_path) as f:
        for row in f:
            
            feature_idx.append (row.split(','))


    feature_idx[0][0]= 'light'
    feature_idx[0][-1] = 'contemplation'
    print(feature_idx[0])
    feature_idx = feature_idx[0]
    print(len(feature_idx))
    f.close()

    counter = 0
    binary_feature = None
    string_features = np.load(string_tag_path,allow_pickle=True)
    for ele in string_features:
        
        binary_row = np.zeros((1,len(feature_idx)))
        for i in ele:
            idx = feature_idx.index(i)
            binary_row[0,idx] = 1
               
        if counter==0:
            binary_feature = binary_row

        else:

            binary_feature = np.concatenate((binary_feature,binary_row),axis=0)

        counter +=1
        print(binary_feature.shape)
            
    np.save(binary_tag_store_path, binary_feature)



def move_rename(Root, Dest_path):

    # Uncomment to rename all files
    count = 0
    for root, dirs, files in os.walk(Root, topdown=True):
    
        for name in files:
            print(os.path.join(root, name))
            curr_name = os.path.join(root, name)
            new_name = os.path.join(root, str(count)+name)
            os.rename(curr_name, new_name)
        count += 1


    # move all files in subfolders into one folder
    for root, dirs, files in os.walk(Root, topdown=True):

        for name in files:
            print(os.path.join(root, name))
            shutil.copyfile(os.path.join(root, name), os.path.join(Dest_path, name))



