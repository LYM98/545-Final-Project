import numpy as np
import nltk
import xml.etree.ElementTree as ET
import os
from bs4 import BeautifulSoup
import enchant
import csv
import inflect


d = enchant.Dict("en_US")
f = open('feature.csv', 'w', encoding='UTF8')
p_checker = inflect.engine()
writer = csv.writer(f)
Root = "processed_dataset_v1/annotation"
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







        #             if (len(temp)<11):
        #     print('False')
        #     # img_dir = os.path.join(Root_img, name[:-4]+'.jpg')
        #     # print(img_dir)
        #     # if os.path.exists(img_dir):
        #     #     os.remove(img_dir)
        #     # if os.path.exists(os.path.join(root, name)):
        #     #     os.remove(os.path.join(root, name))
        #     # print('Removed')

        # else:
        #     print('True')
        #     writer.writerow(temp)
        







