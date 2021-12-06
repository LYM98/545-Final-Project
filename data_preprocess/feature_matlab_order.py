import numpy as np
import nltk
import xml.etree.ElementTree as ET
import os
from bs4 import BeautifulSoup
import enchant
import csv
import inflect
nltk.download('averaged_perceptron_tagger')

d = enchant.Dict("en_US")
f = open('matlab_order_features/feature.csv', 'w', encoding='UTF8')
p_checker = inflect.engine()
writer = csv.writer(f)
root = "processed_dataset_v4/annotation"
feature_idx = ['Feature Index']




path = 'Extracted Features/sift_matlab_order/same_number_order_order.npy'
order = np.load(path)
print(order.shape )
count = 0

alllllfeatures = []
for ele in order:
    name = ele[:-4]+'.xml'


    temp = []
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
    if (len(temp)>3):
        writer.writerow(temp)
        alllllfeatures.append(temp)
        for i in range(len(temp)):
            if temp[i] not in feature_idx:
                feature_idx.append(temp[i])
        count+=1
    print(count)


  
writer.writerow(feature_idx)


f.close()
temp_n = np.asarray(alllllfeatures, dtype=object)
np.save('matlab_order_features/feature_ordered.npy', temp_n)


data = np.load('matlab_order_features/feature_ordered.npy', allow_pickle=True)
print(data[0])



