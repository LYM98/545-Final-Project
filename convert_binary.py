# import numpy as np
# import nltk
# import xml.etree.ElementTree as ET
# import os
# from bs4 import BeautifulSoup
# import enchant
import csv
# import inflect



feature_path = 'feature.csv'
feature_idx_path = 'feature_index.csv'

feature_idx = []
with open(feature_idx_path) as f:
    for row in f:
        
        feature_idx.append (row.split(',')[:-1])
print(len(feature_idx[0]))

feature_idx = feature_idx[0]
f.close()

counter = 0
binary_feature = []
with open(feature_path) as f:
    for row in f:
        binary_row = [0]*len(feature_idx)
        if counter % 2 == 0:
            temp = row.split(',')
            try: 
                idx = temp.index('')
                temp = temp[:idx]
            except:
                pass
            for i in range(1,len(temp)-1):
                idx = feature_idx.index(temp[i])
                binary_row[idx] = 1
            binary_row = [temp[0]]+binary_row
            binary_feature.append(binary_row)

        counter +=1

    f.close()

    f = open('binary_feature.csv', 'w', encoding='UTF8')
    writer = csv.writer(f,delimiter=',')

    for ele in binary_feature:
        writer.writerow(ele)

    f.close()












