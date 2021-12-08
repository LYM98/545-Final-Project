import cv2
import numpy as np
import os
import pandas as pd
import csv
from sklearn.cluster import MiniBatchKMeans
from sklearn.neural_network import MLPClassifier
dico = []

Root = "processed_dataset_v4/image"

# count = 0
# for root, dirs, files in os.walk(Root, topdown=True):
#    for name in files:
#         print(os.path.join(root, name))
#         img = cv2.imread(os.path.join(root, name))
  

#         # convert to greyscale
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         gray = cv2.resize(gray,(500,500),interpolation = cv2.INTER_AREA)
#         sift = cv2.SIFT_create()


#         keypoints, descriptors = sift.detectAndCompute(gray, None)

#         for d in descriptors:
#             dico.append(d)
#         count +=1
#         print(count)
#         if count == 5000:
#             break



k = 1000

# kmeans = MiniBatchKMeans(n_clusters=k,batch_size=16188*3, verbose=1).fit(dico)

import pickle

# # save
# with open('model.pkl','wb') as f:
#     pickle.dump(kmeans,f)

# load
with open('model.pkl', 'rb') as f:
    kmeans = pickle.load(f)


kmeans.verbose = False
histo_list = []



f = open('sift_histo_global.csv', 'a', encoding='UTF8')
writer = csv.writer(f)


start = False
for root, dirs, files in os.walk(Root, topdown=True):
   for name in files:

        if name == '160IMG_3894.jpg':
            start = True

        if start == True:
            print(os.path.join(root, name))
            img = cv2.imread(os.path.join(root, name))
    

            # convert to greyscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray,(500,500),interpolation = cv2.INTER_AREA)
            sift = cv2.SIFT_create()


            keypoints, descriptors = sift.detectAndCompute(gray, None)

            histo = np.zeros(k)
            nkp = np.size(keypoints)

            for d in descriptors:
                idx = kmeans.predict([d])
                histo[idx] += 1/nkp # Because we need normalized histograms, I prefere to add 1/nkp directly
            writer.writerow([name]) 
            writer.writerow(histo)     
        else:
            pass   
        # histo_list.append(histo)
        # break
        



f.close()
