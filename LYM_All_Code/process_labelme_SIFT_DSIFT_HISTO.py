import csv
import numpy as np
import pandas as pd
import os
import cv2
import pickle
from sklearn.cluster import MiniBatchKMeans

def labelme_sift(data_list, data_path,K):

    # load the labelme images' pathes 
    
    with open(data_list) as f:
        data_list = f.readlines()

    f.close()

    for i in range(len(data_list)):
        data_list[i] = (data_list[i])[:-1]



    # uncomment to train kmean model
    # kmeans = train_kmean(data_path,data_list,K)


    # # label histogram
    kmean_histogram(data_path,data_list,K)


def sift(root_path,img_path):
    pth = os.path.join(root_path, img_path)
    print(pth)

    # read the image
    img = cv2.imread(pth)

    # convert to greyscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray,(500,500),interpolation = cv2.INTER_AREA)

    sift = cv2.SIFT_create()
    step_size = 20
    keypoints = [cv2.KeyPoint(x, y, step_size) for y in range(0, gray.shape[0], step_size) 
                                    for x in range(0, gray.shape[1], step_size )]


    keypoints, descriptors = sift.compute(gray, keypoints)
    # keypoints, descriptors = sift.detectAndCompute(gray, None)
    
    return keypoints, descriptors



def train_kmean(data_path,img_list,K):
    dico = []
    counter = 0
    for ele in img_list:
        if counter % 3 == 0:
            keypoints, descriptors = sift(data_path,ele)

            for d in descriptors:
                dico.append(d)

        counter +=1
        

  

    kmeans = MiniBatchKMeans(n_clusters=K,batch_size=15000, verbose=1).fit(dico)

    # save
    with open(os.path.join('data process/labelme', 'dmodel.pkl'),'wb') as f:
        pickle.dump(kmeans,f)

    f.close()
    
    return kmeans

    
def kmean_histogram(data_path,img_list,K):

    with open(os.path.join('data process/labelme', 'dmodel.pkl'), 'rb') as f:
        kmeans = pickle.load(f)
    f.close()

    kmeans.verbose = False
    histo_array = None

    counter = 0
    for ele in img_list:
        keypoints, descriptors = sift(data_path,ele)

        histo = np.zeros(K)
        nkp = np.size(keypoints)

        for d in descriptors:
            idx = kmeans.predict([d])
            histo[idx] += 1/nkp 
        
        histo = np.expand_dims(histo, axis=0)


        if counter == 0:
            histo_array = histo
        else:
            histo_array= np.concatenate((histo_array,histo), axis=0)

        counter+=1
    print(histo_array.shape)
    np.save(os.path.join('data process/labelme', 'dsift_histo_'+str(len(img_list))+'.npy'),histo_array)

    


if __name__ == '__main__':


    data_list = 'data process/labelme/names_modified.txt'

    data_path = 'processed_dataset_v4/image'
    K = 1000
    labelme_sift(data_list,data_path,K)

    # data = np.load(os.path.join('data process/labelme', 'dsift_histo_'+'16139'+'.npy'))
    # print(data.shape)






