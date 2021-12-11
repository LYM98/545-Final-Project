import csv
import numpy as np
import pandas as pd
import os
import cv2
import pickle
from sklearn.cluster import MiniBatchKMeans

def corel5k_sift(train_path, test_path, data_path, store_train, store_test,K):

    # load the training images' pathes 
    
    with open(train_path) as f:
        train_list = f.readlines()

    f.close()

    for i in range(len(train_list)):
        train_list[i] = (train_list[i])[:-1]


    # load the testing images' pathes

    with open(test_path) as f:
        test_list = f.readlines()

    f.close()

    for i in range(len(test_list)):
        test_list[i] = (test_list[i])[:-1]


    # uncomment to train kmean model
    kmeans = train_kmean(data_path,train_list,K)

    # # testing histogram
    kmean_histogram(data_path,test_list,K)
    # training histogram
    kmean_histogram(data_path,train_list,K)


def sift(root_path,img_path):
    pth = os.path.join(root_path, img_path+'.jpeg')
    print(pth)

    # read the image
    img = cv2.imread(pth)

    # convert to greyscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    sift = cv2.SIFT_create()
    step_size = 10
    keypoints = [cv2.KeyPoint(x, y, step_size) for y in range(0, gray.shape[0], step_size) 
                                    for x in range(0, gray.shape[1], step_size )]


    keypoints, descriptors = sift.compute(gray, keypoints)
    # keypoints, descriptors = sift.detectAndCompute(gray, None)
    
    return keypoints, descriptors



def train_kmean(data_path,img_list,K):
    # specify K
    # print('Image List length: ', len(img_list))
    dico = []
    for ele in img_list:
        keypoints, descriptors = sift(data_path,ele)
        for d in descriptors:

          
            # print(' -----------------------------------------------')
            # print(d)
            # print(' -----------------------------------------------')    
       
            dico.append(d)

  

    kmeans = MiniBatchKMeans(n_clusters=K,batch_size=len(img_list)*3, verbose=1).fit(dico)

    # save
    with open(os.path.join(data_path, 'dmodel.pkl'),'wb') as f:
        pickle.dump(kmeans,f)

    f.close()
    
    return kmeans

    
def kmean_histogram(data_path,img_list,K):

    with open(os.path.join(data_path, 'dmodel.pkl'), 'rb') as f:
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
    np.save(os.path.join(data_path, 'sift_histo_'+str(len(img_list))+'.npy'),histo_array)

    


if __name__ == '__main__':


    train_path = 'Corel5k_dataset/corel5k_train_list.txt'
    test_path = 'Corel5k_dataset/corel5k_test_list.txt'
    data_path = 'Corel5k_dataset'
    store_train = 'Corel5k_dataset/train_dsift_histo.npy'
    store_test = 'Corel5k_dataset/test_dsift_histo.npy'
    K= 1000
    corel5k_sift(train_path, test_path, data_path, store_train, store_test,K)






