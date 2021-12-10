# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 02:28:51 2021

@author: LBY
"""
# import modules
import numpy as np
import matplotlib.pyplot as plt
import lib
from sklearn.random_projection import GaussianRandomProjection
import random
from lib import topN_prediction
import f1_for_image

# In[] load raw feature data and 
features_all = np.load('tot_features.npy')
label_all = np.load('binary_feature.npy')

# In[] random projection
a = np.random.RandomState(42)
transformer = GaussianRandomProjection(random_state=a, eps=0.14)
feature_to_use = transformer.fit_transform(features_all)

# In[] transpose to the right dimension
X = feature_to_use.T
Y = label_all.T

# In[] load the label names and strip
file = 'feature_names.csv'
with open(file) as f:
    for row in f:
        cont = row.split(',')
        
cont[0] = 'light'

name_list = []
for i in cont:
    l = i.strip()
    name_list.append(l)
    
# In[] filter out the label names that appears more than 40 times
Y = label_all.T   
M = Y.copy()
idx_list = []

tot_num = 0
for i in range(M.shape[0]):
    if np.sum(M[i, :]) >= 40:
        tot_num += 1
        idx_list.append(i)
idx_mat = M.sum(axis=1) >= 40
Y = Y[idx_mat,:]
name_list_to_use = []
for i in idx_list:
    name_list_to_use.append(name_list[i])
    
    
# In[] load the images' path file
file_img_name = 'names_modified.csv'
path_list = []
with open(file_img_name) as f:
    for row in f:
        path_list.append(row.strip())
        
# In[] filter out the images that do not have a label after previous filtering
M_temp = Y.copy()
idx_list_for_names = []
for j in range(M_temp.shape[1]):
    temp_sum = np.sum(M_temp[:, j])
    if temp_sum != 0:
        idx_list_for_names.append(j)
path_list_to_use = []
for i in idx_list_for_names:
    path_list_to_use.append(path_list[i])
# In[] filter out images that do not have a label after previous filtering
filt = Y.sum(axis=0) != 0
Y = Y[:,filt] 
X = X[:,filt]

# In[] permutation with paths
np.random.seed(42)
new_order = np.random.permutation(Y.shape[1])

path_list_new_order = []
for i in new_order:
    path_list_new_order.append(path_list_to_use[i])
    
# In[] divide the path into training set and test set
n = X.shape[1]
path_order_train, path_order_test = path_list_new_order[:int(n*0.9)], path_list_new_order[int(n*0.9):]

# In[] divide features and labels into training set and test set
np.random.seed(42)
X = X[:, np.random.permutation(X.shape[1])]
np.random.seed(42)
Y = Y[:, np.random.permutation(Y.shape[1])]

n = X.shape[1]
X_train, X_test = X[:,:int(n*0.9)], X[:,int(n*0.9):]
Y_train, Y_test = Y[:,:int(n*0.9)], Y[:,int(n*0.9):]

# In[] normalize training set
X_mean = np.mean(X_train,axis=1)
X_std = np.std(X_train,axis=1)
X_train = (X_train-X_mean[:,None])/X_std[:,None]

# In[] normalize test set
X_test = (X_test - X_mean[:,None])/X_std[:,None]

# In[] load trained weight W
W_to_use = np.load('Our_W.npy')

# In[] calculate the f1 score list for all of the images in test set
f1_image_list = f1_for_image.f1_for_image(W=W_to_use, X=X_test, Y=Y_test, N=8)
f1_image_list_np = np.array(f1_image_list)


# In[] find the 5 high f1 images, 5 random images and 5 low f1 score images
top_index = np.argsort(f1_image_list_np)
n_select = 5

low_index_sel = top_index[0:n_select]

top_index_sel = top_index[-n_select:]

random.seed(2)
rand_index_sel = random.sample(list(top_index), n_select)
rand_index_sel = np.array(rand_index_sel)

# In[] find corresponding label index
X_test_top = X_test[:, top_index_sel]
top_N_idx = topN_prediction(W=W_to_use, X=X_test_top, Y=None, N=8)

X_test_rand = X_test[:, rand_index_sel]
rand_N_idx = topN_prediction(W=W_to_use, X=X_test_rand, Y=None, N=8)

X_test_low = X_test[:, low_index_sel]
low_N_idx = topN_prediction(W=W_to_use, X=X_test_low, Y=None, N=8)

# In[] plot
import matplotlib.pyplot as plt
base_path = r'E:\EECS545\processed_dataset_v4\image'
max_image_plot = 3 * n_select
fig, axes = plt.subplots(3,5)
i = 0
j = 0

for img_num in range(max_image_plot):
    if j >= 5:
        i = i + 1
        j = 0
        plt.subplots_adjust(hspace=0.8)
    if i == 0:
        print('now is best...')
        temp_img_path_rel = path_order_test[top_index_sel[j]]
        temp_img_path = base_path + '\\' + temp_img_path_rel
        temp_img = plt.imread(temp_img_path)
        axes[i, j].set_title(f'{name_list_to_use[top_N_idx[0, j]]}, {name_list_to_use[top_N_idx[1, j]]}, {name_list_to_use[top_N_idx[2, j]]},\n {name_list_to_use[top_N_idx[3, j]]}, {name_list_to_use[top_N_idx[4, j]]}, {name_list_to_use[top_N_idx[5, j]]}, \n {name_list_to_use[top_N_idx[6, j]]}, {name_list_to_use[top_N_idx[7, j]]}', y=-0.5, fontsize=15)
        axes[i, j].imshow(temp_img)
        #plt.subplots_adjust(wspace=0.5)
        axes[i, j].set_xticks([])
        axes[i, j].set_yticks([])
        
    elif i == 1:
        print('now is random...')
        temp_img_path_rel = path_order_test[rand_index_sel[j]]
        temp_img_path = base_path + '\\' + temp_img_path_rel
        temp_img = plt.imread(temp_img_path)
        axes[i, j].set_title(f'{name_list_to_use[rand_N_idx[0, j]]}, {name_list_to_use[rand_N_idx[1, j]]}, {name_list_to_use[rand_N_idx[2, j]]}, \n {name_list_to_use[rand_N_idx[3, j]]}, {name_list_to_use[rand_N_idx[4, j]]}, {name_list_to_use[rand_N_idx[5, j]]}, \n {name_list_to_use[rand_N_idx[6, j]]}, {name_list_to_use[rand_N_idx[7, j]]}', y=-0.5, fontsize=15)
        axes[i, j].imshow(temp_img)
        #plt.subplots_adjust(wspace=0.5)
        axes[i, j].set_xticks([])
        axes[i, j].set_yticks([])
        
    else:
        print('now is low...')
        temp_img_path_rel = path_order_test[low_index_sel[j]]
        temp_img_path = base_path + '\\' + temp_img_path_rel
        temp_img = plt.imread(temp_img_path)
        axes[i, j].set_title(f'{name_list_to_use[low_N_idx[0, j]]}, {name_list_to_use[low_N_idx[1, j]]}, {name_list_to_use[low_N_idx[2, j]]}, \n{name_list_to_use[low_N_idx[3, j]]}, {name_list_to_use[low_N_idx[4, j]]}, {name_list_to_use[low_N_idx[5, j]]}, \n {name_list_to_use[low_N_idx[6, j]]}, {name_list_to_use[low_N_idx[7, j]]}', y=-0.5, fontsize=15)
        axes[i, j].imshow(temp_img)
        #plt.subplots_adjust(wspace=0.5)
        axes[i, j].set_xticks([])
        axes[i, j].set_yticks([])
        
    j = j + 1
    





