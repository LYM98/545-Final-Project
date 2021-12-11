import numpy as np
import matplotlib.pyplot as plt
import lib
from sklearn.random_projection import GaussianRandomProjection
from importlib import reload

feature_directory = '/Users/yushanyang/Yushan file/Umich/EECS545/Final Project/Github/tot_features.npy'
tag_directory = '/Users/yushanyang/Yushan file/Umich/EECS545/Final Project/Github/binary_feature.npy'
W_directory = '/Users/yushanyang/Yushan file/Umich/EECS545/Final Project/Github/Our_data/Our_W.npy'
X = np.load(feature_directory)
Y = np.load(tag_directory)

random_state = np.random.RandomState(42)
combine = [[0.14,0.1,3,0.25]]
for eps, lamda, gamma, p in combine:
    transformer = GaussianRandomProjection(random_state=random_state, eps=eps)
    X_new = transformer.fit_transform(X).copy()
    X_new = X_new.T
    Y_new = Y.T
    
    Y_new = Y_new[Y_new.sum(axis=1) >= 40,:]
    filt = (Y_new.sum(axis=0) != 0).copy()
    Y_new = Y_new[:,filt]
    X_new = X_new[:,filt]
    
    np.random.seed(42)
    X_new = X_new[:, np.random.permutation(X_new.shape[1])]
    np.random.seed(42)
    Y_new = Y_new[:, np.random.permutation(Y_new.shape[1])]
    
    # split train and test data
    n = X_new.shape[1]
    X_train, X_test = X_new[:,:int(n*0.9)].copy(), X_new[:,int(n*0.9):].copy()
    Y_train, Y_test = Y_new[:,:int(n*0.9)].copy(), Y_new[:,int(n*0.9):].copy()
    
    # split train and validation data
    n = X_train.shape[1]
    X_train, X_valid = X_train[:,:int(n*0.9)].copy(), X_train[:,int(n*0.9):].copy()
    Y_train, Y_valid = Y_train[:,:int(n*0.9)].copy(), Y_train[:,int(n*0.9):].copy()
    
    #normalize training data
    X_mean = np.mean(X_train,axis=1).copy()
    X_std = np.std(X_train,axis=1).copy()
    X_train = (X_train-X_mean[:,None])/X_std[:,None]
    X_valid = (X_valid-X_mean[:,None])/X_std[:,None]
    X_test = (X_test-X_mean[:,None])/X_std[:,None]
    
    # initilize matrix
    d = X_train.shape[0]
    t = Y_train.shape[0]
    W = np.zeros((t,d))
    B = np.zeros((t,t))
    
    W_opt, B_opt, total_loss = lib.opt(W, B, X_train, Y_train, p, lamda, gamma)
    precision, recall, f1_score, non_zero_recall = lib.evalution(W_opt, X_test, Y_test,8)
    note.append([eps,lamda,gamma,p,np.round(np.mean(precision),3),np.round(np.mean(recall),3),np.round(np.mean(f1_score),3),non_zero_recall])
print(note)