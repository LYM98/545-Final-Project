#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def f1_for_image(W=None, X=None, Y=None, N=5):
    '''
    calculate precision, recall and F1 score of each tag and number of non-zero recall 
    input:
        W: t*d-dimensional matrix, weights for X
        X: d*n-dimensional matrix, features extraction from n images
        Y: t*n-dimensional matrix, complete tags from n images
    ouput:
        f1_score_image: n-dimensional vector, f1_score for each image
    '''
    topN_index = topN_prediction(W, X, Y, N) #t*n
    n = X.shape[1]
    t = Y.shape[0]
    X=X.T
    percision=0
    recall=0
    f1_score_image=[]
    for j in range(n):
        num_of_tp=0
        true_num=0
        
        for i in range(t):
            if Y[i,j]==1:
                true_num +=1
                if i in topN_index[:,j]:                    
                    num_of_tp +=1
                    
        if len(topN_index[:,j]) == 0:
            precision = 0
        else:
            precision = num_of_tp/5
        
        if true_num == 0:
            recall = 0
        else:
            recall = num_of_tp/(true_num)
            
        if precision + recall == 0:
            f1_score_image.append(0)
        else:
            f1_score_image.append(2*precision*recall/(precision+recall))
    
    return f1_score_image

def get_tags(index_list,Y):
    t=Y.shape[0]
    tag_index_list=[]
    for i in index_list:
        temp_list=[]
        for j in range(t):
            if Y[j,i]==1:
                temp_list.append(j)
        tag_index_list.append(temp_list)
    tag_index_mat=np.mat(tag_index_list)
    tag_index_mat.index=index_list
    return tag_index_mat

def find_bestf1_index():
    best_f1_image=[]
    for i in range(len(f1_for_image)):
        if f1_for_image[i] in heapq.nlargest(5, f1_for_image):
            best_f1_image.append(i)
    #alternative method for the top5 f1 list: 
    #list(map(f1_for_image,heapq.nlargest(5, f1_for_image)))
    return best_f1_image

