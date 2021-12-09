import numpy as np
import matplotlib.pyplot as plt  

def cal_PQ(p=0.5,Y=None):
    '''
    calculate matrix P, Q needed for optimization
    input:
        Y: t*n-dimensional matrix, partial tags from n images
        p: probability of text corruption
    output: 
        P: t*t-dimensional matrix, P = (1 − p)YY^T
        Q: t*t-dimensional matrix,  Q = (1 − p)^2YY^T + p(1 − p)δ(YY^T)      
    '''
    P = (1-p)*Y@Y.T
    Q = (1-p)**2*Y@Y.T+p*(1-p)*np.diagflat((Y@Y.T).diagonal()) 

    return P,Q

def opt_periter(W=None, B=None, X=None, Y=None, P=None, Q=None, lamda=0.1, gamma=0.1, opt_type='total', index=None):
    '''
    calculate updated matrix W, B for one iteration
    input:
        W: t*d-dimensional matrix, weights for X
        B: t*t-dimensional matrix, weights for Y
        X: d*n-dimensional matrix, features extraction from n images
        Y: t*n-dimensional matrix, partial tags from n images
        lamda: regularization factor for W 
        P: t*t-dimensional matrix
        Q: t*t-dimensional matrix
        gamma: regularization factor for B
        opt_type: the type of optimization
        index: index involved in the loss function
    ouput:
        W_new: t*d-dimensional matrix, weights for X, W_new = BYX^T(XX^T + nλI)^−1
        B_new: t*t-dimensional matrix, weights for Y, B_new = (γP+WXY^T)(γQ + YY^T)^-1
    '''
    d = X.shape[0]
    n = X.shape[1] 
    
    if opt_type == 'total':
        W_new = B@Y@X.T@np.linalg.pinv(X@X.T+n*lamda*np.identity(d))
        B_new = (gamma*P+W_new@X@Y.T)@np.linalg.pinv(gamma*Q+Y@Y.T) 
    elif opt_type == 'rare':
        W_old, B_old = W.copy(), B.copy()
        
        W_new = B@Y@X.T@np.linalg.inv(X@X.T+n*lamda*np.identity(d))
        W_old[index,:] = W_new[index,:]
        W_new = W_old
        
        B_new = (gamma*P+W_new@X@Y.T)@np.linalg.inv(gamma*Q+Y@Y.T)
        B_old[index,:] = B_new[index,:]
        B_new = B_old
        
    return W_new, B_new

def total_loss(W=None, B=None, X=None, Y=None, P=None, Q=None, lamda=0.1, gamma=0.1):
    '''
    calculate the total loss
    input:
        W: t*d-dimensional matrix, weights for X
        B: t*t-dimensional matrix, weights for Y
        X: d*n-dimensional matrix, features extraction from n images
        Y: t*n-dimensional matrix, partial tags from n images
        lamda: regularization factor for W 
        P: t*t-dimensional matrix
        Q: t*t-dimensional matrix
        gamma: regularization factor for B
    ouput:
        total loss
    '''
    n = X.shape[1]
    main_loss = 1/n*np.sum((B@Y-W@X)**2)
    W_reg = lamda*np.sum(W**2)
    B_reg = gamma*recon_error(B,Y,P,Q)
    t_loss = main_loss+W_reg+B_reg
    print('main loss: ', np.round(main_loss,4), 'W loss: ', np.round(W_reg,4), 'B loss', np.round(B_reg,4), 'total_loss: ', np.round(t_loss,4))
    
    return t_loss

def recon_error(B=None,Y=None,P=None,Q=None):
    '''
    calculate the reconstruction error
    input:
        B: t*t-dimensional matrix, weights for Y
        Y: t*n-dimensional matrix, partial tags from n images
        P: t*t-dimensional matrix
        Q: t*t-dimensional matrix
    ouput:
        reconstruction error
    '''
    n = Y.shape[1]

    return 1/n*np.trace(B@Q@B.T-2*P@B.T+Y@Y.T)

def opt(W=None, B=None, X=None, Y=None, p=0.5, lamda=0.1, gamma=0.1, opt_type='total', index=None):
    '''
    optimize the parametersin W, B and return the total loss in each iteration
    input:
        W: t*d-dimensional matrix, weights for X
        B: t*t-dimensional matrix, weights for Y
        X: d*n-dimensional matrix, features extraction from n images
        Y: t*n-dimensional matrix, partial tags from n images
        p: probability of text corruption
        lamda: regularization factor for W 
        gamma: regularization factor for B
        opt_type: the type of optimization
        index: index involved in the loss function
    ouput:
        W_new: t*d-dimensional matrix, weights for X
        B_new: t*t-dimensional matrix, weights for Y
        loss list for each iteration
    '''
    loss_list = [] 
    P, Q = cal_PQ(p,Y)
    loss_list.append(total_loss(W, B, X, Y, P, Q, lamda, gamma))
    
    W_new, B_new = opt_periter(W, B, X, Y, P, Q, lamda, gamma, opt_type, index)
    loss_list.append(total_loss(W_new, B_new, X, Y, P, Q, lamda, gamma))

    W_new, B_new = opt_periter(W_new, B_new, X, Y, P, Q, lamda, gamma, opt_type, index)
    loss = total_loss(W_new, B_new, X, Y, P, Q, lamda, gamma)
    
    iter_num = 0
    while abs(loss-loss_list[-1])/loss_list[-1] > 1e-3 and iter_num < 30: 
        loss_list.append(loss)
        W_new, B_new = opt_periter(W_new, B_new, X, Y, P, Q, lamda, gamma, opt_type, index)
        loss = total_loss(W_new, B_new, X, Y, P, Q, lamda, gamma)
        iter_num += 1
    loss_list.append(loss)

    return W_new, B_new, loss_list

def topN_prediction(W=None, X=None, Y=None, N=5):
    '''
    calculate the most five relavent tags for each image
    input:
        W: t*d-dimensional matrix, weights for X
        X: d*n-dimensional matrix, features extraction from n images
        Y: t*n-dimensional matrix, complete tags from n images
    ouput:
        top5_index: the index of the most five relavent tags for each image
    '''
    Y_predict = W@X
    topN_index = np.argsort(Y_predict,axis=0)
    topN_index = topN_index[-N:,:]

    return topN_index
    
def evalution(W=None, X=None, Y=None, N=5):
    '''
    calculate precision, recall and F1 score of each tag and number of non-zero recall 
    input:
        W: t*d-dimensional matrix, weights for X
        X: d*n-dimensional matrix, features extraction from n images
        Y: t*n-dimensional matrix, complete tags from n images
    ouput:
        precision_list: t-dimensional vector, precision for each tag
        recall_list: t-dimensional vector, recall for each tag
        f1_score_list: t-dimensional vector, f1_score for each tag
        non_zero_recall: number of non-zero recall
    '''
    topN_index = topN_prediction(W, X, Y, N)
    t = W.shape[0]

    # initialize the output
    precision_list = []
    recall_list = []
    f1_score_list = []
    non_zero_recall = 0

    for j in range(t):
        true_index = [i for i, v in enumerate(Y[j,:]) if v == 1] 
        predict_index = [i[1] for i, v in np.ndenumerate(topN_index) if v == j]
        num_of_tp = len(np.intersect1d(true_index, predict_index))
        
        if len(predict_index) == 0:
            precision = 0
        else:
            precision = num_of_tp/len(predict_index)
        precision_list.append(precision)
        
        if len(true_index) == 0:
            recall = 0
        else:
            recall = num_of_tp/len(true_index)
        recall_list.append(recall)
        
        if recall > 0:
            non_zero_recall += 1
        
        if precision + recall == 0:
            f1_score_list.append(0)
        else:
            f1_score_list.append(2*precision*recall/(precision+recall))

    return precision_list, recall_list, f1_score_list, non_zero_recall

def bootstrap_periter(W=None, B=None, X=None, Y=None, p=0.5, lamda=0.1, gamma=0.1):
    '''
    do one layer of bootstraping
    input:
        W: t*d-dimensional matrix, weights for X
        B: t*t-dimensional matrix, weights for Y
        X: d*n-dimensional matrix, features extraction from n images
        Y: t*n-dimensional matrix, partial tags from n images
        p: probability of text corruption
        lamda: regularization factor for W 
        gamma: regularization factor for B
    ouput:
        W: t*d-dimensional matrix, weights for X
        B: t*t-dimensional matrix, weights for Y
        loss list for each iteration
    '''
    Y_new = B@Y
    idx_lager_than_1 = (Y_new > 1)
    idx_smaller_than_0 = (Y_new < 0)
    Y_trucated = Y_new.copy()
    Y_trucated[idx_lager_than_1] = 1
    Y_trucated[idx_smaller_than_0] = 0
    
    return weight_opt(W, B, X, Y_trucated, p, lamda, gamma)

def bootstrap(W=None, B=None, X=None, Y=None, p=0.5, lamda=0.1, gamma=0.1, Xhold=None, Yhold=None):
    '''
    do bootstraping until F1 score does not change over 1% or total number of iteration over 20
    input:
        W: t*d-dimensional matrix, weights for X
        B: t*t-dimensional matrix, weights for Y
        X: d*n-dimensional matrix, features extraction from n images
        Y: t*n-dimensional matrix, partial tags from n images
        p: probability of text corruption
        lamda: regularization factor for W 
        gamma: regularization factor for B
        Xhold: feature extraction metrix for hold-out set
        Yhold: complete tag metrix for hold-out set
    ouput:
        W_new: t*d-dimensional matrix, weights for X
        B_new: t*t-dimensional matrix, weights for Y
    '''
    W_new, B_new, _ = bootstrap_periter(W, B, X, Y, p, lamda, gamma)
    _, _, f1_score_old, _ = evalution(W_new, Xhold, Yhold)
    f1_score_old = np.mean(f1_score_old)
    print('F1: ', f1_score_old)
    
    W_new, B_new, _ = bootstrap_periter(W_new, B_new, X, Y, p, lamda, gamma)
    _, _, f1_score_new, _ = evalution(W_new, Xhold, Yhold)
    f1_score_new = np.mean(f1_score_new)
    print('F1: ', f1_score_new)
    
    iter_num = 0
    while (f1_score_new - f1_score_old)/f1_score_old > 0.001 and iter_num < 30:
        f1_score_old = f1_score_new
        W_new, B_new, _ = bootstrap_periter(W_new, B_new, X, Y, p, lamda, gamma)
        _, _, f1_score_new, _ = evalution(W_new, Xhold, Yhold)
        f1_score_new = np.mean(f1_score_new)
        print('F1: ', f1_score_new)
        iter_num += 1

    return W_new, B_new

def reopt_periter(W=None, B=None, X=None, Y=None, p=0.5, lamda=0.1, gamma=0.1, Xhold=None, Yhold=None):
    '''
    do re-optimization of tags whose recall is lower than the average value with one iteration
    input:
        W: t*d-dimensional matrix, weights for X
        B: t*t-dimensional matrix, weights for Y
        X: d*n-dimensional matrix, features extraction from n images
        Y: t*n-dimensional matrix, partial tags from n images
        p: probability of text corruption
        lamda: regularization factor for W 
        gamma: regularization factor for B
        Xhold: feature extraction metrix for hold-out set
        Yhold: complete tag metrix for hold-out set
    ouput:
        W: t*d-dimensional matrix, weights for X
        B: t*t-dimensional matrix, weights for Y
    '''
    _, recall, _, _ = evalution(W, Xhold, Yhold)
    threshold = np.mean(recall)
    index = [i for i, v in enumerate(recall) if v < threshold]
    W_new, B_new, _ = opt(W, B, X, Y, p, lamda, gamma, 'rare', index)
        
    return W_new, B_new

def reopt(W=None, B=None, X=None, Y=None, p=0.5, lamda=0.1, gamma=0.1, Xhold=None, Yhold=None):
    '''
    do re-optimization of tags until F1 score does not change over 5% for hold-out set
    input:
        W: t*d-dimensional matrix, weights for X
        B: t*t-dimensional matrix, weights for Y
        X: d*n-dimensional matrix, features extraction from n images
        Y: t*n-dimensional matrix, partial tags from n images
        p: probability of text corruption
        lamda: regularization factor for W 
        gamma: regularization factor for B
        Xhold: feature extraction metrix for hold-out set
        Yhold: complete tag metrix for hold-out set
    ouput:
        W_new: t*d-dimensional matrix, weights for X
        B_new: t*t-dimensional matrix, weights for Y
    '''
    _, _, f1_score_old, _ = evalution(W, Xhold, Yhold)
    f1_score_old = np.mean(f1_score_old)
    
    W_new, B_new = reopt_periter(W, B, X, Y, p, lamda, gamma, Xhold, Yhold)
    _, _, f1_score_new, _ = evalution(W_new, Xhold, Yhold)
    f1_score_new = np.mean(f1_score_new)

    while (f1_score_new - f1_score_old)/f1_score_old > 0.01:
        f1_score_old = f1_score_new
        W_new, B_new = reopt_periter(W_new, B_new, X, Y, p, lamda, gamma, Xhold, Yhold)
        _, _, f1_score_new, _ = evalution(W_new, Xhold, Yhold)
        f1_score_new = np.mean(f1_score_new)
    
    return W_new, B_new

def weight_matrix(Y=None):  
    '''
    calculate the diagonal weight matrix
    input:
        Y: t*n-dimensional matrix, partial tags from n images
    ouput:
        diagonal weight matrix
    '''
    n_w = np.sum(Y, axis=1)
    t = n_w.shape[0]
    c_w = np.ones(t)
    for i in range(t):
        if n_w[i] != 0:
            c_w[i] = 1/n_w[i]
    weight_each_example = Y.T@c_w.reshape(-1,1)

    return np.diagflat(weight_each_example)


def weight_opt_periter(W=None, B=None, X=None, Y=None, P=None, Q=None, lamda=0.1, gamma=0.1, weight=None):
    '''
    calculate updated matrix W, B for one iteration
    input:
        W: t*d-dimensional matrix, weights for X
        B: t*t-dimensional matrix, weights for Y
        X: d*n-dimensional matrix, features extraction from n images
        Y: t*n-dimensional matrix, partial tags from n images
        lamda: regularization factor for W 
        P: t*t-dimensional matrix
        Q: t*t-dimensional matrix
        gamma: regularization factor for B
        weight: weight matrix
    ouput:
        W_new: t*d-dimensional matrix, weights for X, W_new = BYX^T(XX^T + nλI)^−1
        B_new: t*t-dimensional matrix, weights for Y, B_new = (γP+WXY^T)(γQ + YY^T)^-1
    '''
    d = X.shape[0]
    n = X.shape[1] 
    W_new = B@Y@weight@X.T@np.linalg.inv(X@weight@X.T+n*lamda*np.identity(d))
    B_new = (gamma*P+W_new@X@weight@Y.T)@np.linalg.inv(gamma*Q+Y@weight@Y.T)

    return W_new, B_new

def total_loss_weighted(W=None, B=None, X=None, Y=None, P=None, Q=None, lamda=0.1, gamma=0.1, weight_mat=None):
    '''
    calculate the total loss
    input:
        W: t*d-dimensional matrix, weights for X
        B: t*t-dimensional matrix, weights for Y
        X: d*n-dimensional matrix, features extraction from n images
        Y: t*n-dimensional matrix, partial tags from n images
        lamda: regularization factor for W 
        P: t*t-dimensional matrix
        Q: t*t-dimensional matrix
        gamma: regularization factor for B
    ouput:
        total loss
    '''
    n = X.shape[1]
    weight_vec = np.diag(weight_mat).reshape(-1, 1)
    main_loss = (1/n*(np.sum((B@Y-W@X)**2, axis=0))@weight_vec)[0]
    W_reg = lamda*np.sum(W**2)
    B_reg = gamma*recon_error(B,Y,P,Q)
    t_loss = main_loss + W_reg + B_reg
    print('main loss: ', np.round(main_loss,4), 'W loss: ', np.round(W_reg,4), 'B loss', np.round(B_reg,4), 'total_loss: ', np.round(t_loss,4))
    
    return t_loss

def cal_weight_PQ(p=0.5,Y=None,weight=None):
    '''
    calculate matrix P, Q needed for optimization
    input:
        Y: t*n-dimensional matrix, partial tags from n images
        p: probability of text corruption
    output: 
        P: t*t-dimensional matrix, P = (1 − p)YY^T
        Q: t*t-dimensional matrix,  Q = (1 − p)^2YY^T + p(1 − p)δ(YY^T)      
    '''
    P = (1-p)*Y@weight@Y.T
    Q = (1-p)**2*Y@weight@Y.T+p*(1-p)*np.diagflat((Y@weight@Y.T).diagonal()) 

    return P,Q

def weight_opt(W=None, B=None, X=None, Y=None, p=0.5, lamda=0.1, gamma=0.1):
    '''
    optimize the parametersin W, B and return the total loss in each iteration
    input:
        W: t*d-dimensional matrix, weights for X
        B: t*t-dimensional matrix, weights for Y
        X: d*n-dimensional matrix, features extraction from n images
        Y: t*n-dimensional matrix, partial tags from n images
        p: probability of text corruption
        lamda: regularization factor for W 
        gamma: regularization factor for B
    ouput:
        W_new: t*d-dimensional matrix, weights for X
        B_new: t*t-dimensional matrix, weights for Y
        loss list for each iteration
    '''
    loss_list = [] 
    weight = weight_matrix(Y)
    P, Q = cal_weight_PQ(p,Y,weight)
    
    W_new, B_new = weight_opt_periter(W, B, X, Y, P, Q, lamda, gamma, weight)
    loss_list.append(total_loss_weighted(W_new, B_new, X, Y, P, Q, lamda, gamma, weight))
    
    W_new, B_new = weight_opt_periter(W_new, B_new, X, Y, P, Q, lamda, gamma, weight)
    loss = total_loss_weighted(W_new, B_new, X, Y, P, Q, lamda, gamma, weight)
    iter_num = 0

    while abs(loss-loss_list[-1])/loss_list[-1] > 1e-3 and iter_num < 20: 
        loss_list.append(loss)
        W_new, B_new = weight_opt_periter(W_new, B_new, X, Y, P, Q, lamda, gamma, weight)
        loss = total_loss_weighted(W_new, B_new, X, Y, P, Q, lamda, gamma, weight)
        iter_num += 1
    loss_list.append(loss)

    return W_new, B_new, loss_list