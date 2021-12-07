import numpy as np
import matplotlib.pyplot as plt

def main():
    '''
    the main function of this project
    no input
    X: d*n-dimensional matrix, feature extraction from n images
    Y: t*n-dimensional matrix, tags from n images
    output: 
    four evalution of the test data   
    '''
    feature_directory = ''
    tag_directory = ''
    X = np.load(feature_directory)
    Y = np.load(tag_directory)

    np.random.seed(42)
    X = X[:, np.random.permutation(X.shape[1])]
    np.random.seed(42)
    Y = Y[:, np.random.permutation(Y.shape[1])]

    # split train and test data
    n = X.shape[1]
    X_train, X_test = X[:,:int(n*0.8)], X[:,int(n*0.8):]
    Y_train, Y_test = X[:,:int(n*0.8)], X[:,int(n*0.8):]

    # split train and validation data
    n = X_train.shape[1]
    X_train, X_valid = X_train[:,:int(n*0.75)], X_train[:,int(n*0.75):]
    Y_train, Y_valid = X_train[:,:int(n*0.75)], X_train[:,int(n*0.75):]

    # initilize matrix
    d = X_train.shape[0]
    t = Y_train.shape[0]
    W = np.zeros((t,d))
    B = np.zeros((t,t))
    lamda = 0.1
    gamma = 0.1
    p = 0.5

    # choose one of these two
    W_new, B_new, total_loss = opt(W, B, X_train, Y_train, p, lamda, gamma)
    W_new, B_new, total_loss = weight_opt(W, B, X_train, Y_train, p, lamda, gamma)

    # plot the loss over iteration
    plt.plot(total_loss)
    plt.show()

    W_new, B_new = bootstrap(W_new, B_new, X_train, Y_train, p, lamda, gamma, X_valid, Y_valid)
    W_new, B_new = reopt(W_new, B_new, X_train, Y_train, p, lamda, gamma, X_valid, Y_valid)

    return evalution(W_new, X_test, Y_test)    

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
    # Q = (1-p)**2*Y@Y.T+p*(1-p)*np.diagflat((Y@Y.T).diag())
    Q = (1-p)**2*Y@Y.T+p*(1-p)*np.diagflat((Y@Y.T).diagonal()) # changed here

    return P,Q

def opt_periter(W=None, B=None, X=None, Y=None, P=None, Q=None, lamda=0.1, gamma=0.1):
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
    ouput:
        W_new: t*d-dimensional matrix, weights for X, W_new = BYX^T(XX^T + nλI)^−1
        B_new: t*t-dimensional matrix, weights for Y, B_new = (γP+WXY^T)(γQ + YY^T)^-1
    '''
    d = X.shape[0]
    n = X.shape[1] # change here
    W_new = B@Y@X.T@np.linalg.inv(X@X.T+n*lamda*np.identity(d))
    B_new = (gamma*P+W_new@X@Y.T)@np.linalg.inv(gamma*Q+Y@Y.T) # change here
    
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

    return 1/n*np.sum((B@Y-W@X)**2)+lamda*np.sum(W**2)+gamma*recon_error(B,Y,P,Q)

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

def opt(W=None, B=None, X=None, Y=None, p=0.5, lamda=0.1, gamma=0.1):
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
    P, Q = cal_PQ(p,Y)
    W_new, B_new = opt_periter(W, B, X, Y, P, Q, lamda, gamma)
    loss_list.append(total_loss(W_new, B_new, X, Y, P, Q, lamda, gamma))

    W_new, B_new = opt_periter(W_new, B_new, X, Y, P, Q, lamda, gamma)
    loss = total_loss(W_new, B_new, X, Y, P, Q, lamda, gamma)
    iter_num = 0
    
    while abs(loss-loss_list[-1])/loss_list[-1] > 0.05 and iter_num < 19: # stop iteration when loss decreases no larger than 5% or times of iteration over 20
        loss_list.append(loss)
        print('this is itr: ', iter_num)
        W_new, B_new = opt_periter(W_new, B_new, X, Y, P, Q, lamda, gamma)
        loss = total_loss(W_new, B_new, X, Y, P, Q, lamda, gamma)
        iter_num += 1
    loss_list.append(loss)

    return W_new, B_new, loss_list

def top5_prediction(W=None, X=None, Y=None):
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
    diff = np.abs(Y_predict - Y)
    top5_index = np.argsort(diff,axis=0)
    top5_index = top5_index[:5,:]

    return top5_index
    
def evalution(W=None, X=None, Y=None):
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
    top5_index = top5_prediction(W, X, Y)
    t = W.shape[0]

    # initialize the output
    precision_list = []
    recall_list = []
    f1_score_list = []
    non_zero_recall = 0

    for j in range(t):
        true_index = [i for i, v in enumerate(Y[j,:]) if v == 1] 
        predict_index = [i[1] for i, v in np.ndenumerate(top5_index) if v == j]
        num_of_tp = len(np.intersect1d(true_index, predict_index))
        precision = num_of_tp/len(predict_index)
        precision_list.append(precision)
        recall = num_of_tp/len(true_index)
        if recall > 0:
            non_zero_recall += 1
        recall_list.append(recall)
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
    Y = B@Y;
    idx_lager_than_1 = (Y > 1)
    idx_smaller_than_0 = (Y < 0)
    Y_trucated = Y.copy()
    Y_trucated[idx_lager_than_1] = 1
    Y_trucated[idx_smaller_than_0] = 0
    return opt(W, B, X, Y_trucated, p, lamda, gamma)

def bootstrap(W=None, B=None, X=None, Y=None, p=0.5, lamda=0.1, gamma=0.1, Xhold=None, Yhold=None):
    '''
    do bootstraping until F1 score does not change over 5%
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
    f1_score_old = mean(f1_score_old)
    W_new, B_new, _ = bootstrap_periter(W, B, X, Y, p, lamda, gamma)
    _, _, f1_score_new, _ = evalution(W_new, Xhold, Yhold)
    f1_score_new = mean(f1_score_new)

    while abs(f1_score_new - f1_score_old)/f1_score_old > 0.05:
        f1_score_old = f1_score_new
        W_new, B_new, _ = bootstrap_periter(W_new, B_new, X, Y, p, lamda, gamma)
        _, _, f1_score_new, _ = evalution(W_new, Xhold, Yhold)
        f1_score_new = mean(f1_score_new)

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
    threshold = mean(recall)
    index = [i for i, v in enumerate(recall) if v < threshold]
    # W_new, B_new, _ = opt(W[index,:], B[index,:][:,index], X, Y[index,:], p, lamda, gamma)
    W_new, B_new, _ = opt(W[index,:], B[index,:], X, Y, p, lamda, gamma)
    W[index,:] = W_new
#    j = 0
#    for i in index:
#        B[i,index] = B_new[j,:]
#        j += 1
    B[index,:] = B_new

    return W, B

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
    f1_score_old = mean(f1_score_old)
    W_new, B_new = reopt_periter(W, B, X, Y, p, lamda, gamma, Xhold, Yhold)
    _, _, f1_score_new, _ = evalution(W_new, Xhold, Yhold)
    f1_score_new = mean(f1_score_new)

    while abs(f1_score_new - f1_score_old)/f1_score_old > 0.05:
        f1_score_old = f1_score_new
        W_new, B_new, _ = reopt_periter(W_new, B_new, X, Y, p, lamda, gamma, Xhold, Yhold)
        _, _, f1_score_new, _ = evalution(W_new, Xhold, Yhold)
        f1_score_new = mean(f1_score_new)
    
    return W_new, B_new

#def weight_matrix(Y=None):
#    '''
#    calculate the diagonal weight matrix
#    input:
#        Y: t*n-dimensional matrix, partial tags from n images
#    ouput:
#        diagonal weight matrix
#    '''
#    weight = np.sum(Y==1, axis=1)
#
#    return np.diag(np.diag(weight))


def weight_matrix(Y=None):   # changed here
    '''
    calculate the diagonal weight matrix
    input:
        Y: t*n-dimensional matrix, partial tags from n images
    ouput:
        diagonal weight matrix
    '''
    n_w = np.sum(Y, axis=1).reshape(-1,1)
    c_w = 1/n_w
    weight_each_example = Y.T @ c_w

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
    n = X.shape[1] # change here
    W_new = B@Y@weight@X.T@np.linalg.inv(X@weight@X.T+n*lamda*np.identity(d))
    B_new = (gamma*P+W_new@X@weight@Y.T)@np.linalg.inv(gamma*Q+Y@weight@Y.T)   # changed here

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
    return 1/n*(np.sum((B@Y-W@X)**2, axis=0)) @ weight_vec + lamda*np.sum(W**2)+gamma*recon_error(B,Y,P,Q)


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
    P, Q = cal_PQ(p,Y)
    weight = weight_matrix(Y)
    W_new, B_new = weight_opt_periter(W, B, X, Y, P, Q, lamda, gamma, weight)
    loss_list.append(total_loss_weighted(W_new, B_new, X, Y, P, Q, lamda, gamma, weight))
    
    W_new, B_new = weight_opt_periter(W_new, B_new, X, Y, P, Q, lamda, gamma, weight)
    loss = total_loss_weighted(W_new, B_new, X, Y, P, Q, lamda, gamma, weight)
    iter_num = 0

    while abs(loss-loss_list[-1])/loss_list[-1] > 0.05 and iter_num < 19: # stop iteration when loss decreases no larger than 5% or times of iteration over 20
        print('this is itr: ', iter_num, 'loss: ', loss_list[-1])
        loss_list.append(loss)
        W_new, B_new = weight_opt_periter(W_new, B_new, X, Y, P, Q, lamda, gamma, weight)
        loss = total_loss_weighted(W_new, B_new, X, Y, P, Q, lamda, gamma, weight)
        iter_num += 1
    loss_list.append(loss)

    return W_new, B_new, loss_list