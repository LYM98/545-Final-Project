import numpy as np

def PQ(p=0.5,Y=None):
    '''
    input:
    Y: t*n-dimensional matrix, partial tags from n images
    p: probability of text corruption
    output: 
    P: t*t-dimensional matrix
    Q: t*t-dimensional matrix
    P = (1 − p)YY^T
    Q = (1 − p)^2YY^T + p(1 − p)δ(YY^T)
    '''
    t = Y.shape[0]
    P = (1-p)*Y@Y.T
    Q = (1-p)**2*Y@Y.T+p*(1-p)*np.diagflat((Y@Y.T).diag())
    return P,Q

def opt_periter(W=None, B=None, X=None, Y=None, P=None, Q=None, lamda=0.1, gamma=0.1):
    '''
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
        W: t*d-dimensional matrix, weights for X
        B: t*t-dimensional matrix, weights for Y
        W = BYX^T(XX^T + nλI)^−1
        B = (γP+WXY^T)(γQ + YY^T)^-1
    '''
    d = X.shape[0]
    W = B@Y@X.T@np.linalg.inv(X@X.T+n*lamda*np.identity(d))
    B = (gamma*P+W@X@Y.T)@np.linalg.inv(gamma*Q+Y@Y.T)
    return W,B

def loss(W=None, B=None, X=None, Y=None, P=None, Q=None, lamda=0.1, gamma=0.1):
    '''
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
        loss
    '''
    L = [] # loss list
    P, Q = PQ(p,Y)
    W, B = opt_periter(W, B, X, Y, P, Q, lamda, gamma)
    L.append(loss(W, B, X, Y, P, Q, lamda, gamma))
    W, B = opt(W, B, X, Y, P, Q, lamda, gamma)
    l = loss(W, B, X, Y, P, Q, lamda, gamma)
    while l/L[-1] < 0.95: # stop iteration when loss decreases less than 5%
        L.append(l)
        W, B = opt(W, B, X, Y, P, Q, lamda, gamma)
        l = loss(W, B, X, Y, P, Q, lamda, gamma)
    L.append(l)
    return W, B, L

def tag_bootstrap(W=None, B=None, X=None, Y=None , p, lamda, gamma):
    '''
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
        loss
    '''
    return opt(W, B, X, B@Y , p, lamda, gamma)