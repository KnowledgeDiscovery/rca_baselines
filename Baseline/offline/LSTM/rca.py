# import pyspot as ps
import numpy as np
import pandas as pd
import networkx as nx
from gnn_dag import GNNCI
from causalnex.structure import dynotears
from causalnex.structure.dynotears import from_pandas_dynamic
from pyspot import DSpot, Spot
from pingouin import partial_corr
import torch
from hannlstm import hannLSTM, train_model_pgd
from fastPC import Fast_PC_Causal_Graph
from castle.algorithms import PC, GOLEM
import scipy
from numpy.linalg import norm, inv
from sklearn import preprocessing


def optENMFSoft( A, P, M, c, tau, max_iter=100):

    n = A.shape[0]
    B = (1-c) * inv(np.eye(n) - c * A)
    BB = B.transpose() @ B
    
    t = 1e-30
    
    e = np.ones((n, 1))
    s = scipy.special.softmax(B @ e)
    obj = norm((s @ s.transpose()) * M, 'fro') ** 2 + tau * norm(e, 1)
    obj_old = obj
    err = 1
    iter = 0
    
    # maxIter = 1000
    errorV=[]
    
    while (err > t) and (iter < max_iter):
        s=scipy.special.softmax(B @ e)
        phi=np.diag(s) - s @ s.transpose()
        
        numerator = 4*(B.transpose() @ phi) @ (P*M) @ s
        # print(numerator)
        numerator[numerator<0]=0
        denominator = 4 * B.transpose() @ ((phi@s@s.transpose())*M)@s+ tau * np.ones((n,1))
        e=e * np.sqrt(np.sqrt(numerator/denominator))
        # print(e)
        # %err=norm(e-e_old,'fro')
        obj=norm((s@s.transpose())*M - P,'fro') ** 2 + tau * norm(e,1)
        err=np.abs(obj-obj_old)
        obj_old=obj
        iter = iter +1
        errorV.append(err)
    return e

def spot_detection(X, d: int=10, q: float=1e-4, n_init:int=100, level:float=0.98)->np.ndarray:

    # X_mean = np.mean(X, axis=0)
    # X_std = np.std(X, axis=0)
    # X_std[X_std < 1e-3] = 1
    # X = (X - X_mean) / X_std

    nvar = X.shape[1]
    T = X.shape[0]
    score_list = []
    for i in range(nvar):
        S = DSpot(d=d, q=q, n_init=n_init, level=level)
        score = []
        for t in range(T):
            xt = X[t, i]
            event = S.step(xt)
            st = 0
            if t >= n_init:
                # up alert
                if event == 1:
                    upper_threshold = S.status().to_dict()['z_up']
                    assert(xt >= upper_threshold)

                    if upper_threshold == 0:
                        upper_threshold = 0.0001

                    st = (xt - upper_threshold) / upper_threshold
                    # print('z_up is event!')
                # down alert 
                if event == -1:
                    lower_threshold = S.status().to_dict()['z_down']
                    assert(xt <= lower_threshold)

                    if lower_threshold == 0:
                        lower_threshold = 0.0001

                    st = (lower_threshold - xt) / lower_threshold
                    # print('z_down is event!')
                st = np.abs(st)
            score.append(st)
        score_list.append(score)
    np_score = np.array(score_list).transpose()
    return np_score

def detect_individual_causal(X: np.ndarray,
                   method:str='SPOT',
                   args:dict={'d': 10, 'q': 1e-4, 'n_init': 100, 'level':0.98})->np.ndarray:
    if method == 'SPOT':
        d = args['d']
        q = args['q']
        n_init = args['n_init']
        level = args['level']
        score = spot_detection(X, d, q, n_init, level)
    return score


    
# LSTM based method
def lstm(X: np.ndarray, hidden: int, context: int, lam: float, lam_ridge: float, lr: float, max_iter: int, check_every: int, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    # device = torch.device('cuda:0')
    
    X_np = torch.tensor(X[np.newaxis], dtype=torch.float64, device=device)
    hannlstm = hannLSTM(X.shape[-1], hidden=hidden).to(device=device)
    X_np = X_np.float()
    train_loss_list = train_model_pgd(hannlstm, X_np, context=context, lam=lam, lam_ridge=lam_ridge, lr=lr, max_iter=max_iter, check_every=check_every)
    W_est = hannlstm.GC(False).cpu().data.numpy()

    return W_est


def generate_causal_graph(X: np.ndarray,
                          method: str ='dynotears',
                          args: dict = {'lag': 10,
                                        'lambda_w': 1e-3,
                                        'lambda_a': 1e-3,
                                        'max_iter':30})->np.ndarray:
                                            
    if method == 'pc':
        pc = PC()
        pc.learn(X)
        W_est = np.array(pc.causal_matrix)
    elif method == 'golem':
        golem = GOLEM(num_iter=100)
        golem.learn(X)
        W_est = np.array(golem.causal_matrix)
    elif method == 'lstm':
        torch.set_default_tensor_type(torch.FloatTensor)
        hidden = args['hidden']
        context = args['context']
        lam = args['lam']
        lam_ridge = args['lam_ridge']
        lr = args['lr']
        max_iter = args['max_iter']
        check_every = args['check_every']
        device = args['device']
        W_est = lstm(X, hidden, context, lam, lam_ridge, lr, max_iter, check_every, device)
    elif method == 'dynotears':
        if 'columns' not in args:
            columns = ['V{}'.format(i) for i in range(X.shape[1])]
        else:
            columns = args['columns']
        lag = args['lag']
        lambda_w = args['lambda_w']
        lambda_a = args['lambda_a']
        max_iter = args['max_iter']

        X_lag = np.roll(X,1,axis=0)
        for lag_o in range(2,lag+1):
            X_lag = np.hstack((X_lag,np.roll(X,lag_o, axis=0)))
        W_est = dynotears.from_numpy_dynamic(X, X_lag, lambda_w, lambda_a, max_iter)
    elif method == 'fastpc':
        W_est = Fast_PC_Causal_Graph(pd.DataFrame(X),alpha=10**-6,cuda=True)
    return W_est

# generate transition matrix from weight matrix
# W: W[i,j] i->j
def generate_Q(X:np.ndarray, W:np.ndarray, RI:int, rho:float, columns: list=None):
    n = W.shape[0]
    if columns is None:
        columns=['V{}'.format(i) for i in range(n)]
    df = pd.DataFrame(X, index=[i for i in range(X.shape[0])], columns=columns)
    
    # parent nodes
    PAak = [columns[i] for i, x in enumerate(W[:, RI]) if (x == 1) and (i != RI)]
    vak = columns[RI]
    # PA = [[columns[j] for j, x in enumerate(W[:, i]) if x == 1] for i in range(n)]
    # PAak_minus = [[c for c in PAak if c!=columns[i]] for i in range(n)]

    # partial correlation
    Rpc = []
    for i in range(n):
        if i == RI:
            Rpc.append(0)
            continue
        vi = columns[i]
        PAak_minus_i = [c for c in PAak if c!=columns[i]]
        PAi = [columns[j] for j, x in enumerate(W[:, i]) if (x == 1) and (i != j) and (RI != j)]
        covar = list(set(PAak_minus_i).union(set(PAi)))
        rdf = partial_corr(df, x=vak, y=vi, covar=covar)
        Rpc.append(np.abs(rdf.values[0, 1]))
        
    Q = np.zeros((n, n))
    for i in range(n):
        P = 0
        for j in range(n):
            if i == j:
                continue
            # from result to cause
            if W[j][i] == 1:
                Q[i][j] = Rpc[j]
                # from cause to result:
                if W[i][j] == 0:
                    Q[j][i] = rho * Rpc[i]
                # stay
                P = max(P, Q[i][j])
        Q[i][i] = max(0., Rpc[i] - P)
    # normalize each row
    rsum = np.sum(Q, axis=1).reshape(-1 , 1)
    rsum[rsum==0] = 1
    Q = Q / rsum
    return Q

# random walk with restart
def propagate_error(Q:np.ndarray, start:int, steps:int=1000, rp:float=0.05, max_self:int=10)->np.ndarray:
    n = Q.shape[0]
    count = np.zeros(n)
    current = start
    self_visit = 0
    for step in range(steps):
        # print(current)
        # input()
        if np.random.rand() > rp:
            prob = Q[current, :]
            if np.sum(prob) != 1:
                continue
            next = np.random.choice(n, 1, p=prob)[0]
            # check if need a restart, get stuck in one node
            if next == current:
                self_visit += 1
                if self_visit == max_self:
                    current = start
                    self_visit = 0
                    continue
            current = next
            count[current] += 1
        else:
            current = start
    return count

if __name__ == '__main__':
    data = np.load('may_pod_level_data.npy', allow_pickle=True).item()
    label = 'Book_Info_product'
    X = data[label]['Sequence'][:48000, :]
    X = np.sum(X.reshape((-1, 100, X.shape[1])), axis=1)
    columns = data[label]['Pod_Name'] + data[label]['JMeter_Feature']
    std = np.std(X, axis=0)
    idx = [i for i, x in enumerate(std > 1e-3) if x]
    # idx = list(range(30))
    X = X[:, idx]
    columns = [columns[i] for i in idx]

    print('X shape: ', X.shape)

    print('Detecting Individual Causal ...')
    ind_casual_score = detect_individual_causal(X, method='SPOT', args={'d':10, 'q':1e-4, 'n_init':100, 'level':0.98})
    ind_casual_score = np.sum(ind_casual_score, axis=0)
    normalized_ind_casual_score = ind_casual_score
    normalized_ind_casual_score[:-1] = preprocessing.normalize([ind_casual_score[:-1]])
    print('Detecting Individual Causal Done!')

    # causal graph
    print('Generating Causal Graph ...')
    cg = generate_causal_graph(X, method='gnn', args={'lag': 20, 'lambda_w': 1e-3, 'lambda_a': 1e-2})
    print('Generating Causal Graph Done!')
    # threshold top K
    K = 0.3*len(cg.reshape(-1))
    threshold = sorted(cg.reshape(-1), reverse=True)[K-1] 
    W = np.where(cg>=threshold, 1, 0)
    # Wij : i->j
    W = W.transpose()
    # print('W:', W[:, -1])
    print('Generating Q ...')
    Q = generate_Q(X, W, RI=W.shape[0]-1, rho=1e-2)
    print('Q:', Q[-1, :])
    print('Q sum: ', np.sum(Q))
    print('Generating Q Done!')
    # error propagation
    print('Propagaing Error ...')
    steps = 10000
    count = propagate_error(Q, start=W.shape[0]-1, steps=steps)
    count /= steps
    print('Propagating Eroor Done!')

    # root cause ranking
    print('Individual Causal Score: ', normalized_ind_casual_score)
    print('Topological Causal score: ', count)
    alpha = 0.3
    score = alpha * normalized_ind_casual_score[:-1] + (1 - alpha) * count[:-1]
    # top K
    K = 5
    ranking = np.argsort(score)[::-1]
    for i in range(K):
        print('{}: {} {}'.format(i, columns[ranking[i]], score[ranking[i]]))
