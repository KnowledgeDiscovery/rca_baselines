from numpy.lib.index_tricks import AxisConcatenator
import torch
import numpy as np
from torch.nn import Module, ModuleList, Parameter, BatchNorm1d, ParameterList, MSELoss
from torch.nn import Linear
from torch.nn import ReLU
from lbfgsb_scipy import LBFGSBScipy
from trace_expm import trace_expm
from tqdm import tqdm

torch.set_default_dtype(torch.double)
np.set_printoptions(precision=3)
torch.random.manual_seed(0)
np.random.seed(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')

# shared weights
class GNNMLP(Module):
    def __init__(self, layers, n_node):
        super(GNNMLP, self).__init__()

        self.layers = layers
        self.n_node = n_node
        self.fc = ModuleList()
        self.act = ReLU()
        self.adj_pos = ParameterList()
        self.adj_neg = ParameterList()

        for i in range(len(self.layers) - 1):
            adj_pos = Parameter(torch.zeros(n_node, n_node, device = device))
            adj_neg = Parameter(torch.zeros(n_node, n_node, device = device))

            adj_pos.bounds = self._adj_bounds()
            adj_neg.bounds = self._adj_bounds() 
            self.adj_pos.append(adj_pos)
            self.adj_neg.append(adj_neg)


        for i in range(len(self.layers) - 1):
            if i > 0:
                self.fc.append(Linear(self.layers[i] * 2, self.layers[i+1]))
            else:
                self.fc.append(Linear(self.layers[i], self.layers[i+1]))
            
    
    def get_adj(self):
        adj_list = []
        for i in range(len(self.layers) - 1):
            adj_list.append((self.adj_pos[i] - self.adj_neg[i]).detach().cpu().numpy())
        return adj_list
    
    
    def _adj_bounds(self):
        bounds = []
        for i in range(self.n_node):
            for j in range(self.n_node):
                if i == j:
                    bounds.append((0, 0))
                else:
                    bounds.append((0, None))
        return bounds
    
    def l1_reg(self):
        l1 = 0
        for i in range(len(self.layers) - 1):
            l1 += self.adj_pos[i] + self.adj_neg[i]
        l1 = l1.sum() 
        return l1

    def l2_reg(self):
        l2 = 0
        for i in range(len(self.layers) - 1):
            adj = self.adj_pos[i] - self.adj_neg[i]
            l2 += (adj ** 2).sum()
        for l in self.fc:
            l2 += (l.weight ** 2).sum()
            l2 += (l.bias ** 2).sum()
        # l2 += (self.si ** 2).sum()
        return l2

    def h_func(self):
        h = 0
        for i in range(len(self.layers) - 1):
            adj = self.adj_pos[i] - self.adj_neg[i]
            input = adj**2
            h += trace_expm(input) - self.n_node
        return h

    # X : N x n_node x dim_input
    def forward(self, X):
        # print(adj.shape)
        self.to(device)
        X.to(device)
        for i in range(len(self.layers) - 2):
            adj = torch.unsqueeze((self.adj_pos[i] - self.adj_neg[i]), 0)
            message = 0
            if i > 0:
                message = adj @ X
                X = torch.cat([X, message], dim=2)
            X = self.fc[i](X)
            X = self.act(X)
            # X = self.bn[i](X)
        adj = torch.unsqueeze((self.adj_pos[-1] - self.adj_neg[-1]), 0) 
        message = adj @ X
        X = torch.cat([X, message], dim=2) 
        X = self.fc[-1](X)
        return X

# MLP version
class GNNCI:
    def __init__(self, cfg):
        # super(GNNCI, self).__init__()
        self.n_node = cfg['n_node']

        # node structure
        self.layers = cfg['layers']


        # GNN
        self.gnn = GNNMLP(self.layers, self.n_node).to(device = device)

        self.mse_fn = MSELoss(reduction='sum').to(device = device)

        # LBFGS-B optimizer
        self.optimizer = LBFGSBScipy(list(self.gnn.parameters()))

    def dual_ascent_step(self, X, Y, lambda1, lambda2, rho, alpha, h, rho_max):
        """Perform one step of dual ascent in augmented Lagrangian."""
        h_new = None
        while rho < rho_max:
            def closure():
                self.optimizer.zero_grad()
                Y_hat = torch.squeeze(self.gnn(X))
                # loss = self.squared_loss(Y_hat, Y_torch)
                Y_hat = Y_hat.view(Y.shape)
                mse = self.mse_fn(Y_hat, Y)
                h_val = self.gnn.h_func()
                penalty = 0.5 * rho * h_val * h_val + alpha * h_val
                l2_reg = 0.5 * lambda2 * self.gnn.l2_reg()
                l1_reg = lambda1 * self.gnn.l1_reg()
                primal_obj = mse + penalty + l2_reg + l1_reg
                primal_obj.backward()
                return primal_obj
            

            self.optimizer.step(closure)  # NOTE: updates model in-place
            # print('step')
            with torch.no_grad():
                h_new = self.gnn.h_func().item()
            if h_new > 0.25 * h:
                rho *= 10
            else:
                break
        alpha += rho * h_new
        return rho, alpha, h_new
    
    def train(self,
                      X: np.ndarray,
                      lambda1: float = 0.,
                      lambda2: float = 0.,
                      max_iter: int = 100,
                      h_tol: float = 1e-8,
                      rho_max: float = 1e+10,
                      lag: int = 5):
        rho, alpha, h = 1.0, 0.0, np.inf
        tr_Y = X[:, lag:].transpose()
        #tr_Y =torch.transpose(X[:, lag:], 0, 1)
        tr_X = np.array([X[:,  i:i+lag] for i in range(tr_Y.shape[0])])
        tr_X = torch.tensor(tr_X, device = device)
        tr_Y = torch.tensor(tr_Y, device = device )

        for _ in tqdm(range(max_iter)):
            rho, alpha, h = self.dual_ascent_step(tr_X, tr_Y, lambda1, lambda2,
                                        rho, alpha, h, rho_max)
            print(h)
            print(rho)
            if h <= h_tol or rho >= rho_max:
                break
        with torch.no_grad():
            W_est = self.gnn.get_adj()
            Y_hat = torch.squeeze(self.gnn(tr_X))
                # print('Y_hat', Y_hat.shape)
            loss = self.mse_fn(Y_hat, tr_Y)
            #print('MSE:', loss.item())
        return W_est
