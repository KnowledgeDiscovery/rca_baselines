import yaml
import numpy as np
import time
from sklearn import preprocessing
from causalnex.structure import dynotears
from read_utils import read_aiops_data


#### function edited from causalnex/causalnex/structure/dynotears.p
def from_numpy_dynamic(  # pylint: disable=too-many-arguments
    X: np.ndarray,
    Xlags: np.ndarray,
    lambda_w: float = 0.1,
    lambda_a: float = 0.1,
    max_iter: int = 100,
    h_tol: float = 1e-8,
    w_threshold: float = 0.0,
    tabu_edges = None,
    tabu_parent_nodes = None,
    tabu_child_nodes = None,):
    
    _, d_vars = X.shape
    p_orders = Xlags.shape[1] // d_vars

    bnds_w = 2 * [
        (0, 0)
        if i == j
        else (0, 0)
        if tabu_edges is not None and (0, i, j) in tabu_edges
        else (0, 0)
        if tabu_parent_nodes is not None and i in tabu_parent_nodes
        else (0, 0)
        if tabu_child_nodes is not None and j in tabu_child_nodes
        else (0, None)
        for i in range(d_vars)
        for j in range(d_vars)
    ]

    bnds_a = []
    for k in range(1, p_orders + 1):
        bnds_a.extend(
            2
            * [
                (0, 0)
                if tabu_edges is not None and (k, i, j) in tabu_edges
                else (0, 0)
                if tabu_parent_nodes is not None and i in tabu_parent_nodes
                else (0, 0)
                if tabu_child_nodes is not None and j in tabu_child_nodes
                else (0, None)
                for i in range(d_vars)
                for j in range(d_vars)
            ]
        )

    bnds = bnds_w + bnds_a
    w_est, a_est = dynotears._learn_dynamic_structure(
        X, Xlags, bnds, lambda_w, lambda_a, max_iter, h_tol
    )

    w_est[np.abs(w_est) < w_threshold] = 0
    a_est[np.abs(a_est) < w_threshold] = 0
    # sm = _matrices_to_structure_model(w_est, a_est)
    return w_est, a_est

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Dynotear algorithm')
    parser.add_argument('--dataset', type=str, default='20211203', help='name of the dataset')
    parser.add_argument('--path_dir', type=str, default='../../20211203/', help='path to the dataset')
    parser.add_argument('--output_dir', type=str, default='./20211203_output/', help='path to save the results')
    # Parse the arguments
    args = parser.parse_args()
    dataset = args.dataset
    dag_path = args.output_dir
    data_path = args.path_dir
    label = 'reviews-v3'
    if dataset == '20220606':
        offline_end_offset = 35000
        label = 'reviews-v3'
    elif dataset == '20210517':
        offline_end_offset = 30000
        label = 'Book_Info_product'
    elif dataset == '20211203':
        offline_end_offset = 15000
        label = 'ratings.book-info.svc.cluster.local:9080/*'
    elif dataset == '20210524':
        offline_end_offset = 60000
        label = 'Book_Info_product'
    else:
        raise NotImplementedError
    # data_path = '../Log_Metric_DAG/data'
    data_file = 'integrate_log_metric_data.pkl'
    # dag_path = '../Log_Metric_DAG/dag'
    #lag = 30
start_point = 20000
lag = 15
patch = 100

metric_data, metrics, _, label, _, _ = read_aiops_data('./config/data_config_{}_freq_gs.yaml'.format(dataset), 'all', data_path)
for metric in metrics:
    print('metric:', metric)
    t0 = time.time()
    data = metric_data[metric][label]['Sequence']
    prior_data = data[:start_point, :]
    prior_data = np.sum(prior_data.reshape(-1,patch,prior_data.shape[1]), axis=1)
    prior_data_metric = prior_data[:,:-1]
    prior_data_metric = preprocessing.normalize(prior_data_metric, axis=0, norm='l1')
    prior_data = np.append(prior_data_metric,prior_data[:,-1].reshape(-1,1),axis=1)
    prior_data_lag = np.roll(prior_data, 1, axis=0)
    for lag_o in range(2, lag + 1):
        prior_data_lag = np.hstack((prior_data_lag, np.roll(prior_data, lag_o, axis=0)))
    W_est, a_est = from_numpy_dynamic(prior_data, prior_data_lag, 1e-3, 1e-3, 500)
    print('W_est:', W_est.shape)
    print('a_est:', a_est.shape)
    np.save('./{}/'.format(dag_path) + metric+"_dag.npy", W_est)
    print("{} has been discovered its initial dag! Time cost is {}".format(metric,(time.time()-t0)))


