import os.path
import torch
import time 
from torch_geometric.data import Data
import numpy as np
import pandas as pd
from pyspot import DSpot, Spot
from pingouin import partial_corr
from sklearn import preprocessing
from read_utils import read_aiops_data_new
import scipy.stats as st
import statistics
import bisect
from datetime import datetime
import math

class Color:
    GREEN = '\033[92m'  # Green color
    RED = '\033[91m'    # Red color
    END = '\033[0m'     # Reset to default color


def spot_detection(X, d: int=10, q: float=1e-4, n_init:int=100, level:float=0.95)->np.ndarray:
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
                        upper_threshold = 0.00000001
                    st = (xt - upper_threshold) / upper_threshold
                # down alert
                if event == -1:
                    lower_threshold = S.status().to_dict()['z_down']
                    assert(xt <= lower_threshold)
                    if lower_threshold == 0:
                        lower_threshold = 0.00000001
                    st = (lower_threshold - xt) / lower_threshold
                st = np.abs(st)
            score.append(st)
        score_list.append(score)
    np_score = np.array(score_list).transpose()
    return np_score


def detect_individual_causal(X: np.ndarray,
                   method:str='SPOT',
                   args:dict={'d': 10, 'q': 1e-4, 'n_init': 100, 'level':0.95})->np.ndarray:
    if method == 'SPOT':
        d = args['d']
        q = args['q']
        n_init = args['n_init']
        level = args['level']
        score = spot_detection(X, d, q, n_init, level)
    return score


def generate_Q(X: np.ndarray, W: np.ndarray, RI: int, rho: float, columns: list = None):
    n = W.shape[0]
    if columns is None:
        columns = ['V{}'.format(i) for i in range(n)]
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
        PAak_minus_i = [c for c in PAak if c != columns[i]]
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
    rsum = np.sum(Q, axis=1).reshape(-1, 1)
    rsum[rsum == 0] = 1
    Q = Q / rsum
    return Q

# random walk with restart
def propagate_error(Q: np.ndarray, start: int, steps: int = 50000, rp: float = 0.05, max_self: int = 10) -> np.ndarray:
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

def topological_root_cause_analysis(ts, dag, sensors, steps=50000, compressed_data_size=1000):
    if ts.shape[0] > compressed_data_size:
        patch = int(ts.shape[0]/compressed_data_size)
        ts = ts[:patch*(ts.shape[0]//patch),:]
        ts = np.sum(ts.reshape((-1, patch, ts.shape[1])),axis=1) #summary information to improve efficiency
        # ts_metric = ts[:,:-1]
        ts = preprocessing.normalize(ts,axis=0, norm='l1')
        # ts = np.append(ts_metric,ts[:,-1].reshape(-1,1),axis=1)

    Q = generate_Q(ts, dag, RI=dag.shape[0]-1, rho=1e-2)
    score = propagate_error(Q, start=dag.shape[0]-1, steps=steps)
    score /= steps

    normal_score =  preprocessing.normalize([score[:-1]],norm='l1').ravel()

    #rcs = pd.concat([pd.DataFrame(np.asarray(sensors).reshape(-1,1))
    #                ,pd.DataFrame(np.asarray(score[:-1]).reshape(-1,1))],axis=1)

    rcs = pd.concat([pd.DataFrame(np.asarray(sensors).reshape(-1,1))
                    ,pd.DataFrame(np.asarray(normal_score).reshape(-1,1))],axis=1)

    rcs.columns = ['sensor', 'top_score']
    # rcs.sort_values(by='score', ascending=False, inplace=True)
    return rcs

#def individual_root_cause_analysis(data, sensors, patch=100):
#    data = data.numpy()
#    data = data[:patch*(data.shape[0]//patch),:]
#    data = np.sum(data.reshape((-1,patch,data.shape[1])),axis=1)


def individual_root_cause_analysis(data, sensors, compressed_data_size):
    data = data.numpy()

    if data.shape[0] > compressed_data_size:
        patch = data.shape[0]//compressed_data_size
        data = data[:patch*(data.shape[0]//patch),:]
        data = np.sum(data.reshape((-1,patch,data.shape[1])),axis=1)
    
    data = preprocessing.normalize(data, axis=0, norm = 'l1')

    #0606 only log freq and gs has individual scores, other metrics are 0. So the performance is bad.
    # ind_causal_score = detect_individual_causal(data[:,:-1], method='SPOT',
    #                 args={'d': 50, 'q': 1e-3, 'n_init': 500, 'level': 0.9})

  
    ind_causal_score = detect_individual_causal(data[:,:-1], method='SPOT', args={'d': 10, 'q': 1e-4, 'n_init': 100, 'level': 0.95})
    
    ind_causal_score = np.sum(ind_causal_score,axis=0)

    normal_ind_causal_score =  preprocessing.normalize([ind_causal_score],norm='l1').ravel()

    rcs = pd.concat([pd.DataFrame(np.asarray(sensors).reshape(-1,1))
                    ,pd.DataFrame(np.asarray(normal_ind_causal_score).reshape(-1,1))],axis=1)
    rcs.columns = ['sensor', 'ind_score']
    return rcs


def integrate_top_ind_rca(t_rca, i_rca, data, alpha=0.4):
    results = pd.merge(left=t_rca,right=i_rca,on='sensor',how='inner')
    results['score'] = alpha * results['top_score'] + (1-alpha) * results['ind_score']
    variances = np.std(data.numpy()[:,:-1],axis=0)
    rcs_var = pd.concat([pd.DataFrame(results['sensor']),
                         pd.DataFrame(variances.reshape(-1,1))],axis=1)
    rcs_var.columns = ['sensor','variance']
    final = pd.merge(left=results,right=rcs_var,on='sensor',how='inner')
    final = final[final['variance']>1e-5]
    final = final[['sensor','score']]
    final.sort_values(by='score', ascending=False, inplace=True)
    return final


def locate_topk_root_causes(total_data, ts, dag, sensors, alpha=0.4):
    t_rca = topological_root_cause_analysis(ts, dag, sensors)
    i_rca = individual_root_cause_analysis(total_data, sensors)
    f_rca = integrate_top_ind_rca(t_rca, i_rca, total_data, alpha)

    return f_rca

def evaluation_percentile(rcs, real_rc):
    rcs = list(rcs['sensor'])
    return rcs.index(real_rc)/len(rcs)*100

def get_seq_data(ts, time_window, t, time_overlap):
    s1_data = ts[:20000,:]
    s2_data = ts[t-time_overlap: t+(time_window-time_overlap),:]
    s1_data, s2_data = torch.from_numpy(s1_data).float(), torch.from_numpy(s2_data).float()
    return s1_data, s2_data


def adj_matrix_to_edge_list(adj_matrix):
    edge_list = []
    for i in range(adj_matrix.shape[0]):
        for j in range(adj_matrix.shape[1]):
            if adj_matrix[i,j] != 0:
                edge_list.append([i,j])
    edge_list = np.asarray(edge_list).T
    edge_list = torch.from_numpy(edge_list).long()
    return edge_list

def get_graph_data(x, adj_matrix):
    edge_list = adj_matrix_to_edge_list(adj_matrix)
    data = Data(x=x, edge_index=edge_list)
    return data

def generate_graph_by_thre(graph, thre):
    matrix_size = len(graph.reshape(-1))
    if (matrix_size < 50):
        threshold_factor = 0.6
    else: 
        threshold_factor = 0.3

    K =  math.ceil(threshold_factor*len(graph.reshape(-1)))
    threshold = sorted(graph.reshape(-1), reverse=True)[K-1] 
    threshold = max(thre, threshold)
    graph_binary = np.where(graph>threshold, 1, 0)
    return graph_binary.astype(int)

def load_metric_dags(dag_path,metrics, percentile):
    metric_dags = {x:'' for x in metrics}
    for metric in metrics:
        dag_matrix = np.load(dag_path+metric+'_dag.npy',allow_pickle=True)
        dag_matrix[dag_matrix>=np.percentile(dag_matrix,percentile)] = 1
        dag_matrix[dag_matrix<np.percentile(dag_matrix, percentile)] = 0
        metric_dags[metric] = dag_matrix

    return metric_dags

def add_missing_sensors(df, sensors):
    missing_sensors = set(sensors) - set(df['sensor'])
    mean_score = df['score'].astype(float).mean()
    missing_rows = pd.DataFrame({'sensor': list(missing_sensors), 'score': [mean_score] * len(missing_sensors)})
    return pd.concat([df, missing_rows])

def integrate_ind_top_causal_score_with_mean(weighted_results, ind_scores,top_scores, alpha, metric_sensors, topK_metrics_num):
    total_scores = {}
    weighted_results = dict(zip(weighted_results['metrics'],weighted_results['score']))
    top_K_items = sorted(weighted_results.items(), key=lambda x: x[1], reverse=True)[:topK_metrics_num]
    # If you want to convert it back to a dictionary
    weighted_results = dict(top_K_items)
    
    for metric in weighted_results.keys():
        if metric in top_scores.keys():
            total_scores[metric] = float(weighted_results[metric])*(alpha * ind_scores[metric] + (1-alpha) * top_scores[metric])

    
    total_data_frame = {}
    for metric in weighted_results.keys():
        if metric in total_scores.keys():
            data_frame = np.concatenate((
                np.asarray(metric_sensors[metric]).reshape(-1,1),
                np.asarray(total_scores[metric]).reshape(-1,1)
            ),axis=1)
            data_frame = pd.DataFrame(data_frame)
            data_frame.columns = ['sensor','score']
            total_data_frame[metric] = data_frame
    
    all_sensors = pd.unique(pd.concat(df['sensor'] for _, df in total_data_frame.items()))

    for metric, df in total_data_frame.items():
        total_data_frame[metric] = add_missing_sensors(df, all_sensors)
        total_data_frame[metric] = total_data_frame[metric].sort_values(by='sensor').reset_index(drop=True)
    
    all_data = pd.concat(total_data_frame.values())

    all_data['score'] = all_data['score'].astype(float)

    mean_scores = all_data.groupby('sensor')['score'].mean().reset_index()

    mean_scores['score'] = pd.Series(preprocessing.normalize([mean_scores['score']],norm='l1').ravel())

    mean_scores.columns = ['sensor','final_score']
    mean_scores.sort_values(by='final_score', ascending=False, inplace=True)
    return mean_scores

def integrate_ind_top_causal_score(weighted_results, ind_scores,top_scores, alpha, metric_sensors, topK_metrics_num):
    total_scores = {}
    weighted_results = dict(zip(weighted_results['metrics'],weighted_results['score']))
    top_K_items = sorted(weighted_results.items(), key=lambda x: x[1], reverse=True)[:topK_metrics_num]
    # If you want to convert it back to a dictionary
    weighted_results = dict(top_K_items)
    
    for metric in weighted_results.keys():
        if metric in top_scores.keys():
            total_scores[metric] = float(weighted_results[metric])*(alpha * ind_scores[metric] + (1-alpha) * top_scores[metric])
    
    total_data_frame = {}
    for metric in weighted_results.keys():
        if metric in total_scores.keys():
            data_frame = np.concatenate((
                np.asarray(metric_sensors[metric]).reshape(-1,1),
                np.asarray(total_scores[metric]).reshape(-1,1)
            ),axis=1)
            data_frame = pd.DataFrame(data_frame)
            data_frame.columns = ['sensor','score']
            total_data_frame[metric] = data_frame
    
    # Step 1: Concatenate all dataframes in the dictionary
    all_data = pd.concat(total_data_frame.values())
    all_data['score'] = all_data['score'].astype(float)

    # Step 2: Group by the "sensor" column
    grouped_data = all_data.groupby('sensor')

    # Step 3: Calculate the sum and count of scores for each sensor
    summary_stats = grouped_data.agg(
        total_score=pd.NamedAgg(column='score', aggfunc='sum'),
        count=pd.NamedAgg(column='score', aggfunc='count')
    )

    # Step 4: Calculate the mean score for each sensor
    final_score = summary_stats['total_score'] / summary_stats['count']
    final_score = preprocessing.normalize([final_score],norm='l1').ravel()
    # final_score = scaler.fit_transform(final_score)
    result = pd.concat((pd.DataFrame(grouped_data.indices.keys()),pd.DataFrame(final_score.reshape(-1,1))),axis=1)
    result.columns = ['sensor','final_score']
    result.sort_values(by='final_score', ascending=False, inplace=True)
    return result


    # final_score = ''
    # for i, metric in enumerate(total_scores.keys()):
    #     if i == 0:
    #         final_score = np.pad(total_scores[metric], (0, int(max_pod_len-total_scores[metric].shape[0])),'constant')
    #     else:
    #         final_score += np.pad(total_scores[metric], (0, int(max_pod_len-total_scores[metric].shape[0])),'constant')

    # final_score /= len(weighted_results)
    # final_score = preprocessing.normalize([final_score],norm='l1').ravel()
    
    # return result
    

def get_current_batch_data(start_time, time_window, time_stamps, ts, online_start_time):
    batch_start_time = start_time
    batch_end_time = start_time + time_window

    indices = np.where((time_stamps>=batch_start_time) & (time_stamps <= batch_end_time))
    offline_ind = np.where(time_stamps>=online_start_time)[0][0]
    if indices[0].size > 0:
        # online_kpi = ts[offline_ind:,-1]
        # online_time = time_stamps[offline_ind:]
        # stacked_array = np.column_stack((online_time,online_kpi))
        # stacked_df = pd.DataFrame(stacked_array, columns=['timestampe','kpi'])
        # stacked_df.to_csv("online_data.csv",index=False)
        # print(len(indices[0]))

        batch_start_index, batch_end_index = indices[0][0], indices[0][-1]
        flag = True
        s1_data = ts[:offline_ind, :]
        s2_data = ts[batch_start_index:batch_end_index, :]
        return flag, s1_data, s2_data
    else:
        flag = False
        return flag, None, None


def calculate_prioritized_metric_weights(metric_weights, valid_metrics):
    weight_results = preprocessing.normalize(metric_weights, axis=0, norm='l1').ravel().reshape(-1, 1)
    weight_results = pd.DataFrame(np.hstack((np.array(valid_metrics).reshape(-1, 1), weight_results)))
    weight_results.columns = ['metrics', 'score']
    weight_results.sort_values(by='score', inplace=True, ascending=False)
    print('Prioritized Metric Order with Learned Weights:')
    for m_i in range(weight_results.shape[0]):
        print('{}: {}, {}'.format(m_i + 1, weight_results.iloc[m_i][0], weight_results.iloc[m_i][1]))
    print('Causal graph is updated by the prioritized metrics order!')
    return weight_results


def calculate_prioritized_metric_weights_wtn(metric_weights, valid_metrics,logger):
    weight_results = preprocessing.normalize(metric_weights, axis=0, norm='l1').ravel().reshape(-1, 1)
    weight_results = pd.DataFrame(np.hstack((np.array(valid_metrics).reshape(-1, 1), weight_results)))
    weight_results.columns = ['metrics', 'score']
    weight_results.sort_values(by='score', inplace=True, ascending=False)
    logger.info('Prioritized Metric Order with Learned Weights:')
    for m_i in range(weight_results.shape[0]):
        logger.info('{}: {}, {}'.format(m_i + 1, weight_results.iloc[m_i][0], weight_results.iloc[m_i][1]))
    logger.info('Causal graph is updated by the prioritized metrics order!')
    return weight_results


def get_current_top_batch_data(metric_batch_data, metric, patch, device):
    if len(metric_batch_data[metric]) >= 2:
        s1_batch_data, s2_batch_data = metric_batch_data[metric][0], metric_batch_data[metric][1]
        s1_batch_remainder, s2_batch_remainder = (s1_batch_data.shape[0]%patch), (s2_batch_data.shape[0]%patch)
        s1_batch_data = s1_batch_data if s1_batch_remainder== 0 else s1_batch_data[:-(s1_batch_data.shape[0]%patch),:]
        s2_batch_data = s2_batch_data if s2_batch_remainder== 0 else s2_batch_data[:-(s2_batch_data.shape[0]%patch),:]
        s1_batch_data = np.sum(s1_batch_data.reshape((-1, patch, s1_batch_data.shape[1])), axis=1)
        s2_batch_data = np.sum(s2_batch_data.reshape((-1, patch, s2_batch_data.shape[1])), axis=1)
        if len(s2_batch_data) == 0:
            return None, None
        #new
        #s1_batch_data = preprocessing.normalize(s1_batch_data, axis=0, norm='l1')
        #s2_batch_data = preprocessing.normalize(s2_batch_data, axis=0, norm='l1')
        s1_batch_data, s2_batch_data = torch.from_numpy(s1_batch_data).float().to(device), torch.from_numpy(
            s2_batch_data).float().to(device)
        return s1_batch_data, s2_batch_data
    else:
        return None, None

def make_dir(save_path):
    save_path = save_path.strip()
    save_path = save_path.rstrip()
    isExists = os.path.exists(save_path)
    if not isExists:
        os.makedirs(save_path)
        print(save_path+" is created successfully!")
    else:
        print(save_path+" has been created!")





def log_euclidean(a,b):
  return np.linalg.norm(np.log(a)-np.log(b))

def ts_to_cov(x):
  res = np.zeros(shape = [x.shape[1], x.shape[1]])
  for i in range(x.shape[0]):
    xi = x[i].reshape(x.shape[1], 1)
    res += xi@xi.T
  return res/x.shape[0]


def dist_cho(x, y, keepdims=False):
  x_diag, x_lower = _diag_and_strictly_lower(x)
  y_diag, y_lower = _diag_and_strictly_lower(y)
  lower = torch.linalg.norm(
      x_lower - y_lower, axis=[-2, -1], ord="fro", keepdims=keepdims
  )
  diag = torch.linalg.norm(
      torch.log(x_diag) - torch.log(y_diag),
      axis=-1,
      keepdims=keepdims,
  )
  return torch.sqrt(lower**2 + diag.reshape(lower.shape)**2)

def _diag_and_strictly_lower(x):
  x_lower = torch.tril(x, -1, 0)
  diag = torch.diagonal(x, dim1=-1, dim2=-1)
  return diag, x_lower - torch.diag_embed(diag)

def log_cholesky(x,y):
  x_chol = torch.linalg.cholesky(x)
  y_chol = torch.linalg.cholesky(y)
  res = dist_cho(x_chol, y_chol)
  return res

def detect_cusum(x, threshold=1, drift=0, ending=False, show=True, ax=None):
    """Cumulative sum algorithm (CUSUM) to detect abrupt changes in data.
    Parameters
    ----------
    x : 1D array_like
        data.
    threshold : positive number, optional (default = 1)
        amplitude threshold for the change in the data.
    drift : positive number, optional (default = 0)
        drift term that prevents any change in the absence of change.
    ending : bool, optional (default = False)
        True (1) to estimate when the change ends; False (0) otherwise.
    show : bool, optional (default = True)
        True (1) plots data in matplotlib figure, False (0) don't plot.
    ax : a matplotlib.axes.Axes instance, optional (default = None).
    Returns
    -------
    ta : 1D array_like [indi, indf], int
        alarm time (index of when the change was detected).
    tai : 1D array_like, int
        index of when the change started.
    taf : 1D array_like, int
        index of when the change ended (if `ending` is True).
    amp : 1D array_like, float
        amplitude of changes (if `ending` is True).
    Notes
    -----
    """

    x = np.atleast_1d(x).astype('float64')
    gp, gn = np.zeros(x.size), np.zeros(x.size)
    ta, tai, taf = np.array([[], [], []], dtype=int)
    tap, tan = 0, 0
    amp = np.array([])
    # Find changes (online form)
    for i in range(1, x.size):
        s = x[i] - x[i-1]
        gp[i] = gp[i-1] + s - drift  # cumulative sum for + change
        gn[i] = gn[i-1] - s - drift  # cumulative sum for - change
        if gp[i] < 0:
            gp[i], tap = 0, i
        if gn[i] < 0:
            gn[i], tan = 0, i
        if gp[i] > threshold or gn[i] > threshold:  # change detected!
            ta = np.append(ta, i)    # alarm index
            tai = np.append(tai, tap if gp[i] > threshold else tan)  # start
            gp[i], gn[i] = 0, 0      # reset alarm
    # THE CLASSICAL CUSUM ALGORITHM ENDS HERE

    # Estimation of when the change ends (offline form)
    if tai.size and ending:
        _, tai2, _, _ = detect_cusum(x[::-1], threshold, drift, show=False)
        taf = x.size - tai2[::-1] - 1
        # Eliminate repeated changes, changes that have the same beginning
        tai, ind = np.unique(tai, return_index=True)
        ta = ta[ind]
        # taf = np.unique(taf, return_index=False)  # corect later
        if tai.size != taf.size:
            if tai.size < taf.size:
                taf = taf[[np.argmax(taf >= i) for i in ta]]
            else:
                ind = [np.argmax(i >= ta[::-1])-1 for i in taf]
                ta = ta[ind]
                tai = tai[ind]
        # Delete intercalated changes (the ending of the change is after
        # the beginning of the next change)
        ind = taf[:-1] - tai[1:] > 0
        if ind.any():
            ta = ta[~np.append(False, ind)]
            tai = tai[~np.append(False, ind)]
            taf = taf[~np.append(ind, False)]
        # Amplitude of changes
        amp = x[taf] - x[tai]

    return ta, tai, taf, amp


def get_cpd_time_seq(node_data, data_sample, mul_kpi, label, start_time):
   if data_sample in ['0517','0524']:
      node_time_seq = node_data[label]['Sequence'][:,:-1]
   else:
      node_time_seq = node_data[label]['Sequence']
   node_time_stamp = pd.DataFrame(node_data[label]['time'])
   node_time_stamp.columns = ['timeStamp']
   node_seqs = pd.concat((node_time_stamp, pd.DataFrame(node_time_seq)),axis=1)
   if data_sample in ['0517','0524']:
      node_df = pd.merge(node_seqs, mul_kpi, on='timeStamp', how='inner')
   else:
      node_df = node_seqs 
   ts = node_df.values[:,1:] 
   cpd_target = np.argmin(np.abs(np.array(node_df['timeStamp'])-start_time))
   time_len = node_df['timeStamp'].shape[0]
   time_stamps = node_df['timeStamp']
   return ts, cpd_target, time_len, time_stamps

def system_data_load(data_sample, config_path, node_data_path, kpi_path):
    data, metrics, real_rc, label, online_start, failure_time = read_aiops_data_new(config_path,'all')
    node_data = np.load(node_data_path, allow_pickle=True).item()
    if data_sample in ['0517','0524']:
      mul_kpi = pd.read_csv(kpi_path)[['timeStamp','Latency','Error_rate']]
    else:
      mul_kpi = [] 
    return data, metrics, real_rc, label, online_start, node_data, mul_kpi, failure_time

def distance_calculation_for_conceutive_iterations(ts, cpd_start, cpd_window_size, cpd_stride, cpd_manifold_stride):
    cpd_first_win = ts[cpd_start:cpd_start+cpd_window_size,:]
    cpd_second_win = ts[cpd_start+cpd_manifold_stride:cpd_start+cpd_window_size+cpd_manifold_stride,:]

    cpd_first_win = np.mean(cpd_first_win[:cpd_first_win.shape[0]//cpd_stride*cpd_stride].reshape(-1,cpd_stride,cpd_first_win.shape[1]), axis=1)
    cpd_second_win = np.mean(cpd_second_win[:cpd_second_win.shape[0]//cpd_stride*cpd_stride].reshape(-1,cpd_stride,cpd_second_win.shape[1]), axis=1)

    batch_tensor = ts_to_cov(cpd_first_win)
    batch_tensor_next = ts_to_cov(cpd_second_win)
    dist0 = log_euclidean(batch_tensor + 1, batch_tensor_next + 1)

    return dist0

def change_point_detection_cusum(dist_list, beta = 2, gamma = 5):
    dist_mean = np.mean(dist_list)
    dist_std = np.std(dist_list)
    threshold = beta*dist_mean+gamma*dist_std

    dist_array_list = np.asarray(dist_list)
    dist_array_list[np.isnan(dist_array_list) | np.isinf(dist_array_list)] = 0
    alarm_time, early_time, end_time, _ = detect_cusum(dist_array_list, threshold=threshold, drift=0, ending=False, show=False, ax=None)

    return alarm_time


def update_total_data(metrics,start_time, time_window, data, label, metric_batch_data, metric_total_data, online_start, offline_length, is_cpd, cpd_time_freq, stride_len):
    for metric in metrics:
        flag, s1_data, s2_data = get_current_batch_data(start_time, time_window, np.asarray(data[metric][label]['time']), data[metric][label]['Sequence'],online_start)
        if flag == False: # if the current batch data is invalid
            continue
        metric_batch_data[metric].append(s1_data)
        metric_batch_data[metric].append(s2_data)
        if len(metric_total_data[metric])==0:
            metric_total_data[metric] = torch.cat((torch.from_numpy(s1_data).float(),torch.from_numpy(s2_data).float()))
        else:
            if not is_cpd:
                metric_total_data[metric] = torch.cat((metric_total_data[metric],torch.from_numpy(s2_data[-cpd_time_freq:,:]).float())) 
            else:
                metric_total_data[metric] = torch.cat((metric_total_data[metric],torch.from_numpy(s2_data[-stride_len:,:]).float()))

    for metric in metrics:
        metric_total_data[metric] = metric_total_data[metric][-offline_length:,:]

    return metric_total_data, metric_batch_data

def initial_causal_model(metrics, metric_batch_data, learning_rate, lag, exp_name, device, metric_models, metric_optimizer, mineral_Model, compressed_data_size):
    metric_batch_patch = {}
    for metric in metrics:
        if len(metric_batch_data[metric]) > 0: 
            #if metric == 'log_freq' or metric == 'log_gs':
            #    dj_model = mineral_Model(metric_batch_data[metric][0].shape[0]//log_patch, metric_batch_data[metric][1].shape[0]//log_patch, lag=lag, exp_name=exp_name).to(device)
            #else:
            if metric_batch_data[metric][0].shape[0] > compressed_data_size and metric_batch_data[metric][1].shape[0] > compressed_data_size:
                patch = min(metric_batch_data[metric][0].shape[0], metric_batch_data[metric][1].shape[0])//compressed_data_size
                metric_batch_patch[metric] = patch
                dj_model = mineral_Model(metric_batch_data[metric][0].shape[0]//patch, metric_batch_data[metric][1].shape[0]//patch, lag=lag, exp_name=exp_name).to(device)
            else:
                dj_model = mineral_Model(metric_batch_data[metric][0].shape[0], metric_batch_data[metric][1].shape[0], lag=lag, exp_name=exp_name).to(device)
                metric_batch_patch[metric] = 1
            metric_models[metric] = dj_model
            optimizer = torch.optim.Adam(dj_model.parameters(), lr=learning_rate)
            metric_optimizer[metric] = optimizer
        else:
            metric_batch_patch[metric] = 1
    
    return metric_models, metric_optimizer, metric_batch_patch


def confidence_generation(data: list):
    n_data = len(data)
    #create 99% confidence interval for population mean weight
    if n_data < 30:
        norm_max = st.t.interval(alpha=0.99, df=len(data[1:])-1, loc=np.mean(data[1:]), scale=st.sem(data[1:])) 
    else:
        norm_max = st.norm.interval(alpha=0.99, loc=np.mean(data[1:]), scale=st.sem(data[1:])) 

    confidence_interval = []
    confidence = []
    std = statistics.stdev(data[1:])

    for i in range(9):
        confidence_interval.append(0.1*(i+1)*(norm_max[1]+4*std))
    
    confidence = 0.1*bisect.bisect(confidence_interval,data[0])
    
    return confidence


def individual_causal_analysis_with_multiple_metrics(metrics, ind_scores, metric_total_data, data, log_compressed_data_size, metric_compressed_data_size, label, metric_weights, valid_metrics, ind_t0, count, ind_saving_path, is_save):
    for metric in metrics:
        sensors = data[metric][label]['Pod_Name']
        if metric == "log_freq" or metric == "log_gs":
           # print("ind log", metric_total_data[metric].shape)
            irc_score = individual_root_cause_analysis(metric_total_data[metric], sensors, log_compressed_data_size)
        else:
        #    print("ind metrics", metric_total_data[metric].shape)
            irc_score = individual_root_cause_analysis(metric_total_data[metric], sensors, metric_compressed_data_size)
        # irc_score['ind_score'] = scaler.fit_transform(irc_score['ind_score'].to_numpy().reshape(-1,1))
        if is_save:
            irc_score.to_csv(f"{ind_saving_path}{metric}_individual_causal_score_batch_{count}.csv",index=False)
        ind_scores[metric] = np.array(irc_score['ind_score'])
        ind_scores[metric] = np.nan_to_num(ind_scores[metric])
        metric_weights.append([np.max(ind_scores[metric], axis=0)])
        valid_metrics.append(metric)
    weight_metrics_results = calculate_prioritized_metric_weights(metric_weights, valid_metrics)
    print(f"The time cost of current batch for individual causal discovery is {(time.time()-ind_t0)/60} minutes!")
    return ind_scores, weight_metrics_results

def update_causal_model(metrics, metric_batch_data, device, metric_models, metric_optimizer, max_iter, metric_dags, lag, metric_iters):
    for metric in metrics:
        #if metric == 'log_freq' or metric == 'log_gs':
        #    s1_batch_data, s2_batch_data = get_current_top_batch_data(metric_batch_data, metric, log_patch, device)
        #else:
        s1_batch_data, s2_batch_data = get_current_top_batch_data(metric_batch_data, metric, device)

        dj_model = metric_models[metric]
        optimizer = metric_optimizer[metric]
        
        if s1_batch_data == None and s2_batch_data == None:
            continue
        for iter_ind in range(max_iter):
            optimizer.zero_grad()
            new_dag, loss, invariant_loss, dependent_loss, invariant_pred_loss, dependent_pred_loss\
                = dj_model(s1_batch_data.T,s2_batch_data.T,metric_dags[metric],lag)
            # print("current {} batch {} epoch, loss {}".format(count, iter_ind, loss.item()))
            loss.backward()
            optimizer.step()
            metric_iters[metric] += iter_ind
        causal_graph = generate_graph_by_thre(new_dag.cpu().detach().numpy())
        metric_dags[metric] = causal_graph
    return metric_dags


def topological_root_cause_analysis_with_multiple_metrics(metrics, metric_batch_data, metric_batch_patch, device, data, label, metric_models, metric_optimizer, max_iter, metric_dags, metric_iters, lag, metric_total_data, top_scores, count, is_save, top_saving_path, propagate_steps, topology_compressed_data_size):
    for metric in metrics:
        #if metric == 'log_freq' or metric == 'log_gs':
            # continue
        #    print("log min", np.array(metric_batch_data[metric]).shape)
        #    s1_batch_data, s2_batch_data = get_current_top_batch_data(metric_batch_data, metric, log_patch, device)
        #    print("log min", s1_batch_data.shape,s2_batch_data.shape)

        #else:
            # continue
            #print("metric min", np.array(metric_batch_data[metric]).shape)
        patch = metric_batch_patch[metric]
        s1_batch_data, s2_batch_data = get_current_top_batch_data(metric_batch_data, metric, device, patch)
            #print("metric min", s1_batch_data.shape,s2_batch_data.shape)


        sensors = data[metric][label]['Pod_Name']
        dj_model = metric_models[metric]
        optimizer = metric_optimizer[metric]
        if s1_batch_data == None and s2_batch_data == None:
            continue
        for iter_ind in range(max_iter):
            optimizer.zero_grad()
            if np.all(metric_dags[metric] == 0):
                continue
            
            new_dag, loss, invariant_loss, dependent_loss, invariant_pred_loss, dependent_pred_loss\
                = dj_model(s1_batch_data.T,s2_batch_data.T,metric_dags[metric],lag)

            # print("current {} batch {} epoch, loss {}".format(count, iter_ind, loss.item()))
            loss.backward()
            optimizer.step()
            metric_iters[metric] += iter_ind
            dj_model.writer.add_scalar(metric+' total loss', loss, metric_iters[metric])
            dj_model.writer.add_scalar(metric+' invariant loss', invariant_loss, metric_iters[metric])
            dj_model.writer.add_scalar(metric+' dependent loss', dependent_loss, metric_iters[metric])
            dj_model.writer.add_scalar(metric+' invariant pred loss', invariant_pred_loss, metric_iters[metric])
            dj_model.writer.add_scalar(metric+' dependent pred loss', dependent_pred_loss, metric_iters[metric])

   
        #if np.all(metric_dags[metric] == 0):
        # Check if new_dag is a PyTorch tensor
        if isinstance(new_dag, torch.Tensor):
        # If it's a PyTorch tensor, move it to CPU and convert to NumPy array
            new_dag_numpy = new_dag.cpu().detach().numpy()
        else:
        # If it's a NumPy array, simply assign it
            new_dag_numpy = new_dag

        # Check if all elements of new_dag_numpy are zeros
        if np.all(new_dag_numpy == 0): 
        #if np.all(new_dag.cpu().detach().numpy() == 0):
        #if np.all(new_dag.cpu().detach().numpy()  == 0):  
            causal_graph = metric_dags[metric]
        else:
            causal_graph = generate_graph_by_thre(new_dag_numpy)

        #print("topo total", metric_total_data[metric].shape)

        trc_score = topological_root_cause_analysis(metric_total_data[metric].numpy(), causal_graph, sensors, steps=propagate_steps, compressed_data_size = topology_compressed_data_size)
        # trc_score['top_score'] = scaler.fit_transform(trc_score['top_score'])

        if is_save:
            trc_score.to_csv(f"{top_saving_path}{metric}_topological_causal_score_batch_{count}.csv",index=False)
        top_scores[metric] = np.array(trc_score['top_score'])
        metric_dags[metric] = causal_graph

    return top_scores, metric_dags

def early_output_integration_result(weight_metrics_results, ind_scores,top_scores, alpha, metric_sensors, count, real_rc, ranked_list, is_save, inte_saving_path, start_time, topK_metrics_num, padding_way):
    if padding_way == "mean":
        final_results = integrate_ind_top_causal_score_with_mean(weight_metrics_results, ind_scores,top_scores, alpha, metric_sensors, topK_metrics_num)
    if padding_way == "zero":
        final_results = integrate_ind_top_causal_score(weight_metrics_results, ind_scores,top_scores, alpha, metric_sensors, topK_metrics_num)
    
    print(f"{Color.GREEN}Early detection result is as follows:{Color.END}")
    print(f"{Color.GREEN}The detected time of the early detection is {datetime.utcfromtimestamp(start_time+ 3600*9).strftime('%Y-%m-%d %H:%M:%S')}{Color.END}")
    print(f"{Color.GREEN}current {count} batch, top 10 result is {final_results.iloc[:10]}{Color.END}")
    

    confidence_score = 0

    if len(list(final_results['final_score'])) <= 3:
        print(f"{Color.GREEN}The number of entities are too few to estimate the confidence{Color.END}")
    else:
        confidence_score = confidence_generation(list(final_results['final_score']))
        print(f'{Color.GREEN}Root cause confidence: {confidence_score}{Color.END}')

    confidence = pd.DataFrame([confidence_score], columns = ['confidence'])

    if is_save:
        final_results.to_csv(f"{inte_saving_path}_early_detection_integrate_causal_score_batch_{count}.csv",index=False)
        confidence.to_csv(f"{inte_saving_path}_early_detection_integrate_confidence_score_batch_{count}.csv",index=False)

    rank_per = evaluation_percentile(final_results, real_rc)
    print(f"{Color.GREEN}rank percentile:{rank_per}%.{Color.END}")
    

def topological_individual_causal_integration(weight_metrics_results, ind_scores,top_scores, alpha, metric_sensors, count, real_rc, ranked_list, is_save, inte_saving_path, topK_metrics_num, padding_way):
    if padding_way == "mean":
        final_results = integrate_ind_top_causal_score_with_mean(weight_metrics_results, ind_scores,top_scores, alpha, metric_sensors, topK_metrics_num)
    if padding_way == "zero":
        final_results = integrate_ind_top_causal_score(weight_metrics_results, ind_scores,top_scores, alpha, metric_sensors,topK_metrics_num)
    print("current {} batch, top 10 result is {}".format(count,final_results.iloc[:10]))

    confidence_score = 0

    if len(list(final_results['final_score'])) <= 3:
        print("The number of entities are too few to estimate the confidence")
    else:
        confidence_score = confidence_generation(list(final_results['final_score']))
        print('Root cause confidence: ', confidence_score)

    confidence = pd.DataFrame([confidence_score], columns = ['confidence'])

    if is_save:
        final_results.to_csv(f"{inte_saving_path}_integrate_causal_score_batch_{count}.csv",index=False)
        confidence.to_csv(f"{inte_saving_path}_integrate_confidence_score_batch_{count}.csv",index=False)

    rank_per = evaluation_percentile(final_results, real_rc)
    print("rank percentile:{}%".format(rank_per))
    ranked_list.append(np.array(final_results['sensor']))
    return ranked_list
    
