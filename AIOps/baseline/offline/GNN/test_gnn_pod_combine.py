import numpy as np
import pandas as pd
from sklearn import preprocessing
from rca import detect_individual_causal, generate_causal_graph, generate_Q, propagate_error
from sklearn.feature_selection import VarianceThreshold
import os

class SlidingWindow():
    def __init__(self):
        pass

    def sliding_window(self, time_data, event_data, window_size, step_size, scale=1, name='Latency'):
        """
        :param df: dataframe columns=[timestamp, label, eventid, time duration, lineid]
        :param window_size: seconds,
        :param step_size: seconds
        :return: dataframe columns=[eventids, label, eventids, time durations, lineids]
        """
        log_size = time_data.shape[0]
        print('Log size:', log_size)
        time_data = time_data//scale
        new_data = []
        start_end_index_pair = set()
        start_time = int(time_data[0])
        # start_time = 0
        end_time = start_time + window_size
        start_index = 0
        end_index = 0

        # get the first start, end index, end time
        for cur_time in time_data:
            if cur_time < end_time:
                end_index += 1
            else:
                break

        start_end_index_pair.add(tuple([start_index, end_index]))

        # move the start and end index until next sliding window
        num_session = 1
        while end_index < log_size:
            start_time = start_time + step_size
            end_time = start_time + window_size
            for i in range(start_index, log_size):
                if time_data[i] < start_time:
                    i += 1
                else:
                    break
            for j in range(end_index, log_size):
                if time_data[j] < end_time:
                    j += 1
                else:
                    break
            start_index = i
            end_index = j

            # when start_index == end_index, there is no value in the window
            if start_index != end_index:
                start_end_index_pair.add(tuple([start_index, end_index]))
                # print('Adding start_idx, end_idx=({},{})'.format(start_index, end_index))
            num_session += 1
            if num_session % 1000 == 0:
                print("process {} time window".format(num_session), end='\r')

        for (start_index, end_index) in start_end_index_pair:
            event_sequence = event_data[start_index: end_index].mean(axis=0)
            # for event in event_data[start_index: end_index]:
            #     if event not in event_sequence:
            #         event_sequence.append(event)
            # event_sequence = '. '.join(event_sequence)
            event_sequence = np.array(event_sequence)
            new_data.append([
                time_data[start_index],
                time_data[min(end_index, time_data.shape[0]-1)],
                event_sequence,
            ])
        new_df = pd.DataFrame(new_data, columns=['StartTime','EndTime', name])
        assert len(start_end_index_pair) == len(new_data)
        print('\nThere are %d instances (sliding windows) in this dataset' % len(start_end_index_pair))
        return new_df                                                                                                       


if __name__ == '__main__':
    #st = time.time()
    #Assign weight for each metric: default equal weight
    # POD_log_FILE = {'Score': 1}
    dataset = '0517'
    print('dataset: ', dataset)
    window_size = 30
    step_size = 0.5
    log_label = 'ratings.book-info.svc.cluster.local:9080/*'
    POD_METRIC_FILE = {'cpu_usage': 1, 'memory_usage': 1, 'rate_transmitted_packets': 1, 'rate_received_packets': 1, 'received_bandwidth': 1, 'transmit_bandwidth': 1, 'Score':1}
    # POD_combine_FILE = {'cpu_usage': 1, 'memory_usage': 1, 'rate_transmitted_packets': 1, 'rate_received_packets': 1, 'received_bandwidth': 1, 'transmit_bandwidth': 1, 'score':1}
    log_patch = 6
    log_data = {}
    columns_common = {}
    pathset = "./{}_output/".format(dataset)
    if not(os.path.exists(pathset)):
        os.mkdir(pathset)
    if dataset == '1203':
        log_file = 'backup_version/joint_training/1203_log_BERT_tokenize_template_pod_level_removed_30.npy'
        KPI_file = '/nfs/users/zach/aiops/data/1203/kpi.csv'
        metric_label = 'ratings.book-info.svc.cluster.local:9080/*'
    elif dataset == '0606':
        log_file = 'backup_version/joint_training/0606_log_frequency_pod_level_window_size_30_removed.npy'
        metric_label = 'reviews-v3'
        KPI_file = '/nfs/users/zach/aiops/data/0606/reviews-v3_Incoming_Success_Rate_non5xxresponses_By_Source.csv'
    elif dataset == '0524':
        log_file = 'backup_version/joint_training/0524_log_BERT_tokenize_template_pod_level_30_removed.npy'
        KPI_file = '/nfs/users/zach/aiops/data/0524/KPI.csv'
        metric_label = 'Book_Info_product'
    elif dataset == '0517':
        log_file = 'backup_version/joint_training/0517_log_BERT_tokenize_template_pod_level_window_size_30_removed.npy'
        KPI_file = '/nfs/users/zach/aiops/data/0517/KPI.csv'
        metric_label = 'Book_Info_product'
    else:
        raise 'Incorret Dataset Error'
    label = metric_label
    # log_file
    # path_dirs = "/nfs/users/zach/aiops_data/data/0517/"
    path_dirs = "/nfs/users/lecheng/REASON/log_analysis/src/"
    #Find common pods    
    for metric, weight in POD_METRIC_FILE.items():
        if metric != 'Score':
            metric_file = '/nfs/users/zach/aiops/data/{}/pod_level_data_{}.npy'.format(dataset, metric)
            log_data[metric] = np.load(metric_file, allow_pickle=True).item()
            if columns_common:
                columns_common = list(set(log_data[metric][label]['Pod_Name']).intersection(columns_common))
            else:
                columns_common = list(log_data[metric][label]['Pod_Name'])
        else:
            log_data[metric] = np.load(path_dirs + log_file, allow_pickle=True).item()
            if columns_common:
                columns_common = list(set(log_data[metric][log_label]['Node_Name']).intersection(columns_common))
            else:
                columns_common = list(log_data[metric][log_label]['Node_Name'])
    columns_common.sort()
    
    df = pd.read_csv(KPI_file)
    ### check the length of the timestamps in KPI data and log data and ensure that both have the same granularity.
    kpi_length = len(str(df['timeStamp'].iloc[1]))
    log_length = len(str(log_data[metric][log_label]['time'][1]))
    scale = 1
    if kpi_length > log_length:
        scale = 10 ** (kpi_length - log_length)
    window = SlidingWindow()
    processed_sequence = window.sliding_window(df["timeStamp"], df["Latency"],
                                            window_size=float(window_size) * 60,
                                            step_size=float(step_size) * 60,
                                            scale = scale)
    processed_sequence = processed_sequence.sort_values(by='StartTime', ignore_index=True)

    kpi_data = np.zeros(log_data[metric][log_label]['time'].shape)
    KPI_timestamp = 0
    j = 0
    ALIGNMENT_ERROR = 'Out of index error!!! Please double-check if the parameters (time_size and window_size) for log data and kpi are identical!'
    for i, log_timestamp in enumerate(log_data[metric][log_label]['time']):
        KPI_timestamp = (processed_sequence['StartTime'][j] // (step_size * 60)+1) * (step_size * 60)
        while KPI_timestamp < log_timestamp:
            j += 1
            if j >= processed_sequence['StartTime'].shape[0]:
                print('Index exceeds the length of KPI. Exit!')
                break
            # KPI_timestamp = (processed_sequence['StartTime'].iloc[j] // (step_size * 60) +1) * (step_size * 60)
            KPI_timestamp = (processed_sequence['StartTime'][j] // (step_size * 60)+1) * (step_size * 60)
        if j >= processed_sequence['StartTime'].shape[0]:
            print('Index exceeds the length of KPI. Exit!')
            break
        # print('i=', i, ',current log timestamp is ',log_timestamp, ' and the KPI timestamp is ', KPI_timestamp)
        if KPI_timestamp == log_timestamp:
            # print('Set KPI to the value ({}) at index {} at timestamp {}'.format(df['Latency'].iloc[j], j, KPI_timestamp))
            kpi_data[i] = processed_sequence['Latency'].iloc[j]
        elif KPI_timestamp > log_timestamp and j-1>0:
            # set the value to the one in the previous timestamp 
            kpi_data[i] = processed_sequence['Latency'].iloc[j-1]
            # print('i=', i, 'Set KPI to the value at index ', j-1)
        else:
            # print('Error found in log timestamp: ', log_timestamp)
            continue
    kpi_data = kpi_data[:log_data[metric][log_label]['Sequence'].shape[1]]
    print('Finish alignment!')
    sample = kpi_data.shape[0]// log_patch
    kpi_data = kpi_data[:log_patch*sample]
    kpi_data = kpi_data.reshape(-1, 1)
    kpi_data = np.mean(kpi_data.reshape((-1, log_patch, kpi_data.shape[1])), axis=1)
    kpi_data = preprocessing.normalize(kpi_data, axis=0, norm = 'l1')
    
    # POD_METRIC_FILE = POD_log_FILE
    metric_data = log_data
    index_data = {}
    metric_names = []
    metric_weight_assigned = []
    for metric, weight in POD_METRIC_FILE.items():
        if metric == 'Score':
            label = log_label
            index_data[metric] = [metric_data[metric][label]['Node_Name'].index(x) for x in columns_common]
            metric_data[metric][label]['Sequence'] = metric_data[metric][label]['Sequence'].sum(axis=2).transpose()
        else:
            label = metric_label
            index_data[metric] = [metric_data[metric][label]['Pod_Name'].index(x) for x in columns_common]
        metric_names = metric_names + [metric]
        metric_weight_assigned = metric_weight_assigned  + [weight]
    causal_score_combine = 0
    pod_results_combine = np.zeros((len(POD_METRIC_FILE),len(columns_common)))
    metric_weight  =  np.zeros((len(POD_METRIC_FILE),1))
    metric_id = 0
    individual_causal_score = {}
    alpha = 0.00
    
    
    
    
    for metric, weight in POD_METRIC_FILE.items():
        #print('For metric:', metric)
        if metric == 'Score':
            label = log_label
        else:
            label = metric_label
        data = metric_data[metric]
        X = data[label]['Sequence']
        index = index_data[metric]

        #Preprocessing to reduce the redundant samples
        patch = log_patch
        sample = X.shape[0]//patch
        X = X[:patch*sample,:]        
        X = np.sum(X.reshape((-1, patch, X.shape[1])), axis=1)
        X_metric = X[:, index]
        
        print('Shape of X', X.shape)
        print('shape pf kpi', kpi_data.shape)
        if metric != 'Score':
            X = np.append(X_metric, X[:, -1].reshape(-1,1), axis=1) 
        else:
            X = np.append(X_metric, kpi_data.reshape(-1,1), axis=1) 
        columns = list(columns_common) + ['KPI_Feature']
        #print('Original X shape: ', X.shape)

        std = np.std(X[:, :-1], axis=0)
        idx_std = [i for i, x in enumerate(std > 1e-5) if x]
        if len(idx_std) == 0:
            metric_weight[metric_id] = 0
            metric_id = metric_id + 1
            print(metric,' all pods are all constant or quasi-constant')
            continue

        selector = VarianceThreshold(threshold = 0)
        X_var = selector.fit_transform(X[:, :-1])
        idx = selector.get_support(indices = True)
        #print('X shape after variance: ', X_var.shape)
        if X_var.shape[1] < 1:
            metric_weight[metric_id] = 0
            metric_id = metric_id + 1
            print(metric,' all pods are all constant or quasi-constant')
            continue
        
        causal_score = np.zeros(len(columns_common))
        mask = np.full(len(columns_common), False,dtype=bool)
        mask[idx] = True      
        idx = list(idx) + [X.shape[1]-1] 
        X = X[:, idx]     
        columns = [columns[i] for i in idx]
        #print('Remain X ', X.shape)            
            
        #print('Detecting individual causal ...')
        ind_causal_score = detect_individual_causal(X[:,:-1], method='SPOT', args={'d':10, 'q':1e-4, 'n_init':100, 'level':0.95})
        
        ind_causal_score = np.sum(ind_causal_score, axis=0)
        normalized_ind_causal_score = preprocessing.normalize([ind_causal_score], norm='l1').ravel()
        individual_causal_score[metric] = normalized_ind_causal_score
        #Select one pod as an representative to calculate metric weight      
        metric_weight_ind =  np.max(normalized_ind_causal_score, axis=0)  
        metric_weight[metric_id] = metric_weight_ind
        metric_id = metric_id + 1

        score = alpha * normalized_ind_causal_score  
        causal_score[mask] = score
        if len(individual_causal_score) == 0:
            individual_causal_score = pd.DataFrame(causal_score, columns = [metric])     
        else:
            individual_causal_score[metric] = causal_score
        print('Detectin individual causal done for metric:', metric)
   
    metric_weight = preprocessing.normalize(metric_weight, axis = 0, norm = 'l1').ravel()
    metric_weight = metric_weight.ravel()
    weight_results = {} 
    K = len(metric_weight)
    ranking = np.argsort(metric_weight,axis=0)[::-1]
    weight_results['ranking'] = [i+1 for i in range(K)]
    weight_results = pd.DataFrame(weight_results, columns = ['ranking'])
    weight_results['metric'] = np.array(metric_names)[ranking]
    weight_results['weight'] = metric_weight[ranking.ravel()]
    output_filename = "learned_metric_weight_pod.csv"
    weight_results.to_csv(os.path.join(pathset, output_filename))
    metric_names  = weight_results['metric']

    print('Metric prioritization done!')
    print('Prioritized metric order with learned weights:')
    for i in range(K):
    	print('{}: {},{}'.format(i+1, metric_names[ranking[i]], metric_weight[ranking.ravel()[i]])) 
    print('Causal graph generation are running in the prioritized order!')

    #Combine the assigned weight
    metric_weight = np.array(metric_weight_assigned)[ranking.ravel()] * metric_weight[ranking.ravel()]
    metric_weight = preprocessing.normalize([metric_weight], axis = 1, norm = 'l1').ravel()
    metric_id = 0
    for metric in metric_names:
        #print('For metric:', metric)
        if metric == 'Score':
            label = log_label
        else:
            label = metric_label
        data = metric_data[metric]
        X = data[label]['Sequence']
        index = index_data[metric]

        #Preprocessing to reduce the redundant samples
        patch = log_patch
        sample = X.shape[0]//patch
        X = X[:patch*sample,:]        
        X = np.sum(X.reshape((-1, patch, X.shape[1])), axis=1)
        X_metric = X[:, index]
        X_metric = preprocessing.normalize(X_metric, axis=0, norm = 'l1')
        if metric != 'Score':
            X = np.append(X_metric, X[:, -1].reshape(-1,1), axis=1) 
        else:
            X = np.append(X_metric, kpi_data.reshape(-1,1), axis=1) 
        columns = list(columns_common) + ['KPI_Feature']
        #print('Original X shape: ', X.shape)

        std = np.std(X[:, :-1], axis=0)
        idx_std = [i for i, x in enumerate(std > 1e-5) if x]
        if len(idx_std) == 0:
            metric_weight[metric_id] = 0
            metric_id = metric_id + 1
            print(metric,' all pods are all constant or quasi-constant')
            continue

        selector = VarianceThreshold(threshold = 0)
        X_var = selector.fit_transform(X[:, :-1])
        idx = selector.get_support(indices = True)
        #print('X shape after variance: ', X_var.shape)
        if X_var.shape[1] < 1:
            metric_weight[metric_id] = 0
            metric_id = metric_id + 1
            print(metric,' all pods are all constant or quasi-constant')
            continue
        
        causal_score = np.zeros(len(columns_common))
        mask = np.full(len(columns_common), False,dtype=bool)
        mask[idx] = True      
        idx = list(idx) + [X.shape[1]-1] 
        X = X[:, idx]     
        columns = [columns[i] for i in idx]       
        print('Generating causal graph for metric:', metric)
        normalized_ind_causal_score = individual_causal_score[metric][mask]

        #Graph neural network based method
        cg = generate_causal_graph(X, method='gnn', args={'lag': 20, 'layers': [20, 20], 'lambda1': 1, 'lambda2': 1e-2})

        #LSTM based method
        #cg = generate_causal_graph(X, method='lstm', args={'hidden': 100, 'context': 50, 'lam': 10., 'lam_ridge': 1e-2, 'lr': 1e-3, 'max_iter': 30000, 'check_every': 100, 'device': torch.device('cuda')})
        
        print('Generating Causal Graph Done!')
        # threshold top K
        threshold_factor = 0.3
        K = int(threshold_factor*len(cg.reshape(-1)))
        threshold = sorted(cg.reshape(-1), reverse=True)[K-1] 
        W_save = np.array(cg.transpose())
        W = np.where(cg>=threshold, 1, 0)
        W = W.transpose()
        Q = generate_Q(X, W, RI=W.shape[0]-1, rho=1e-2)

        # error propagation
        #print('Propagaing Error ...')
        steps = 10000
        count = propagate_error(Q, start=W.shape[0]-1, steps=steps)
        count /= steps
        #print('Propagating Error Done!')

        count = np.mean(W, axis=1)

        #Combine individual causal detection and topology causal detection        
        score =  normalized_ind_causal_score + (1 - alpha) * count[:-1] 
        causal_score[mask] = score
        causal_score = causal_score.ravel()
        normalized_score = score
        pod_results_combine[metric_id,:] = causal_score
        metric_id = metric_id + 1
        #Display top K result
        K = len(causal_score)
        ranking = np.argsort(causal_score)[::-1]
        print('Ranking for metric {}:'.format(metric))
    
        for i in range(K):
            print('{}: {} {}'.format(i+1, columns_common[ranking[i]], causal_score[ranking[i]]))
        
        pod_results = {}
        pod_results['ranking'] = [i+1 for i in range(K)]
        pod_results = pd.DataFrame(pod_results, columns = ['ranking'])
        pod_results ['pod'] = [columns_common[ranking[i]] for i in range(K)]
        pod_results ['score'] = [causal_score[ranking[i]] for i in range(K)]
         
        info = {'W': W_save, 
                'Q': Q,
                'columns': columns, 
                'ind_causal_score': normalized_ind_causal_score[:-1],
                'count': count[:-1],
                'score': normalized_score}
    
        metric_file ='inrc_pod_{}'.format(metric)  
        output_file = os.path.join(pathset, metric_file)        
        np.save(output_file+'.npy', info)
        pod_results.to_csv(output_file +'_ranking.csv')
        print('Pod root cause detection done for metric:', metric)
    
    pod_results_combine = np.matmul(metric_weight, pod_results_combine) 
    K = len(list(columns_common))
    causal_score_combine = preprocessing.normalize([pod_results_combine]).ravel()
    ranking = np.argsort(causal_score_combine)[::-1]
    print('Ranking after intergrated analysis of all selected metrics')
    print(dataset)
    for i in range(K):
    	print('{}: {} {}'.format(i+1, columns_common[ranking[i]], causal_score_combine[ranking[i]]))  
    pod_results_combine = {}
    pod_results_combine['ranking'] = [i+1 for i in range(K)]
    pod_results_combine = pd.DataFrame(pod_results_combine, columns = ['ranking'])
    pod_results_combine ['pod'] = [columns_common[ranking[i]] for i in range(K)]
    pod_results_combine ['score'] = [causal_score_combine[ranking[i]] for i in range(K)]
    output_file = os.path.join(pathset, 'Pod_level_combine_ranking.csv')
    pod_results_combine.to_csv(output_file)
    
    info_all = {
            'columns': columns_common, 
            'score': causal_score_combine}
    metric_file ='inrc_pod_all'
    output_file = os.path.join(pathset, metric_file)        
    np.save(output_file+'.npy', info_all)

