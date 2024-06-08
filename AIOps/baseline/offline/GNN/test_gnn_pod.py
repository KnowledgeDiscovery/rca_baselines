import numpy as np
import pandas as pd
from sklearn import preprocessing
from rca import detect_individual_causal, generate_causal_graph, generate_Q, propagate_error
from sklearn.feature_selection import VarianceThreshold
import os

if __name__ == '__main__':
    dataset = '0524'
    #Assign weight for each metric: default equal weight
    POD_METRIC_FILE = {'cpu_usage': 1, 'memory_usage': 1, 'rate_transmitted_packets': 1, 'rate_received_packets': 1, 'received_bandwidth': 1, 'transmit_bandwidth': 1}
    metric_data = {}
    columns_common = {}
    pathset = "./output/"
    if not(os.path.exists(pathset)):
        os.mkdir(pathset)

    #KPI label
    if dataset == '1203':
        # KPI label  for 1203
        label = 'ratings.book-info.svc.cluster.local:9080/*'
    elif dataset == '0606':
        # KPI label for 0606
        label = 'reviews-v3'
    elif dataset == '0524':
        # KPI label for 0606
        # root cause: catalogue-85fd4965b7-q8477
        label = 'Book_Info_product'
    elif dataset == '0517':
        # KPI label for 0606
        label = 'Book_Info_product'
    else:
        raise 'Incorret Dataset Error'
    path_dirs = "/nfs/users/zach/aiops/data/{}/".format(dataset)
    
    
    
    #Find common pods    
    for metric, weight in POD_METRIC_FILE.items():
        metric_file = path_dirs+'pod_level_data_{}'.format(metric)
        metric_file = metric_file + '.npy'
        metric_data[metric] = np.load(metric_file, allow_pickle=True).item()
        if columns_common:
            columns_common = list(set(metric_data[metric][label]['Pod_Name']).intersection(columns_common))
        else:
            columns_common = list(metric_data[metric][label]['Pod_Name'])
    
    index_data = {}
    metric_names = []
    metric_weight_assigned = []
    for metric, weight in POD_METRIC_FILE.items():
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
        data = metric_data[metric]
        X = data[label]['Sequence']
        index = index_data[metric]

        #Preprocessing to reduce the redundant samples
        patch = 100
        sample = X.shape[0]//patch
        X = X[:patch*sample,:]        
        X = np.sum(X.reshape((-1, patch, X.shape[1])), axis=1)
        X_metric = X[:, index]
        X = np.append(X_metric, X[:, -1].reshape(-1,1), axis=1) 
        columns = list(columns_common) + data[label]['KPI_Feature']
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
        data = metric_data[metric]
        X = data[label]['Sequence']
        index = index_data[metric]

        #Preprocessing to reduce the redundant samples
        patch = 100
        sample = X.shape[0]//patch
        X = X[:patch*sample,:]        
        X = np.sum(X.reshape((-1, patch, X.shape[1])), axis=1)
        X_metric = X[:, index]
        X_metric = preprocessing.normalize(X_metric, axis=0, norm = 'l1')
        X = np.append(X_metric, X[:, -1].reshape(-1,1), axis=1) 
        columns = list(columns_common) + data[label]['KPI_Feature']
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

