import yaml
import sys
import numpy as np
import pickle
from collections import defaultdict

def read_pickle_data(file_path):
    with open(file_path,"rb") as f:
        return pickle.load(f)

def read_aiops_data_new(data_config, metric_key):
    f = open(data_config)
    config = yaml.safe_load(f)
    data = read_pickle_data(config['path_dirs'])
    metrics = list(data.keys())
    return data, metrics, config['root_cause'], config['label'],config['online_start'], config['failure_time']


def read_aiops_data(data_config, metric_key, data_path):
    f = open(data_config)
    config = yaml.safe_load(f)
    metric_data = {}
    metrics = []
    # Find common pods among different metrics
    shared_pods = []
    min_time, max_time = sys.maxsize,-1111
    for metric, weight in config['metrics'].items():
        metric_file = data_path+'pod_level_data_{}.npy'.format(metric)
        metrics.append(metric)
        # metric_file = '{}_{}.npy'.format(data_path, metric)
        metric_data[metric] = np.load(metric_file, allow_pickle=True).item()
        time_series = metric_data[metric][config['label']]['time']
        time_series = np.array(list(range(len(time_series))))+time_series[0]
        metric_data[metric][config['label']]['time'] = time_series
        if shared_pods:
            shared_pods = list(set(metric_data[metric][config['label']]['Pod_Name']).intersection(shared_pods))
        else:
            shared_pods = list(metric_data[metric][config['label']]['Pod_Name'])

        if np.max(metric_data[metric][config['label']]['time']) > max_time:
            max_time = np.max(metric_data[metric][config['label']]['time'])
        if np.min(metric_data[metric][config['label']]['time']) < min_time:
            min_time = np.min(metric_data[metric][config['label']]['time'])

    for metric, weight in config['metrics'].items():
        idx_pods = [metric_data[metric][config['label']]['Pod_Name'].index(x) for x in shared_pods]
        time_seq = metric_data[metric][config['label']]['Sequence'][:,:-1]
        kpi_seq = metric_data[metric][config['label']]['Sequence'][:,-1].reshape(-1,1)
        time_seq = time_seq[:,idx_pods]
        final_seq = np.hstack((time_seq,kpi_seq))
        metric_data[metric][config['label']]['Sequence'] = final_seq
        metric_data[metric][config['label']]['Pod_Name'] = np.array(metric_data[metric][config['label']]['Pod_Name'])[idx_pods]
    if metric_key != 'all':
        return metric_data[metric_key][config['label']], metrics, config['root_cause'], min_time, max_time
    else:
        return metric_data, metrics, config['root_cause'], config['label'], min_time, max_time


def read_swat_data(data_config):
    f = open(data_config)
    config = yaml.safe_load(f)
    file_paths = config['path_dirs'].keys()
    data = defaultdict(dict)
    for ind,file_key in enumerate(file_paths):
        with open(config['path_dirs'][file_key],"rb") as f:
            metric_data = pickle.load(f)
        min_time, max_time = sys.maxsize, -1111
        shared_pods = []
        metrics = []
        for metric, weight in config['metrics'].items():
            metrics.append(metric)
            time_series = metric_data[metric][config['label']]['Time']
            time_series = np.array(list(range(len(time_series)))) + time_series[0]
            metric_data[metric][config['label']]['Time'] = time_series
            if np.max(metric_data[metric][config['label']]['Time']) > max_time:
                max_time = np.max(metric_data[metric][config['label']]['Time'])
            if np.min(metric_data[metric][config['label']]['Time']) < min_time:
                min_time = np.min(metric_data[metric][config['label']]['Time'])
            if shared_pods:
                shared_pods = list(set(metric_data[metric][config['label']]['Pod_Name']).intersection(shared_pods))
            else:
                shared_pods = list(metric_data[metric][config['label']]['Pod_Name'])
        for metric, weight in config['metrics'].items():
            idx_pods = [metric_data[metric][config['label']]['Pod_Name'].index(x) for x in shared_pods]
            time_seq = metric_data[metric][config['label']]['Sequence'][:, :-1]
            kpi_seq = metric_data[metric][config['label']]['Sequence'][:, -1].reshape(-1, 1)
            time_seq = time_seq[:, idx_pods]
            final_seq = np.hstack((time_seq, kpi_seq))
            metric_data[metric][config['label']]['Sequence'] = final_seq
            pods_name = metric_data[metric][config['label']]['Pod_Name']
            pods_name = ['S'+str(x[0])+'_'+str(x) for x in pods_name]
            metric_data[metric][config['label']]['Pod_Name'] = np.array(pods_name)[idx_pods]
        data[ind]['data'] = metric_data
        data[ind]['min_time'] = min_time
        data[ind]['max_time'] = max_time
        data[ind]['label'] = config['label']
        data[ind]['metrics'] = metrics
        data[ind]['root_cause'] = list(config['root_cause'].values())[ind]
        print("The {}-th fault has been loaded!".format(ind))
    return data


def read_wadi_data(data_config):
    f = open(data_config)
    config = yaml.safe_load(f)
    file_paths = config['path_dirs'].keys()
    data = defaultdict(dict)
    for ind,file_key in enumerate(file_paths):
        with open(config['path_dirs'][file_key],"rb") as f:
            metric_data = pickle.load(f)
        min_time, max_time = sys.maxsize, -1111
        shared_pods = []
        metrics = []
        for metric, weight in config['metrics'].items():
            metrics.append(metric)
            time_series = metric_data[metric][config['label']]['Time']
            time_series = np.array(list(range(len(time_series)))) + time_series[0]
            metric_data[metric][config['label']]['Time'] = time_series
            if np.max(metric_data[metric][config['label']]['Time']) > max_time:
                max_time = np.max(metric_data[metric][config['label']]['Time'])
            if np.min(metric_data[metric][config['label']]['Time']) < min_time:
                min_time = np.min(metric_data[metric][config['label']]['Time'])
            if shared_pods:
                shared_pods = list(set(metric_data[metric][config['label']]['Pod_Name']).intersection(shared_pods))
            else:
                shared_pods = list(metric_data[metric][config['label']]['Pod_Name'])
        for metric, weight in config['metrics'].items():
            idx_pods = [metric_data[metric][config['label']]['Pod_Name'].index(x) for x in shared_pods]
            time_seq = metric_data[metric][config['label']]['Sequence'][:, :-1]
            kpi_seq = metric_data[metric][config['label']]['Sequence'][:, -1].reshape(-1, 1)
            time_seq = time_seq[:, idx_pods]
            final_seq = np.hstack((time_seq, kpi_seq))
            metric_data[metric][config['label']]['Sequence'] = final_seq
            pods_name = metric_data[metric][config['label']]['Pod_Name']
            pods_name = ['S'+str(x[0])+'_'+str(x) for x in pods_name]
            metric_data[metric][config['label']]['Pod_Name'] = np.array(pods_name)[idx_pods]
        data[ind]['data'] = metric_data
        data[ind]['min_time'] = min_time
        data[ind]['max_time'] = max_time
        data[ind]['label'] = config['label']
        data[ind]['metrics'] = metrics
        data[ind]['root_cause'] = list(config['root_cause'].values())[ind]
        print("The {}-th fault has been loaded!".format(ind))
    return data


