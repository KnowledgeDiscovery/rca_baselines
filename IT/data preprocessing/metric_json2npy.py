import glob
import orjson
import os
import pandas as pd
import time
import numpy as np
from collections import defaultdict

# Minimum sample size for finding the common timestamps
min_sample_size = 1e5

def main():
    st = time.time()

    # KPI location
    JAEGER_PATH = './reviews-v2-5d8f8b6775-sl4vp.csv'
    jaeger_df = pd.read_csv(JAEGER_PATH)
    # convert timestamp
    jaeger_df['timeStamp'] = jaeger_df['timeStamp'] // 1000000
    jaeger_col = ['Latency'] # feature of interests
    columns = ['timeStamp', 'label'] + jaeger_col
    jaeger_df = jaeger_df[columns]
    groups = jaeger_df.groupby('label')
    g2i = groups.indices
    jaeger_data = {}
    
    for label, indices in g2i.items():
        group = jaeger_df.iloc[indices, :].sort_values(by=['timeStamp']).reset_index()
        # take average if multiple records with the same timestamp exist
        group = group.groupby('timeStamp').mean().reset_index()
        jaeger_data[label] = group
    
    ##### 1. Pod metric data Path ###
    POD_METRIC_PATH='./Metrics/pod/'
    ##### 2. Node metric data path###
    node_path = "./Metrics"
    
    # Metric name list
    pod_metrics_list = ['cpu_usage','memory_usage','received_bandwidth','transmit_bandwidth','rate_received_packets','rate_transmitted_packets']
    node_metrics_list = ['cpu_usage','cpu_saturation','memory_usage','memory_saturation','net_usage_receive(bytes)','net_usage_transmit(bytes)','net_saturation_receive','net_saturation_transmit','net_disk_io_usage','net_disk_io_saturation','net_disk_space_usage']
    
    # pod_file json -> csv 
    header_name = "pod"
    
    for pod_metrics in pod_metrics_list:
        pod_file_convert(POD_METRIC_PATH, pod_metrics, header_name, jaeger_data, jaeger_col)
       
    header_name = "node"
    #node_file json -> csv
    for node_metric in node_metrics_list:
       node_json_file = node_path + "/node/node_" + node_metric + "*.json"
       node_file_convert(node_path, node_metric, node_json_file, header_name, jaeger_data, jaeger_col)
    
    et = time.time()
    elapsed_time = et - st
    print('Execution time:', elapsed_time, 'seconds') 

# Execution time°
def func_time(func):
    t1 = time.time() 
    t2 = time.time()
    elapsed_time = t2-t1


# Convert pod-level metrics data 
def pod_file_convert(POD_METRIC_PATH, metric, header_name, jaeger_data, jaeger_col):
    pod_data = defaultdict(lambda : {})
    data_list = []
    data_list_col=['pod','time','value']
    
    for nslabel in os.listdir(POD_METRIC_PATH):
        ns_path = os.path.join(POD_METRIC_PATH, nslabel)
        
        if not os.path.isdir(ns_path):
            continue
            
        metric_file ='{}_{}*.json'.format(nslabel, metric)
        json_file = os.path.join(ns_path, metric_file)
        
        # globã§jsonã‚’èª­ã¿è¾¼ã¿
        for filename in glob.glob(json_file):
            json_data = orjson.loads(open(filename).read())
            for result in json_data['data']['result']:
                for value in result['values']:
                    time_date = value[0]
                    data_list.append([result['metric']['pod'], time_date, float(value[1])])
    
    df = pd.DataFrame(data_list,columns=data_list_col)
    df = df.fillna(0)
    df = df.sort_values(['pod','time'])
    groups = df.groupby('pod')   
    g2i = groups.indices
    
    for plabel, indices in g2i.items():
        group = df.iloc[indices, [1, 2] ].groupby('time').mean().reset_index()
        group = group.sort_values(by='time').reset_index()
        if group.shape[0] < min_sample_size:
            continue
        pod_data[plabel] = group
            
    processed_data = {}
    metric_file ='pod_level_data_{}.npy'.format(metric)
    
    # find common timestamps
    for jlabel, jgroup in jaeger_data.items():
        ct = set(list(jgroup['timeStamp'].values.reshape(-1)))
        # ct = set()
        for plabel, pgroup in pod_data.items():
            ct = ct.intersection(set(list(pgroup['time'].values.reshape(-1))))         
        
        data_list=[]
        plabel_list = []
        for plabel, pgroup in pod_data.items():
            plabel_list.append(plabel)
            subset = pgroup.loc[pgroup['time'].isin(ct)].sort_values(by='time')
            data_list.append(subset.iloc[:, -1].values.reshape((-1, 1)))
            
        # jaeger data
        subset = jgroup.loc[jgroup['timeStamp'].isin(ct)].sort_values(by='timeStamp')
        data_list.append(subset[jaeger_col].values.reshape((-1, len(jaeger_col))))
        agg_data = np.concatenate(data_list, axis=1)
        processed_data[jlabel] = {
            'KPI_Label': jlabel,
            'Pod_Name': plabel_list,
            'KPI_Feature': jaeger_col,
            'Sequence': agg_data,
            'time': sorted(list(ct))
        }

    np.save(metric_file, processed_data)
    print('Pod Data of Metric', metric, 'Preprocessed')

# Convert node-level metric data
def node_file_convert(node_path, metric, json_file, header_name, jaeger_data, jaeger_col):
    data_list = []
    data_list_col=['node','time','value']
    metric_data = {}

    for filename in glob.glob(json_file):
        json_data = orjson.loads(open(filename).read())
        for result in json_data['data']['result']:
            for value in result['values']:
                time_date = value[0]
                data_list.append([result['metric']['instance'], time_date, float(value[1])])
    df = pd.DataFrame(data_list,columns=data_list_col)
    
    df['time'] = df['time'].astype(np.int64)
    # df timestampã®æ•´åˆ—
    df = df.sort_values(['node','time'])
    groups = df.groupby('node')
    g2i = groups.indices
    node_data = {}
    
    for node, indices in g2i.items():
        group = df.iloc[indices, :].sort_values(by='time').reset_index()
        group = group.groupby('time').mean().reset_index()
        node_data[node] = group
    metric_data[metric] = node_data
    
   
    processed_data = {}
    metric_file ='node_level_data_{}.npy'.format(metric)

    
    for jlabel, jgroup in jaeger_data.items():
        ct = set(list(jgroup['timeStamp'].values.reshape(-1)))
        for nlabel, ngroup in node_data.items():
            ct = ct.intersection(set(list(ngroup['time'].values.reshape(-1))))
        
        # aggregate data for each jlable
        # node data
        data_list=[]
        nlabel_list = []
        
        for nlabel, ngroup in node_data.items():
            nlabel_list.append(nlabel)
            subset = ngroup.loc[ngroup['time'].isin(ct)].sort_values(by='time')
            data_list.append(subset['value'].values.reshape((-1, 1)))

        # jaeger data
        subset = jgroup.loc[jgroup['timeStamp'].isin(ct)].sort_values(by='timeStamp')
        data_list.append(subset[jaeger_col].values.reshape((-1, len(jaeger_col))))
        agg_data = np.concatenate(data_list, axis=1)
     
        processed_data[jlabel] = {
            'KPI_Label': jlabel,
            'Node_Name': nlabel_list,
            'KPI_Feature': jaeger_col,
            'Sequence': agg_data,
            'time': sorted(list(ct))
        }
        
    np.save(metric_file, processed_data)  
    print('Node Data of Metric', metric, 'Preprocessed')

       

if __name__ == "__main__":
    main()
