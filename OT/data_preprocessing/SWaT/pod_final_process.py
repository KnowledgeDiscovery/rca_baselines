import numpy as np
import pandas as pd
import pickle
import time
import glob


files = sorted(glob.glob('tmp_data/*_pod_data_cut.pkl'),key=lambda x:int(x.split('/')[1].split('_')[0]))
metrics = ['FIT','LIT','MV','P','PIT','AIT','DPIT','UV']
for ind,file in enumerate(files):
    with open(file,'rb') as f:
        seg = pickle.load(f)
        metric_data = {}
        for metric in metrics:
            res_mid = {}
            res_metric = {}
            pod_name = []
            met_data = []
            for pod, val in seg.items():
                pod_name.append(pod)
                met_data.append(list(val[metric].values))
            met_data.append(list(val['label'].values))
            met_data = pd.DataFrame(met_data).T
            res_metric['Sequence'] = met_data.to_numpy()
            times = val['Timestamp'].values.tolist()
            res_metric['Time'] = times
            res_metric['Pod_Name'] = pod_name
            res_metric['KPI_Label'] = 'rca'
            res_metric['KPI_Feature'] = ['label']
            res_mid['rca'] = res_metric
            metric_data[metric] = res_mid
        with open('tmp_data/{}_pod_level_final_data.pkl'.format(ind), 'wb') as f:
            pickle.dump(metric_data, f, protocol=1)
        print("{} segment {} metric has been done!".format(ind, metric))





