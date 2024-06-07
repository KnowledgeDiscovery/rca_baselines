import numpy as np
import pandas as pd
import pickle
import time
import glob

files = sorted(glob.glob('tmp_data/*_node_data_cut.pkl'),key=lambda x:int(x.split('/')[1].split('_')[0]))
metrics = ['MV','LS','LT','FIT','AIT','MCV','P']
for ind,file in enumerate(files):
    with open(file,'rb') as f:
        seg = pickle.load(f)
        metric_data = {}
        for metric in metrics:
            res_mid = {}
            res_metric = {}
            met_data = []
            nod_name = []
            for nod, val in seg.items():
                nod_name.append(nod)
                met_data.append(list(val[metric].values))
            met_data.append(list(val['label'].values))
            met_data = pd.DataFrame(met_data).T
            res_metric['Sequence'] = met_data.to_numpy()
            times = val['Timestamp'].values.tolist()
            res_metric['Time'] = times
            res_metric['Node_Name'] = nod_name
            res_metric['KPI_Label'] = 'rca'
            res_metric['KPI_Feature'] = ['label']
            res_mid['rca'] = res_metric
            metric_data[metric] = res_mid
        # data.append(metric_data)
    with open('tmp_data/{}_node_level_final_data.pkl'.format(ind), 'wb') as f:
        pickle.dump(metric_data, f, protocol=1)
    print("{} segment has been done!".format(ind))








