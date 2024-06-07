import numpy as np
import pandas as pd
import pickle

with open('tmp_data/data_segments.pkl','rb') as f:
       segments = pickle.load(f)

metrics = ['MV','LS','LT','FIT','AIT','MCV','P']

n2p = {
    'S1':[
        '1_001','1_002','1_003','1_004','1_005','1_006'
    ],
    'S2':[
        '2_101','2_201','2_301','2_401','2_501','2_601',
        '2_001','2_002','2_003', '2_004', '2_005','2_006','2_007', '2_009'
    ],
    'S3':[
        '3_001','3_002','3_003','3_004', '3_005'
    ]
}

np.save('tmp_data/n2p.npy',n2p)
n2p = np.load('tmp_data/n2p.npy',allow_pickle=True).item()

p2n = {}
for node,pods in n2p.items():
       for pod in pods:
              p2n[pod] = node
np.save('tmp_data/p2n.npy',p2n)
p2n = np.load('tmp_data/p2n.npy',allow_pickle=True).item()

stage_to_sensor = {'S1': ['1_AIT_001', '1_AIT_002', '1_AIT_003', '1_AIT_004', '1_AIT_005', '1_FIT_001', '1_LS_001', '1_LS_002', '1_LT_001', '1_MV_001', '1_MV_002', '1_MV_003', '1_MV_004', '1_P_001', '1_P_002', '1_P_003', '1_P_004', '1_P_005', '1_P_006'], 
                   'S2': ['2_AIT_001', '2_AIT_002', '2_AIT_003', '2_AIT_004', '2_FIT_001', '2_FIT_002', '2_FIT_003', '2_LS_101', '2_LS_201', '2_LS_301', '2_LS_401', '2_LS_501', '2_LS_601', '2_LT_001', '2_LT_002', '2_MCV_007', '2_MCV_101', '2_MCV_201', '2_MCV_301', '2_MCV_401', '2_MCV_501', '2_MCV_601', '2_MV_001', '2_MV_002', '2_MV_003', '2_MV_004', '2_MV_005', '2_MV_006', '2_MV_009', '2_MV_101', '2_MV_201', '2_MV_301', '2_MV_401', '2_MV_501', '2_MV_601', '2_P_003', '2_P_004'], 
                   'S3': ['3_AIT_001', '3_AIT_002', '3_AIT_003', '3_AIT_004', '3_AIT_005', '3_FIT_001', '3_LS_001', '3_LT_001', '3_MV_001', '3_MV_002', '3_MV_003', '3_P_001', '3_P_002', '3_P_003', '3_P_004']}
np.save('tmp_data/stage_to_sensor.npy',stage_to_sensor)
stage_to_sensor = np.load('tmp_data/stage_to_sensor.npy',allow_pickle=True).item()

nodes = list(n2p.keys())
pods = list(p2n.keys())

# finals = []
for ind,segment in enumerate(segments):
       segs = {}
       for node in nodes:
              result_data = []
              node_data = segment[stage_to_sensor[node[1]]]
              node_columns = node_data.columns
              result_data.append(list(segment['Timestamp']))
              for metric in metrics:
                     node_cols = [x for x in node_columns if metric in x]
                     if metric == 'P':
                            tmp = []
                            for item in node_cols:
                                   if 'PIT' in item or 'DPIT' in item:
                                          continue
                                   else:
                                          tmp.append(item)
                            node_cols = tmp
                     if metric == 'PIT':
                            tmp = []
                            for item in node_cols:
                                   if 'DPIT' in item:
                                          continue
                                   else:
                                          tmp.append(item)
                            node_cols = tmp
                     metric_data = node_data[node_cols].T.values
                     node_sum = np.zeros((metric_data.shape[1]))
                     for item in metric_data:
                            node_sum += item
                     result_data.append(list(node_sum))
              result_data.append(list(segment['label']))
              result_data = pd.DataFrame(result_data)
              result_data = result_data.T
              result_data.columns = ['Timestamp','MV','LS','LT','FIT','AIT','MCV','P','label']
              segs[node] = result_data
              print('{}-th segment {} has been done!'.format(ind,node))
       with open("tmp_data/{}_node_data_cut.pkl".format(ind),"wb") as f:
            pickle.dump(segs,f,protocol=1)
       print("{} segment has been done!".format(ind))



















