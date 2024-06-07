import numpy as np
import pandas as pd
import pickle


segments = np.load('tmp_data/data_segments.npz',allow_pickle=True)['segments']
metrics = ['FIT','LIT','MV','P','PIT','AIT','DPIT','UV']
n2p = {'S1':['101','102'],
       'S2':['201','202','203','204','205','206'],
       'S3':['301','302','303','304'],
       'S4':['401','402','403','404'],
       'S5':['501','502','503','504'],
       'S6':['601','602','603']}

np.save('tmp_data/n2p.npy',n2p)
n2p = np.load('tmp_data/n2p.npy',allow_pickle=True).item()

p2n = {}
for node,pods in n2p.items():
       for pod in pods:
              p2n[pod] = node
np.save('tmp_data/p2n.npy',p2n)
p2n = np.load('tmp_data/p2n.npy',allow_pickle=True).item()

stage_to_sensor = {'S1': ['FIT101', 'LIT101', 'MV101', 'P101', 'P102'], 
                   'S2': ['AIT201', 'AIT202', 'AIT203', 'FIT201', 'MV201', 'P201', 'P202', 'P203', 'P204', 'P205', 'P206'],
                   'S3': ['DPIT301', 'FIT301', 'LIT301', 'MV301', 'MV302', 'MV303', 'MV304', 'P301', 'P302'], 
                   'S4': ['AIT401', 'AIT402', 'FIT401', 'LIT401', 'P401', 'P402', 'P403', 'P404', 'UV401'], 
                   'S5': ['AIT501', 'AIT502', 'AIT503', 'AIT504', 'FIT501', 'FIT502', 'FIT503', 'FIT504', 'P501', 'P502', 'PIT501', 'PIT502', 'PIT503'], 
                   'S6': ['FIT601', 'P601', 'P602', 'P603']}
stage_to_sensor = np.save('tmp_data/stage_to_sensor.npy', stage_to_sensor)
stage_to_sensor = np.load('tmp_data/stage_to_sensor.npy',allow_pickle=True).item()

nodes = list(n2p.keys())
pods = list(p2n.keys())

finals = []
for ind,segment in enumerate(segments):
       segs = {}
       for node,pods in n2p.items():
              node_data = segment[stage_to_sensor[node]]
              node_columns = node_data.columns
              for pod in pods:
                     result_data = []
                     result_data.append(list(segment['Timestamp']))
                     pod_columns = [x for x in node_columns if pod in x]
                     for metric in metrics:
                         pod_cols = [x for x in pod_columns if metric in x]
                         if metric == 'P':
                             tmp = []
                             for item in pod_cols:
                                 if 'PIT' in item or 'DPIT' in item:
                                     continue
                                 else:
                                     tmp.append(item)
                             pod_cols = tmp
                         if metric == 'PIT':
                             tmp = []
                             for item in pod_cols:
                                 if 'DPIT' in item:
                                     continue
                                 else:
                                     tmp.append(item)
                             pod_cols = tmp
                         metric_data = node_data[pod_cols].T.values
                         pod_sum = np.zeros((node_data.shape[0]))
                         for item in metric_data:
                             pod_sum += item
                         result_data.append(list(pod_sum))
                     result_data.append(list(segment['label']))
                     result_data = pd.DataFrame(result_data)
                     result_data = result_data.T
                     result_data.columns = ['Timestamp', 'FIT', 'LIT', 'MV', 'P', 'PIT', 'AIT', 'DPIT', 'UV', 'label']
                     segs[pod] = result_data

                     print('{}-th segment {} {} has been done!'.format(ind, node, pod))
       print("{} segment has been done!".format(ind))
       finals.append(segs)

with open("tmp_data/pod_data_cut.pkl","wb") as f:
    pickle.dump(finals,f,protocol=1)


