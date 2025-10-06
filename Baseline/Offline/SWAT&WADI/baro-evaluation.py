import sys
import glob
import pandas as pd
import numpy as np
from collections import defaultdict

files = glob.glob('./final*.csv')
model = 'baro'

model_files = defaultdict(list)

for file in files:
    model_files[model].append(file)

for key in model_files:
    model_files[key] = sorted(model_files[key], key=lambda x: x.split('/')[-2])

print(model_files)

predicts = []
mfiles = model_files['baro'] 
for mf in mfiles:
    print(mf)
    mf_data = pd.read_csv(mf)
    root_cause_list = list(mf_data['root_cause'].values)
    if 'Latency' in root_cause_list:
        root_cause_list.remove('Latency')
    predicts.append(root_cause_list)
        
reals = [
    ['1_MV_001'],
    ['1_FIT_001'],
    ['2_LIT_002', '1_AIT_001'],
    ['2_MCV_101', '2_MCV_201', '2_MCV_301', '2_MCV_401', '2_MCV_501', '2_MCV_601'],
    ['2_MCV_101', '2_MCV_201'],
    ['1_AIT_002', '2_MV_003'],
    ['2_MCV_007'],
    ['1_P_006'],
    ['1_MV_001'],
    ['2_MCV_007'],
    ['2_MCV_007'],
    ['2_AIT_003'],
    ['2_MV_201', '2_P_201', '2_P_202', '2_P_203', '2_P_204', '2_P_205', '2_P_206'],
    ['2_LIT_002', '1_AIT_001'],
]

def precision_on_topk(predicts,reals,k):
    pr = 0
    for pred, real in zip(predicts, reals):
        pred = pred[:k]
        hit_count = len(set(pred) & set(real))
        min_len = min(k,len(real))
        pr += hit_count/min_len
    return pr/len(reals)

def mean_precision_k(predicts,reals,k):
    pr = 0
    for i in range(1,k+1):
        pr += precision_on_topk(predicts,reals,i)
    return pr/k

def mrr(predicts,reals):
    mrr_val = 0
    for preds,real in zip(predicts,reals):
        tmp = []
        for real_item in real:
            index = preds.index(real_item) if real_item in preds else sys.maxsize-1
            tmp.append(index+1)
        mrr_val += 1/min(tmp)
    return mrr_val/len(reals)

k = [1,3,5,10]
for item in k:
    pr_k = precision_on_topk(predicts,reals,item)
    map_k = mean_precision_k(predicts,reals,item)
    print("Precision@{}:{}".format(item,pr_k))
    print('MAP@{}:{}'.format(item,map_k))
print('MRR:{}'.format(mrr(predicts,reals)))
