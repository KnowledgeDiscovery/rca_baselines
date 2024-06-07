import pandas as pd
import numpy as np

normal = pd.read_csv("normal.csv",skiprows=4)
attack = pd.read_csv("attack.csv")

attack_cols = []
for col in list(attack.columns):
    attack_cols.append(col.split('\\')[-1])

normal_cols = []
for col in list(normal.columns):
    normal_cols.append(col.split('\\')[-1])

normal.columns = normal_cols
attack.columns = attack_cols

attack = attack.dropna(axis=1,how='all')
normal = normal.dropna(axis=1,how='all')
attack = attack.fillna(0)
normal = normal.fillna(0)

attack.insert(1,'Timestamp',attack['Date'] + ' '+ attack['Time'])
del attack['Date']
del attack['Time']
del attack['Row']
attack['Timestamp'] = pd.to_datetime(attack['Timestamp'])
attack.set_index(attack['Timestamp'],inplace=True)


normal.insert(1,'Timestamp',normal['Date'] + ' '+ normal['Time'])
del normal['Date']
del normal['Time']
del normal['Row']
normal['Timestamp'] = pd.to_datetime(normal['Timestamp'])
normal.set_index(normal['Timestamp'],inplace=True)

normal['label'] = 1
attack['label'] = 1
attack['2017-10-09 19:25:00':'2017-10-09 19:50:16']['label'] = -1
attack['2017-10-10 10:24:10':'2017-10-10 10:34:00']['label'] = -1
attack['2017-10-10 10:55:00':'2017-10-10 11:24:00']['label'] = -1
attack['2017-10-10 11:30:40':'2017-10-10 11:44:50']['label'] = -1
attack['2017-10-10 13:39:30':'2017-10-10 13:50:40']['label'] = -1
attack['2017-10-10 14:48:17':'2017-10-10 14:59:55']['label'] = -1
attack['2017-10-10 17:40:00':'2017-10-10 17:49:40']['label'] = -1
attack['2017-10-11 10:55:00':'2017-10-11 10:56:27']['label'] = -1
attack['2017-10-11 11:17:54':'2017-10-11 11:31:20']['label'] = -1
attack['2017-10-11 11:36:31':'2017-10-11 11:47:00']['label'] = -1
attack['2017-10-11 11:59:00':'2017-10-11 12:05:00']['label'] = -1
attack['2017-10-11 12:07:30':'2017-10-11 12:10:52']['label'] = -1
attack['2017-10-11 12:16:00':'2017-10-11 12:25:36']['label'] = -1
attack['2017-10-11 15:26:30':'2017-10-11 15:37:00']['label'] = -1

attack.to_hdf('tmp_data/attack.hdf',key='wdj',complevel=9)
normal.to_hdf('tmp_data/normal.hdf',key='wdj',complevel=9)


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

columns = set(normal.columns) & set(attack.columns)

remain_cols = []
for metric in metrics:
    for col in columns:
        col_me = col.split('_')
        if len(col_me) > 2:
            if metric == col_me[1]:
                remain_cols.append(col)


remain_cols.sort()
monitor_attack = attack[remain_cols]
monitor_normal = normal[remain_cols]

a2=sorted([x for x in columns if '2A' in x])
b2=sorted([x for x in columns if '2B' in x])

for a2_item,b2_item in zip(a2,b2):
    ca = monitor_attack[a2_item] + monitor_attack[b2_item]
    ca_name = a2_item.replace('2A','2')
    monitor_attack[ca_name] = ca
    del monitor_attack[a2_item], monitor_attack[b2_item]

    cn =  monitor_normal[a2_item] + monitor_normal[b2_item]
    cn_name = a2_item.replace('2A','2')
    monitor_normal[cn_name] = cn
    del monitor_normal[a2_item], monitor_normal[b2_item]

monitor_attack.sort_index(axis=1,inplace=True)
monitor_normal.sort_index(axis=1,inplace=True) #按照列名排序

monitor_attack.insert(0,'Timestamp',attack['Timestamp'])
monitor_normal.insert(0,'Timestamp',normal['Timestamp'])

monitor_attack['label'] = attack['label']
monitor_normal['label'] = normal['label']

monitor_attack.to_hdf('tmp_data/monitor_attack.hdf',complevel=9,key='wdj')
monitor_normal.to_hdf('tmp_data/monitor_normal.hdf',complevel=9,key='wdj')


monitor_attack = pd.read_hdf('tmp_data/monitor_attack.hdf')
monitor_normal = pd.read_hdf('tmp_data/monitor_normal.hdf')

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

for key,vals in n2p.items():
    for val in vals:
        for metric in metrics:
            name = val.split('_')[0]+'_'+metric+'_'+val.split('_')[1]
            cols = [x for x in list(monitor_attack.columns) if name in x]
            if len(cols) == 1:
                monitor_attack.rename(columns={cols[0]:name},inplace=True)
                monitor_normal.rename(columns={cols[0]:name},inplace=True)
            elif len(cols) > 1:
                col_result = monitor_attack[cols[0]]
                for i in range(1,len(cols)):
                    col_result += monitor_attack[cols[i]]
                monitor_attack[name] = col_result
                for col in cols:
                    del monitor_attack[col]

                col_result_n = monitor_normal[cols[0]]
                for i in range(1,len(cols)):
                    col_result_n += monitor_normal[cols[i]]
                monitor_normal[name] = col_result_n
                for col in cols:
                    del monitor_normal[col]
            else:
                continue

monitor_attack.sort_index(axis=1,inplace=True)
monitor_normal.sort_index(axis=1,inplace=True)

attack_ts = monitor_attack['Timestamp']
monitor_attack.drop('Timestamp',axis=1,inplace=True)
monitor_attack.insert(0,'Timestamp',attack_ts)


normal_ts = monitor_normal['Timestamp']
monitor_normal.drop('Timestamp',axis=1,inplace=True)
monitor_normal.insert(0,'Timestamp',normal_ts)

monitor_attack.to_hdf("tmp_data/attack_final.hdf",key='wdj',complevel=9)
monitor_normal.to_hdf("tmp_data/normal_final.hdf",key='wdj',complevel=9)


attack = pd.read_hdf('tmp_data/attack_final.hdf')
normal = pd.read_hdf('tmp_data/normal_final.hdf')

attack_time_ranges = [
['2017-10-09 19:25:00','2017-10-09 19:50:16'],
['2017-10-10 10:24:10','2017-10-10 10:34:00'],
['2017-10-10 10:55:00','2017-10-10 11:24:00'],
['2017-10-10 11:30:40','2017-10-10 11:44:50'],
['2017-10-10 13:39:30','2017-10-10 13:50:40'],
['2017-10-10 14:48:17','2017-10-10 14:59:55'],
['2017-10-10 17:40:00','2017-10-10 17:49:40'],
['2017-10-11 10:55:00','2017-10-11 10:56:27'],
['2017-10-11 11:17:54','2017-10-11 11:31:20'],
['2017-10-11 11:36:31','2017-10-11 11:47:00'],
['2017-10-11 11:59:00','2017-10-11 12:05:00'],
['2017-10-11 12:07:30','2017-10-11 12:10:52'],
['2017-10-11 12:16:00','2017-10-11 12:25:36'],
['2017-10-11 15:26:30','2017-10-11 15:37:00']
]

attack_segments = []
for time_range in attack_time_ranges:
    attack_segments.append(attack[time_range[0]:time_range[1]])

data = pd.concat([normal['2017-10-05':'2017-10-06'],attack],axis=0)

time_ranges_before = [
    ['2017-10-05 00:00:00','2017-10-05 19:24:59'],
    ['2017-10-05 00:00:00','2017-10-05 10:24:09'],
    ['2017-10-05 00:00:00','2017-10-05 10:54:59'],
    ['2017-10-05 00:00:00','2017-10-05 11:30:39'],
    ['2017-10-05 00:00:00','2017-10-05 13:39:29'],
    ['2017-10-05 00:00:00','2017-10-05 14:48:16'],
    ['2017-10-05 00:00:00','2017-10-05 17:39:59'],
    ['2017-10-05 00:00:00','2017-10-05 10:54:59'],
    ['2017-10-05 00:00:00','2017-10-05 11:17:53'],
    ['2017-10-05 00:00:00','2017-10-05 11:36:30'],
    ['2017-10-05 00:00:00','2017-10-05 11:58:59'],
    ['2017-10-05 00:00:00','2017-10-05 12:07:29'],
    ['2017-10-05 00:00:00','2017-10-05 12:15:59'],
    ['2017-10-05 00:00:00','2017-10-05 15:26:29'],
]

time_ranges_after = [
    ['2017-10-05 19:50:17','2017-10-06 23:59:59'],
    ['2017-10-05 10:34:01','2017-10-06 23:59:59'],
    ['2017-10-05 11:24:01','2017-10-06 23:59:59'],
    ['2017-10-05 11:44:51','2017-10-06 23:59:59'],
    ['2017-10-05 13:50:41','2017-10-06 23:59:59'],
    ['2017-10-05 14:59:56','2017-10-06 23:59:59'],
    ['2017-10-05 17:49:41','2017-10-06 23:59:59'],
    ['2017-10-05 10:56:28','2017-10-06 23:59:59'],
    ['2017-10-05 11:31:21','2017-10-06 23:59:59'],
    ['2017-10-05 11:47:01','2017-10-06 23:59:59'],
    ['2017-10-05 12:05:01','2017-10-06 23:59:59'],
    ['2017-10-05 12:10:53','2017-10-06 23:59:59'],
    ['2017-10-05 12:25:37','2017-10-06 23:59:59'],
    ['2017-10-05 15:37:01','2017-10-06 23:59:59'],
]

normal_pattern_before = []
for time_range in time_ranges_before:
    normal_pattern_before.append(data[time_range[0]:time_range[1]])

normal_pattern_after = []
for time_range in time_ranges_after:
    normal_pattern_after.append(data[time_range[0]:time_range[1]])

final = []
for before,attack,after in zip(normal_pattern_before,attack_segments,normal_pattern_after):
    final.append(pd.concat([before,attack,after],axis=0))

import pickle
with open('tmp_data/data_segments.pkl','wb') as f:
    pickle.dump(final,f,protocol=1)
    
print('The process of data segment has been accomplished!')




