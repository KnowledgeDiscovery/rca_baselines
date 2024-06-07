import pandas as pd

attack_info = pd.read_csv("attack_info.csv")

start_date_time = attack_info['Start Time'].values
end_time = attack_info['End Time'].values

end_date_time = []
for i,date_time in enumerate(start_date_time):
    date_value = date_time.split(' ')[0]
    end_date_time.append(date_value+' '+end_time[i])

attak_time_ranges = list(zip(start_date_time,end_date_time))

normal = pd.read_excel('normal.xlsx')
attack = pd.read_excel('attack.xlsx')

normal['label'].replace(0,1,inplace=True)

normal['Timestamp'] = pd.to_datetime(normal['Timestamp'])
normal.set_index(normal['Timestamp'],inplace=True)
attack['Timestamp'] = pd.to_datetime(attack['Timestamp'])
attack.set_index(attack['Timestamp'],inplace=True)

attack['label'].replace('Normal',1,inplace=True)
attack['label'].replace('Attack',-1,inplace=True)
attack['label'].replace('A ttack',-1,inplace=True)

attack_segments = []
for time_range in attak_time_ranges:
    attack_segments.append(attack[time_range[0]:time_range[1]])

time_ranges_before = [
    ['2015-12-26 00:00:00','2015-12-26 10:29:13'],
    ['2015-12-26 00:00:00','2015-12-26 10:51:07'],
    ['2015-12-26 00:00:00','2015-12-26 11:47:38'],
    ['2015-12-26 00:00:00','2015-12-26 11:11:24'],
    ['2015-12-26 00:00:00','2015-12-26 11:35:39'],
    ['2015-12-26 00:00:00','2015-12-26 14:38:11'],
    ['2015-12-26 00:00:00','2015-12-26 18:29:59'],
    ['2015-12-26 00:00:00','2015-12-26 22:55:17'],
    ['2015-12-26 00:00:00','2015-12-26 01:42:33'],
    ['2015-12-26 00:00:00','2015-12-26 09:51:07'],
    ['2015-12-26 00:00:00','2015-12-26 10:01:49'],
    ['2015-12-26 00:00:00','2015-12-26 01:17:07'],
    ['2015-12-26 00:00:00','2015-12-26 15:31:59'],
    ['2015-12-26 00:00:00','2015-12-26 15:47:39'],
    ['2015-12-26 00:00:00','2015-12-26 17:12:39'],
    ['2015-12-26 00:00:00','2015-12-26 17:18:55'],
]

data = pd.concat([normal['2015-12-25':'2015-12-28'],attack],axis=0)

normal_pattern_before = []
for time_range in time_ranges_before:
    normal_pattern_before.append(data[time_range[0]:time_range[1]])


time_ranges_after = [
    ['2015-12-26 10:44:54','2015-12-27 23:59:59'],
    ['2015-12-26 10:58:31','2015-12-27 23:59:59'],
    ['2015-12-26 11:54:09','2015-12-27 23:59:59'],
    ['2015-12-26 11:15:18','2015-12-27 23:59:59'],
    ['2015-12-26 11:42:51','2015-12-27 23:59:59'],
    ['2015-12-26 14:50:09','2015-12-27 23:59:59'],
    ['2015-12-26 18:42:01','2015-12-27 23:59:59'],
    ['2015-12-26 23:03:01','2015-12-27 23:59:59'],
    ['2015-12-26 01:54:11','2015-12-27 23:59:59'],
    ['2015-12-26 09:56:29','2015-12-27 23:59:59'],
    ['2015-12-26 10:12:02','2015-12-27 23:59:59'],
    ['2015-12-26 01:45:19','2015-12-27 23:59:59'],
    ['2015-12-26 15:34:01','2015-12-27 23:59:59'],
    ['2015-12-26 16:07:11','2015-12-27 23:59:59'],
    ['2015-12-26 17:14:21','2015-12-27 23:59:59'],
    ['2015-12-26 17:26:57','2015-12-27 23:59:59'],
]
normal_pattern_after = []
for time_range in time_ranges_after:
    normal_pattern_after.append(data[time_range[0]:time_range[1]])

final = []
for before,attack,after in zip(normal_pattern_before,attack_segments,normal_pattern_after):
    final.append(pd.concat([before,attack,after],axis=0))

print('end!')

import pickle
with open('tmp_data/data_segments.pkl','wb') as f:
    pickle.dump(final,f,protocol=1)



