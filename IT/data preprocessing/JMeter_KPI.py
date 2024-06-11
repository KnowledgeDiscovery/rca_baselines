#Generate KPI from Jmeter data
import pandas as pd
import os

pathset = "./output/"
JMETER_PATH = './JMeter/result/Book_Info_Case04_49h_176400req_result.jtl'
jmeter_df = pd.read_csv(JMETER_PATH)

# convert timestamp
jmeter_df['timeStamp'] = jmeter_df['timeStamp'] // 1000

jmeter_col = ['Latency'] # feature of interests
columns = ['timeStamp', 'label'] + jmeter_col
jmeter_df = jmeter_df[columns]

# label of interests
label = 'Book_Info_product'

kpi = jmeter_df.loc[jmeter_df['label'] == label]

if not os.path.isdir(pathset):
    os.makedirs(pathset)
    
output_file = os.path.join(pathset, 'KPI.csv')
kpi.to_csv(output_file)
