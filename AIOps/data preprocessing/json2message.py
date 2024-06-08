import json
import glob
import os
import pandas as pd
import csv
import re  # Import regular expressions


def remove_timestamps(message):
    # Remove datetime info of different formats
    message = re.sub(r'\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\]', '', message) 
    message = re.sub(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z', '', message)  
    message = re.sub(r'\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}', '', message)  
    message = re.sub(r'\d{4}\.\d{2}\.\d{2} \d{2}:\d{2}:\d{2}', '', message)  
    message = re.sub(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}\+\d{4}', '', message)  
    message = re.sub(r'\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}', '', message)
    message = re.sub(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', '', message)
    message = re.sub(r'\[\d{2}/\w{3}/\d{4}:\d{2}:\d{2}:\d{2} \w{3}\]', '', message)
    message = re.sub(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}\+\d{2}:\d{2}', '', message)
    message = re.sub(r'\w{3} \d{1,2}, \d{4}', '', message)
    message = re.sub(r'\d{1,2} \w{3} \d{4}', '', message)
    message = re.sub(r'\d{2}:\d{2} [AP]M', '', message)
    message = re.sub(r'\[\d{2}/\w{3}/\d{4} \d{2}:\d{2}:\d{2}\]', '', message)
    message = re.sub(r'^I\d{4} \d{2}:\d{2}:\d{2}\.\d{6}\s+\d+\s+\w+.\w+:\d+\] ', '', message)
    message = re.sub(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z', '', message)
    
    return message.strip()



def extract_log_message(log):
    # Check if the log is in JSON format
    if "msg:" in log or "\"msg\":" in log:
        log_json = json.loads(log)
        log_message = log_json.get('msg', '')
    elif "msg=" in log or "\"msg\"=" in log:
        msg_index = log.find("msg=") if "msg=" in log else log.find("\"msg\"=")
        first_quote_index = log.find('"', msg_index)
        last_quote_index = log.find('"', first_quote_index + 1)
        if last_quote_index != -1:
            log_message = log[first_quote_index + 1:last_quote_index]
        else:
            log_message = log[first_quote_index + 1:]
    else:
        log_message = log
    
    return log_message.strip()

def dependency(path,output_dir):
    extract_pod_list=[]
    extract_node_list=[]
    folder_list = os.listdir(path)
     
    for folder in folder_list:  
        json_file = path + folder + "/" + "*.json"
        for readfile in glob.glob(json_file):
            print(readfile)
            with open(readfile) as f:
                jsn = json.load(f)
                for jsn_hit in jsn['hits']['hits']:
                    all_proc = []
                    all_node = []
                    if "kubernetes" in jsn_hit['_source'] and "pod_name" in jsn_hit['_source']['kubernetes']:
                        pod = jsn_hit['_source']['kubernetes']['pod_name']
                        message = jsn_hit['_source']['message']
                        timestamp = jsn_hit['_source']['@timestamp']
                        if message.startswith('"'):
                            message = message[1:]
                        if message.endswith('"'): 
                            message = message[:-1]
                        if "msg" in message:
                            # print(message)
                            message = extract_log_message(message)
                            # message = json.loads(message)['msg']
                        
                        message = remove_timestamps(message)
                        all_proc.append(pod)
                        all_proc.append(timestamp)
                        all_proc.append(message)
                        if all_proc:
                            extract_pod_list.append(all_proc)
                    if "systemd" in jsn_hit['_source'] and "t" in jsn_hit['_source']['systemd']:
                        node = jsn_hit['_source']['hostname']
                        message = jsn_hit['_source']['message']
                        timestamp = jsn_hit['_source']['@timestamp']
                        if message.startswith('"'):
                            message = message[1:]
                        if message.endswith('"'): 
                            message = message[:-1]
                        if "msg:" in message  or "\"msg\":" in message:
                            message = extract_log_message(message)
                            # message = json.loads(message)['msg']
                        all_node.append(node)
                        all_node.append(timestamp)
                        all_node.append(message)
                        if all_node:
                            extract_node_list.append(all_node)                  
    # output file
    data_list_col=['Node','Timestamp','Messages']
    node_df = pd.DataFrame(extract_node_list,columns=data_list_col)
    node_df.dropna()
    filename = 'Node_messages'
    node_df = node_df.sort_values(by='Timestamp')
    node_df.to_csv(output_dir + filename, index = False) 
    csv_file = output_dir + filename 
    partition_csv(csv_file, output_dir3)

    data_list_col=['Pod','Timestamp','Messages']
    pod_df = pd.DataFrame(extract_pod_list,columns=data_list_col)
    pod_df.dropna()
    filename = 'Pod_messages'
    pod_df = pod_df.sort_values(by='Timestamp')
    pod_df.to_csv(output_dir + filename, index = False) 
    csv_file = output_dir + filename 
    partition_csv(csv_file, output_dir2)
    

    
def partition_csv(csv_file, output_dir):
    isExist = os.path.exists(output_dir)
    if not isExist:
        os.mkdir(output_dir)
    # Creates empty set - this will be used to store the values that have already been used
    filelist = set()
    # Opens the large csv file in "read" mode
    with open(csv_file,'r') as csvfile:
        read_rows = csv.reader(csvfile)
        # Skip the column names
        next(read_rows)
        for row in read_rows:
            # Store the whole row as a string (rowstring)
            rowstring='\t'.join(row[1:])
            # Defines filename as the first entry in the row - This could be made dynamic so that the user inputs a column name to use
            filename = (row[0])
            # This basically makes sure it is not looking at the header row.
            # If the filename is not in the filelist set, add it to the list and create new csv file with header row.
            if filename not in filelist:    
                filelist.add(filename)
                temp_file = output_dir + str(filename +'_messages')
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                with open(temp_file,'a') as f:
                    f.write(rowstring)
                    f.write("\n")
                    f.close()
            # If the filename is in the filelist set, append the current row to the existing csv file.     
            else:
                temp_file = output_dir + str(filename +  '_messages')
                with open(temp_file,'a') as f:
                    f.write(rowstring)
                    f.write("\n")
                    f.close() 
    
if __name__ == "__main__":
    # Input log data directory
    path = 'Path-to-the-dataset-directory' 
    # Output directories  
    output_dir='./output/'  
    output_dir2='./output/log_prep_pod/' 
    output_dir3='./output/log_prep_node/' 
    dependency(path,output_dir)
