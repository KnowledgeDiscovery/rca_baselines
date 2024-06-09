import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
import argparse
import os
import sys
sys.path.append('../')
import time
import re
import matplotlib.pyplot as plt

class SlidingWindow():
    def __init__(self):
        pass

    def sliding_window(self, df, templates, template_label_dict,  window_size, step_size):
        """
        :param df: dataframe columns=[timestamp, label, eventid, time duration, lineid]
        :param window_size: seconds,
        :param step_size: seconds
        :return: dataframe columns=[eventids, label, eventids, time durations, lineids]
        """
        log_size = df.shape[0]
        time_data, event_data = df.iloc[:, 0], df.iloc[:, 1]
        new_data = []
        start_end_index_pair = set()
        start_time = time_data[0]
        end_time = start_time + window_size
        start_index = 0
        end_index = 0
        # get the first start, end index, end time
        for cur_time in time_data:
            if cur_time < end_time:
                end_index += 1
            else:
                break
        start_end_index_pair.add(tuple([start_index, end_index]))
        # move the start and end index until next sliding window
        num_session = 1
        while end_index < log_size:
            start_time = start_time + step_size
            end_time = start_time + window_size
            for i in range(start_index, log_size):
                if time_data[i] < start_time:
                    i += 1
                else:
                    break
            for j in range(end_index, log_size):
                if time_data[j] < end_time:
                    j += 1
                else:
                    break
            start_index = i
            end_index = j
            # when start_index == end_index, there is no value in the window
            if start_index != end_index:
                start_end_index_pair.add(tuple([start_index, end_index]))
            num_session += 1
            if num_session % 1000 == 0:
                print("process {} time window".format(num_session), end='\r')
        template_frequency_in_dataset = {}
        for (start_index, end_index) in start_end_index_pair:
            event_sequence = []
            sequence_label = []
            template_frequency = {}
            for event in event_data[start_index: end_index]:
                if event not in template_label_dict:
                    template_label_dict[event] = is_abnormal_log(event)
                if event not in template_frequency:
                    template_frequency[event] = 1
                else:
                    template_frequency[event] += 1
                if event not in template_frequency_in_dataset:
                    template_frequency_in_dataset[event] = 1
                else:
                    template_frequency_in_dataset[event] += 1
                templates.add(event)
            for event, frequency in template_frequency.items():
                event_sequence.append(event)
                event_sequence.append(frequency)
                sequence_label.append(template_label_dict[event] * frequency)
            # if one template is labeled as abnormal, then this sequence is consider as abnormal
            sequence_label = sum(sequence_label)
            event_sequence = np.array(event_sequence)
            new_data.append([
                time_data[start_index],
                time_data[min(end_index, time_data.shape[0]-1)],
                event_sequence,
                sequence_label,
            ])
        new_df = pd.DataFrame(new_data, columns=['StartTime', 'EndTime', 'EventSequences', 'SequenceLabel'])
        assert len(start_end_index_pair) == len(new_data)
        print('There are %d instances (sliding windows) in this dataset' % len(start_end_index_pair))
        return new_df, templates

def is_abnormal_log(log_message):
    # List of keywords to detect abnormal log data
    keywords = ["error", "exception", "critical", "fatal", "timeout", "connection refused",
                "No space left on device", "out of memory", "terminated unexpectedly", "backtrace", "stack trace",
                "service unavailable", "502 Bad Gateway", "503 Service Unavailable", "504 Gateway Timeout",
                "unable to connect to", "rate limit exceeded", "request limit exceeded", "cloud system down", 
                "cloud service not responding", "failure", "corrupted data", "data loss", "file not found",
                "high CPU utilization", "CPU spike", "CPU saturation", "excessive CPU usage", "failed", "shutdown", 
                "Permission denied", "DEBUG", "cannot be contacted"]
    # Regular expressions for additional patterns
    # Example: Detect HTTP status codes outside the normal range (e.g., 400, 500)
    http_status_code_pattern = r"\b(4\d{2}|5\d{2})\b"
    cloud_provider_error_pattern = r"\b(AWS|Azure|Google Cloud)\b"
    custom_pattern = r"<\*> send error:.*"
    # Combine all patterns into a single regex
    combined_pattern = re.compile("|".join(keywords + [http_status_code_pattern, cloud_provider_error_pattern, custom_pattern]), re.IGNORECASE)
    # Check if the log message matches any of the patterns. 
    if re.search(combined_pattern, log_message):
        return 1
    else:
        return 0

def check_qualification(file_name):
    data = pd.read_csv(file_name, encoding='Latin-1', nrows=1500)
    length = data['EventTemplate'].shape[0]
    length_threshold = 1000
    if length < length_threshold:
        print('The number of records in {} is less than {}. Skipping it!'.format(file_name, length_threshold))
        return False
    return True

def check_directory_and_add_file(dir, files, file_path, file_list=None):
    if file_list:
        for f in file_list:
            if os.path.isfile(os.path.join(dir, f)) and f.endswith('_messages_structured.csv') and check_qualification(os.path.join(dir, f)):
                files.append(f)
                file_path.append(os.path.join(dir, f))
    else:
        for f in os.listdir(dir):
            if os.path.isfile(os.path.join(dir, f)) and f.endswith('_messages_structured.csv') and check_qualification(os.path.join(dir, f)):
                files.append(f)
                file_path.append(os.path.join(dir, f))
    return files, file_path


def align_two_lists(data, embeddings, length, idx):
    j = 0
    i = 0
    while i  < length:
        if j + 1 < len(idx):
            i = idx[j]
            end_i = idx[j+1]
            if end_i == i:
                j += 1
                continue
            if end_i - i <= 10:
                data[i:end_i] = np.array([embeddings[j] for _ in range(end_i - i)])
            else:
                data[i:i+10] = np.array([embeddings[j] for _ in range(10)])
                if j == 0:
                    mean_padding = embeddings[0]
                else:
                    mean_padding = np.mean(embeddings[:j], axis=0)
                data[i+10:end_i] = np.array([mean_padding for _ in range(end_i - i - 10)])
            j += 1
            i = end_i
        else:
            j = len(idx) - 1
            end_i = length
            if end_i - i <= 10: 
                data[i:end_i] = np.array([embeddings[j] for _ in range(end_i - i)])
            else:
                data[i:i+10] = np.array([embeddings[j] for _ in range(10)])
                mean_padding = np.mean(embeddings[:j], axis=0)
                data[i+10:end_i] = np.array([mean_padding for _ in range(end_i - i-10)])
            break
    return data


def normalization(X, axis=0):
    minimal = np.min(X, axis=axis, keepdims=True)
    maximal = np.max(X, axis=axis, keepdims=True)
    data = (X - minimal) / np.clip(maximal - minimal, 1e-10, None)
    return  data


## NOTE: 
# training from scratch command: python log_BERT_extraction_tokenize_template.py --dataset 0517 --from_scratch --level pod
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Content-Aware Transformer')
    parser.add_argument('--root_path', type=str, default='./data/0524/log_data/', help='root path of the data file')
    parser.add_argument("--output_dir", default="./output/cpf_log_result/", metavar="DIR", help="output directory")
    parser.add_argument("--save_dir", default="./output/cpf_log_result/", help="the directory to save bert model")
    parser.add_argument('--dataset', default='0524', help='dataset name')
    parser.add_argument('--step_size', default=0.5, type=float, help='step size(mins)')
    parser.add_argument('--window_size', default=10, type=float, help='window size(mins)')
    parser.add_argument('--level', default='pod', help='pod level or node level')
    parser.add_argument("--time_format", default='%Y-%m-%dT%H:%M:%S.%f+00:00', type=str, help="input time format")
    parser.add_argument('--use_small', action='store_true', help='Whether to use small dataset to test the code')
    parser.add_argument('--from_scratch', action='store_true', help='Whether to train BERT models from stratch')
    args = parser.parse_args()
    print('Args in experiment:')
    print(str(args))
    figure_path = './label_figure_{}_duplicate'.format(args.dataset)
    if not os.path.exists(figure_path):
        os.mkdir(figure_path)
    jlabel = 'ratings.book-info.svc.cluster.local:9080/*'
    ## NOTE: Typically, This value serves as the threshold to filter out the the pod with insufficient instances.
    minimum_instances = 1000
    node_names = []
    file_path = []
    files = []
    if args.level == 'pod':
        files, file_path = check_directory_and_add_file(args.root_path + 'pod_removed/', files, file_path)
    else:
        files, file_path = check_directory_and_add_file(args.root_path + 'node_removed/', files, file_path)
    for f in files:
        node_name = f.split('/')[-1].replace('_messages_structured.csv', '')
        node_names.append(node_name)
    start_time = time.time()
    sequences = {}
    templates = set()
    template_label_dict = {}
    for i, node_name in enumerate(node_names):
        filename = file_path[i]
        print('Processing the file: ', filename)
        df = pd.read_csv(filename, encoding='Latin-1')
        window = SlidingWindow()       
        df['datetime'] = pd.to_datetime(df["Time"], format=args.time_format)
        df['timestamp'] = df["datetime"].values.astype(np.int64) // 10 ** 9
        df = df.sort_values(by='datetime', ignore_index=True)
        df['deltaT'] = df['datetime'].diff() / np.timedelta64(1, 's')
        df['deltaT'].fillna(0)
        processed_sequence, templates = window.sliding_window(df[["timestamp", "EventTemplate"]], templates, template_label_dict,
                                            window_size=float(args.window_size) * 60,
                                            step_size=float(args.step_size) * 60)
        processed_sequence = processed_sequence.sort_values(by='StartTime', ignore_index=True)
        sequences[node_name] = {'StartTime': processed_sequence['StartTime'], 
                                'EventSequences': processed_sequence['EventSequences'],
                                'SequenceLabel': processed_sequence['SequenceLabel']}
    templates = list(templates)
    print('Total number of templates:', len(templates))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()
    print('It takes ', time.time()-start_time, 'seconds to generate window-sliding data.')
    begin_time = time.time()
    podwise_labels = []
    for node_name in node_names:
        df_sequence = sequences[node_name]
        if len(podwise_labels) == 0:
            podwise_labels.append(df_sequence['SequenceLabel'].values)
        else:
            podwise_labels.append(df_sequence['SequenceLabel'].values)
    labels = podwise_labels
    print('len of labels:', len(labels), 'len of node_name:', len(node_names))
    scale = 0
    node_idx = {}
    ## get the starttime and the end time
    start_time = np.inf
    end_time = 0
    last_start_time = 0
    least_end_time = np.inf
    final_node_names = []
    idx = 0
    # get the start time and end time
    for key, item in sequences.items():
        timestamp = item['StartTime'].iloc[0]
        timestamp_max = item['StartTime'].iloc[-1]
        node_idx[key] = idx
        idx += 1
        ### check if the timestamp_max is smaller than last_start_time to avoid the case where least_end_time < last_start_time
        if timestamp_max <= last_start_time:
            print('Detecting time.max() <= last_start_time. Removing {} from the training set'.format(key))
            node_names.remove(key)
            continue
        ### check if the timestamp_max is smaller than last_start_time to avoid the case where least_end_time < last_start_time
        if timestamp >= least_end_time:
            print('Detecting time.min() >= least_end_time. Removing {} from the training set'.format(key))
            node_names.remove(key)
            continue
        if start_time > timestamp:
            start_time = timestamp
        if last_start_time < timestamp:
            last_start_time = timestamp
        if  end_time < timestamp_max:
            end_time = timestamp_max
        if least_end_time > timestamp_max:
            least_end_time = timestamp_max
    print('Time : ', start_time, last_start_time, least_end_time, end_time)
    length = int((end_time - start_time) //  (args.step_size * 60)) + 1
    
    # Align the time sequence
    data = sequences
    # log_sequence is the output time-series data with shape: # of nodes by # of time windows by 1
    log_sequence = np.zeros((len(node_names), length, 1))
    index = 0
    for node_name in node_names:
        temp_data = np.zeros((length, 1))
        idx = list(map(lambda x: int((x - start_time) //  (args.step_size * 60)), data[node_name]['StartTime']))
        print('shape of temp_data:', temp_data.shape, 'shape of labels:', labels[node_idx[node_name]].shape, 'shape of idx:', len(idx))
        temp_data = align_two_lists(temp_data, labels[node_idx[node_name]].reshape(-1, 1), length, idx)
        print('The number of zeros in embedding is ', length - np.nonzero(temp_data.sum(axis=1))[0].shape[0])
        log_sequence[index, :, :] = temp_data
        index += 1
        print('Finish Aligning pod:', node_name)
    print('We collect {} pods as the output of BERT model.'.format(log_sequence.shape))
    output_dir = os.path.join(args.output_dir, '{}_golden_signal_{}_level_{}_removed_scale_{}.npy'.format(args.dataset, args.level, args.window_size, scale))
    print('Saving the aligned sequence in {}'.format(output_dir))
    print('It takes ', time.time()- begin_time, 'seconds to finish alignment.')
    save_model_dir = os.path.join(args.save_dir, '{}_{}_{}_golden_frequency.npy'.format(args.level, args.dataset, args.window_size))
    ct = np.arange(start_time, end_time, args.step_size * 60, dtype=int)
    processed_data = {
        'KPI_Label': jlabel,
        'Node_Name': node_names,
        'KPI_Feature': 'golden_signal',
        'Sequence': log_sequence,
        'time': ct,
        'step_size': args.step_size,
        'window_size': args.window_size
    }
    print('Shape of step size:', args.step_size, ' and Shape of window size:', args.window_size)
    np.save(output_dir, processed_data)
