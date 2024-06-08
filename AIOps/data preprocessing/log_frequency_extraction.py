import sys
sys.path.append('../')
import os
from argparse import ArgumentParser
import pandas as pd
import numpy as np
import pickle as pkl
# import time

## Compute the frequency of each pod with the given pod templates and generate 3D log data with shape:
# 1 X  Number of time windows X  Number of pods.

class SlidingWindow():
    def __init__(self):
        pass

    def sliding_window(self, df, window_size, step_size):
        """
        :param df: dataframe columns=[timestamp, label, eventid, time duration, lineid]
        :param window_size: seconds,
        :param step_size: seconds
        :return: dataframe columns=[eventids, label, eventids, time durations, lineids]
        """
        log_size = df.shape[0]
        label_data, time_data, line_data = df.iloc[:, 1], df.iloc[:, 0], df.iloc[:, 4]
        logkey_data, deltaT_data = df.iloc[:, 2], df.iloc[:, 3]
        new_data = []
        occurrence = []
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

        for (start_index, end_index) in start_end_index_pair:
            dt = deltaT_data[start_index: end_index].values
            dt[0] = 0
            new_data.append([
                time_data[start_index: end_index].values,
                max(label_data[start_index:end_index]),
                logkey_data[start_index: end_index].values,
                dt,
                line_data[start_index: end_index].values,
            ])
            occurrence.append(logkey_data[start_index: end_index].values.shape[0])
        new_df = pd.DataFrame(new_data, columns=df.columns)
        occurrence_df = pd.DataFrame({'occurrence': occurrence})
        new_df = new_df.join(occurrence_df)
        assert len(start_end_index_pair) == len(new_data)
        print('\nThere are %d instances (sliding windows) in this dataset' % len(start_end_index_pair))
        return new_df

        
def arg_parser(level='node'):
    """
    add parser parameters
    :return:
    """
    parser = ArgumentParser()
    parser.add_argument("--log_dir", default="./0524/log_data/", metavar="DIR", help="log data directory")
    parser.add_argument("--output_dir", default="./output/log_frequency.npy", metavar="DIR", help="output directory")
    parser.add_argument("--dataset_name", default='cpf', help="which dataset to use"),
    parser.add_argument("--node_name", default='all', help="which node to use, or all"),
    parser.add_argument("--window_type", default='sliding', type=str, choices=["sliding", "session"],
                        help="generating log sequence")
    parser.add_argument('--window_size', default=30, type=float, help='window size(mins)')
    parser.add_argument('--step_size', default=0.5, type=float, help='step size(mins)')
    # parser.add_argument("--time_format", default='%Y-%m-%dT%H:%M:%S.%fZ', type=str,
    #                    help="input time format")
    parser.add_argument("--time_format", default='%Y-%m-%dT%H:%M:%S.%f+00:00', type=str,
                        help="input time format")
    parser.add_argument('--use_small', action='store_true', help='Whether to use small dataset to test the code')
    return parser

def find_closest_value_index_sorted(input_list, target_value):
    left = 0
    right = len(input_list) - 1
    closest_value = None
    closest_index = None

    while left <= right:
        mid = (left + right) // 2
        closest_value = input_list[mid]
        closest_index = mid

        if closest_value == target_value:
            return closest_index
        elif closest_value < target_value:
            left = mid + 1
        else:
            right = mid - 1

    # Check if the closest value is the last element
    if closest_value is None or abs(closest_value - target_value) > abs(input_list[right] - target_value):
        closest_value = input_list[right]
        closest_index = right

    return closest_index


if __name__ == "__main__":
    level = 'node'
    dataset = '0524'
    parser = arg_parser(level)
    args = parser.parse_args()
    largest_file_name = None
    largest_file_size = 0
    node_names = []
    # check the total timestamps and compuate the time length after applying sliding windows algorithms to extract features.
    
    files = [f for f in os.listdir(args.log_dir) if os.path.isfile(os.path.join(args.log_dir, f))]
    print(files)
    for f in files:
        if '_messages_structured' in f:
            node_name = f.replace('_messages_structured.csv', '')
            df = pd.read_csv(os.path.join(args.log_dir,  f), encoding='Latin-1', nrows=1500)
            length = df.size
            length_threshold = 1000
            if length < length_threshold:
                print('The number of records in {} is less than {}. Skipping it!'.format(os.path.join(args.log_dir, f), length_threshold))
                continue
            else:
                node_names.append(node_name)
    
    occurrence_sequence = {}
    for node_name in node_names: 
        data_dir=args.log_dir
        output_dir = os.path.join(args.output_dir, node_name)
        log_file = node_name + '_messages_structured.csv'
        dataset_name=args.dataset_name
        window_type=args.window_type
        window_size=args.window_size
        step_size=args.step_size
        time_format = args.time_format
        print("\nLoading", f'{data_dir}{log_file}')
        df = pd.read_csv(f'{data_dir}{log_file}', dtype=object,encoding='Latin-1')
        df.index.rename("LineId")

        window = SlidingWindow()
        # data preprocess
        df['datetime'] = pd.to_datetime(df["Time"], format=time_format)
        df = df.sort_values(by='datetime', ignore_index=True)
        print('Time span:', min(df['datetime']), '-', max(df['datetime']))

        # do not have label in cpf
        df['Label'] = 0
        df['timestamp'] = df["datetime"].values.astype(np.int64) // 10 ** 9
        df['deltaT'] = df['datetime'].diff() / np.timedelta64(1, 's')
        df['deltaT'].fillna(0)

        window_df = window.sliding_window(df[["timestamp", "Label", "EventId", "deltaT", "LineId"]],
                                            window_size=float(window_size) * 60,
                                            step_size=float(step_size) * 60)
        window_df['starttime'] = window_df['timestamp'].apply(np.min)
        window_df['endtime'] = window_df['timestamp'].apply(np.max)
        window_df = window_df.sort_values(by='starttime')
        print(window_df.head())
        occurrence_sequence[node_name] = window_df[['starttime', 'occurrence']]
    
    # Align the time sequence
    data = occurrence_sequence
    print(data)
    start_time = np.inf
    end_time = 0
    # get the start time and end time
    for key, item in data.items():
        timestamp = item['starttime'].min()
        if timestamp < start_time:
            start_time = timestamp
        timestamp = item['starttime'].max()
        if timestamp > end_time:
            end_time = timestamp
    print(start_time)
    print(end_time)
    length = int((end_time - start_time) // (args.step_size * 60)) + 1
    
    # log_sequence is the output time-series data with shape: # of nodes by # of time windows by 1
    log_sequence = np.zeros((len(node_names), length, 1))
    index = 0
    for node_name in node_names:
        item = data[node_name]
        temp_data = np.zeros((length, 1))
        idx = list(map(lambda x: int((x - start_time)//(args.step_size*60)), item['starttime']))
        temp_data[idx] = np.array(item['occurrence']).reshape(-1, 1)
        start_idx = int((item['starttime'].iloc[0] - start_time) //  (args.step_size * 60))
        end_idx = int((item['starttime'].iloc[-1] - start_time) //  (args.step_size * 60))
        print('For Entity:{}, the start_idx is {} and the end_idx is {}'.format(node_name, start_idx, end_idx))
        for the_index in range(length):
            if the_index not in idx:
                the_correct_idx = find_closest_value_index_sorted(idx, the_index)
                temp_data[the_index] = item['occurrence'][the_correct_idx]
        print('The number of zeros in embedding is ', length - np.nonzero(temp_data.sum(axis=1))[0].shape[0])
        log_sequence[index, :, :] = temp_data
        
        index += 1
    print(log_sequence.sum(axis=1).reshape(-1,))
    output_dir = args.output_dir
    print('Saving the aligned sequence in {}'.format(output_dir))
    ct = np.linspace(start_time, end_time, num=length, endpoint=True, dtype=int)
    jlabel = 'ratings.book-info.svc.cluster.local:9080/*'
    processed_data = {
        'KPI_Label': jlabel,
        'Node_Name': node_names,
        'KPI_Feature': 'occurrence',
        'Sequence': log_sequence,
        'time': ct
    }
    np.save(output_dir, processed_data)

