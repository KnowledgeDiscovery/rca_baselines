#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append('../')
import os
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from models import *
from preprocess.preprocessing import FeatureExtractor, Vectorizer, Iterator
from preprocess.data_load import *
from datetime import datetime
from argparse import ArgumentParser
from preprocess.dataset import split_train_test


def arg_parser():
    """
    add parser parameters
    :return:
    """
    parser = ArgumentParser()

    parser.add_argument("--dataset_name", default='cpf', help="which dataset to use"),
    parser.add_argument("--dataset", default='0524', help="which dataset to use"),
    parser.add_argument("--level", default='pod', help="pod or node"),
    parser.add_argument("--pod_name", default='all', help="which pod to use, or all"),
    parser.add_argument("--log_dir", default="./data/0524/log_data/", metavar="DIR", help="log data directory")
    parser.add_argument("--output_dir", default="./output/cpf_log_result/", metavar="DIR", help="output directory")
    parser.add_argument("--data_dir", default="./output/cpf_log_result/", metavar="DIR", help="train test data directory")
    parser.add_argument("--split_time", default='2022-12-02 22:00:00', type=str, 
                        help="split data before it as train, after as test, overwrite train size if exists")
    parser.add_argument("--window_type", default='sliding', type=str, choices=["sliding", "session"], help="generating log sequence")
    parser.add_argument('--window_size', default=30, type=float, help='window size(mins)')
    parser.add_argument('--step_size', default=0.5, type=float, help='step size(mins)')
    parser.add_argument('--train_size', default=0.4, type=float, help="train size", metavar="float or int")
    # parser.add_argument("--time_format", default='%Y-%m-%dT%H:%M:%S.%fZ', type=str, help="input time format")
    parser.add_argument("--time_format", default='%Y-%m-%dT%H:%M:%S.%f+00:00', type=str, help="input time format")
   
    parser.add_argument("--model", default='PCA', type=str, choices=['PCA', 'InvariantsMiner', 'IsolationForest', 'Deeplog', 'LogClustering'],
                        help="anomaly detection model")
    return parser

# deeplog setting
batch_size = 32
hidden_size = 32
num_directions = 2
topk = 5
train_ratio = 0.4
window_size = 10
epoches = 5
num_workers = 2
device = 0
threshold = 0.2

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


if __name__ == '__main__':
    parser = arg_parser()
    args = parser.parse_args()
    dataset = args.dataset
    level = args.level
    data_dir = args.data_dir
    pod_name = args.pod_name
    pod_names = []
    files = [f for f in os.listdir(args.log_dir) if os.path.isfile(os.path.join(args.log_dir, f))]
    # files = [f for f in file_list if os.path.isfile(os.path.join(args.log_dir, f))]
    for f in files:
        if '_messages_structured' in f:
            pod_name = f.replace('_messages_structured.csv', '')
            df = pd.read_csv(os.path.join(args.log_dir, f), encoding='Latin-1', nrows=1000)
            length = df.shape[0]
            length_threshold = 500
            if length < length_threshold:
                print('The number of records in {} is less than {}. Skipping it!'.format(os.path.join(args.log_dir, f), length_threshold))
                continue
            else:
                pod_names.append(pod_name)

    print('%d pods to process' % len(pod_names))

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    processed = [f for f in os.listdir(args.output_dir) if os.path.isdir(os.path.join(args.output_dir, f))]
    for pod_name in pod_names:
        if pod_name in processed:
            print('%s exists, skip' % pod_name)
            # continue
        
        pod_output_dir = os.path.join(args.output_dir, pod_name)
        if not os.path.exists(pod_output_dir):
            os.makedirs(pod_output_dir, exist_ok=True)

        log_file = pod_name + '_messages_structured.csv'
        split_time = args.split_time
        time_format = args.time_format
        if pod_name == 'storage_log':
            time_format = '%Y-%m-%d %H:%M:%S+09:00'
            split_time = '2021-12-03 08:00:00'

        split_train_test(data_dir=args.log_dir,
                         output_dir=pod_output_dir,
                         log_file=log_file,
                         dataset_name=args.dataset_name,
                         window_type=args.window_type,
                         window_size=args.window_size,
                         step_size=args.step_size,
                         train_size=args.train_size,
                         time_format=time_format,
                         split_time=split_time)


    print('Total pod number: %d' % len(pod_names))
    PCA_feature = {}
    for pod_name in pod_names:
        print('Evaluating %s' % pod_name)
        run_models = [args.model]
        pod_dir = os.path.join(data_dir, pod_name)
        pod_out_dir = os.path.join(args.output_dir, pod_name)
        if not os.path.exists(pod_out_dir):
            os.makedirs(pod_out_dir, exist_ok=True)

        train_file = os.path.join(pod_dir, 'train')
        test_file = os.path.join(pod_dir, 'test_normal')
        test_timestamp_file = os.path.join(pod_dir, 'test_normal_timestamp')
        train_timestamp_file = os.path.join(pod_dir, 'train_timestamp')
        x_tr = load_data(train_file)
        x_te = load_data(test_file)
        # t_te = np.array([int(t) for t in load_data(test_timestamp_file)])
        t_tr = np.array([int(t) for t in load_data(train_timestamp_file)])
        # train_timestamp = np.array([t for t in load_data(train_timestamp_file)])
        y_train = np.zeros(x_tr.shape[0])   # for evaluation test
        y_test = np.ones(x_te.shape[0])     # for evaluation test

        for _model in run_models:
            print('Using model {}:'.format(_model))
            if _model == 'PCA':
                feature_extractor = FeatureExtractor()
                x_train = feature_extractor.fit_transform(x_tr, term_weighting='tf-idf',
                                                            normalization='zero-mean',
                                                            oov=True)
                model = PCA()
                model.fit(x_train)

            elif _model == 'InvariantsMiner':
                feature_extractor = FeatureExtractor()
                x_train = feature_extractor.fit_transform(x_tr, oov=True)
                model = InvariantsMiner(epsilon=0.5)
                model.fit(x_train)

            elif _model == 'LogClustering':
                feature_extractor = FeatureExtractor()
                x_train = feature_extractor.fit_transform(x_tr, term_weighting='tf-idf', oov=True)
                model = LogClustering(max_dist=0.3, anomaly_threshold=0.3)
                model.fit(x_train[y_train == 0, :]) # Use only normal samples for training

            elif _model == 'IsolationForest':
                feature_extractor = FeatureExtractor()
                x_train = feature_extractor.fit_transform(x_tr, oov=True)
                model = IsolationForest(random_state=2019, max_samples=0.9999, contamination=0.03,
                                        n_jobs=4)
                model.fit(x_train)

            elif _model == 'Deeplog':
                (x_train, window_y_train, y_train), (x_test, window_y_test, y_test) = process_data_deeplog(x_tr, y_train,
                                                                                                            x_te, y_test,
                                                                                                            window_size=window_size)
                feature_extractor = Vectorizer()
                train_dataset = feature_extractor.fit_transform(x_train, window_y_train, y_train)
                test_dataset = feature_extractor.transform(x_test, window_y_test, y_test)

                train_loader = Iterator(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers).iter
                test_loader = Iterator(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers).iter

                model = DeepLog(num_labels=feature_extractor.num_labels,
                                hidden_size=hidden_size,
                                num_directions=num_directions,
                                topk=topk,
                                device=device,
                                threshold=threshold)
                model.fit(train_loader, epoches)

                y_pred = model.predict(test_loader)
                y_pred_score = model.predict_prob(test_loader)
            if _model == 'PCA':
                components = np.dot(x_train, model.components)
                proj_C = model.proj_C
                # we use the first components as the final output
                y_pred_score = components[:, 0]
                y_pred = components[:, 0]
                distance = model.predict_prob(x_train)

                print('Shape of components:', components.shape)
                print('Shape of proj_C:', proj_C.shape)
                print('Shape of x_train', x_train.shape)
                print('Shape of distance', distance.shape)
            elif _model != 'Deeplog':
                x_test = feature_extractor.transform(x_te)
                y_pred = model.predict(x_test)
                y_pred_score = model.predict_prob(x_test)
            
            y_pred_abnormal = np.ones((x_train.shape[0]), dtype=bool)
            print('%d anomalies detected from %d instances!\n' % (y_pred_abnormal.shape[0], len(y_pred)))
            # ano_times = pd.to_datetime(t_te[y_pred_abnormal], unit='s')
            ano_times = pd.to_datetime(t_tr[y_pred_abnormal], unit='s')

            # output predicted anomaly score
            f_pred = os.path.join(pod_out_dir, _model+'_results-2.csv')
            with open(f_pred, 'w') as f:
                f.write('Time,Pod,Score,Anomaly,Distance\n')
                # for ts, score, anomaly in zip(t_te, y_pred_score, y_pred):
                for ts, score, anomaly, dis in zip(t_tr, y_pred_score, y_pred, distance):
                    ts = pd.to_datetime(ts, unit='s')
                    if pod_name == 'storage_log':
                        ts_str = datetime.strftime(ts, '%Y-%m-%dT%H:%M:%S+09:00')
                    else:
                        ts_str = datetime.strftime(ts, '%Y-%m-%dT%H:%M:%S+00:00')
                    f.write('%s,%s,%.32f,%d, %.32f\n' % (ts_str, pod_name, score, anomaly, dis))
            # print('result saved to file: %s' % f_pred)
            time_stamp = t_tr
            # 'Score' corresponding to the first component of PCA algorithm.
            data = {'starttime': time_stamp, 'Pod': pod_name, 'Score': y_pred_score, 'Anomaly': y_pred, 'Distance': distance}
            # print('Shape of timestamp:', train_timestamp.shape)
            print('Shape of Score:', y_pred_score.shape)
            print('Shape of anomaly:', y_pred.shape)
            # np.save(f_pred, data)
            print('result saved to file: %s' % f_pred)
            PCA_feature[pod_name] = data
    print('Finish PCA extraction')


    node_names= pod_names
    # Align the time sequence
    data = PCA_feature
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
        temp_data[idx] = np.array(item['Score']).reshape(-1, 1)
        start_idx = int((item['starttime'][0] - start_time) //  (args.step_size * 60))
        end_idx = int((item['starttime'][-1] - start_time) //  (args.step_size * 60))
        print('For Entity:{}, the start_idx is {} and the end_idx is {}'.format(node_name, start_idx, end_idx))
        for the_index in range(length):
            if the_index not in idx:
                the_correct_idx = find_closest_value_index_sorted(idx, the_index)
                temp_data[the_index] = item['Score'][the_correct_idx]
        print('The number of zeros in embedding is ', length - np.nonzero(temp_data.sum(axis=1))[0].shape[0])
        log_sequence[index, :, :] = temp_data
        index += 1
    print(log_sequence.sum(axis=1).reshape(-1,))
    output_dir = os.path.join(args.output_dir, '{}_log_sequence_PCA_{}_level_removed.npy'.format(dataset, level))
    print('Saving the aligned sequence in {}'.format(output_dir))
    
    ct = np.arange(start_time, end_time, args.step_size * 60, dtype=int)
    jlabel = 'ratings.book-info.svc.cluster.local:9080/*'
    processed_data = {}
    processed_data = {
        'KPI_Label': jlabel,
        'Node_Name': node_names,
        'KPI_Feature': 'PCA',
        'Sequence': log_sequence,
        'time': ct
    }
    np.save(output_dir, processed_data)

