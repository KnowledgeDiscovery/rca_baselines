import math
import sys
import numpy as np
import pandas as pd
import torch
from castle.algorithms import DAG_GNN, Notears, GOLEM
import time
import rbo
import os
# from sklearn.preprocessing import MinMaxScaler
from read_utils import read_aiops_data
from rca_utils import *
import logging
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Dynotear algorithm')
    parser.add_argument('--dataset', type=str, default='20211203', help='name of the dataset')
    parser.add_argument('--path_dir', type=str, default='../../20211203/', help='path to the dataset')
    parser.add_argument('--output_dir', type=str, default='./20211203_output/', help='path to save the results')
    parser.add_argument('--compressed_data_size', type=int, default=300, help='Individual metric compressed data size')
    parser.add_argument('--method', type=str, default='NOTEARS', help='NOTEARS or GOLEM, default is NOTEARS')
    # Parse the arguments
    args = parser.parse_args()
    #st = time.time()
    dataset = args.dataset
    output_dir = args.output_dir
    path_dirs = args.path_dir
    #Assign weight for each metric: default equal weight
    POD_METRIC_FILE = {'cpu_usage': 1, 'memory_usage': 1, 'rate_transmitted_packets': 1, 'rate_received_packets': 1, 'received_bandwidth': 1, 'transmit_bandwidth': 1}
    metric_data = {}
    columns_common = {}
    if not(os.path.exists(output_dir)):
        os.mkdir(output_dir)

    #KPI label
    if dataset == '20211203':
        label = 'ratings.book-info.svc.cluster.local:9080/*'
    elif dataset == '20220606':
        label = 'reviews-v3'
    elif dataset == '20210524':
        label = 'Book_Info_product'
    elif dataset == '20220517':
        label = 'Book_Info_product'
    else:
        raise 'Incorret Dataset Error'
        
    data_samples = ['0901','0517','0606','1203','0524']
    models = [args.method]

    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    #FileHandler
    file_handler = logging.FileHandler('./{}/baseline_online_model.log'.format(args.output_dir),mode='w')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    # StreamHandler
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)


    for model_name in models:
        logger.info("{} model start!".format(model_name))
        # for data_sample_name in data_samples:
        data_sample_name = args.dataset
        logger.info("{} data sample has started!".format(data_sample_name))
        
        data, metrics, real_rc, label, start_time, end_time = read_aiops_data('./config/data_config_{}_freq_gs.yaml'.format(data_sample_name),'all', path_dirs)
        metric_dags = load_metric_dags('./{}/'.format(output_dir),metrics,90)


        start_point = 20000
        start_time += start_point
        time_window = 2000
        count = 0
        sensors = data[metrics[0]][label]['Pod_Name']
        metric_total_data = {x:'' for x in metrics}
        metric_iters = {x:0 for x in metrics}
        max_iter = 20
        lag = 15
        device = "cuda" if torch.cuda.is_available() else "cpu"
        metric_models = {}
        metric_optimizer = {}
        learning_rate = 0.001
        patch = 100
        alpha = 0.4
        save_path = args.output_dir
        make_dir(save_path)
        ranked_list = []

        converge_count = 0

        t0 = time.time()
        while start_time <= end_time:
            count+=1
            metric_weights = []
            valid_metrics = []
            ind_scores = {}
            top_scores = {}
            metric_batch_data = {x: [] for x in metrics}
            '''
            Individual Causal Discovery
            1. Extreme Value Theory based Individual Causal Score
            2. Metric Weights based on Individual Causal Score
            '''
            for metric in metrics:
                flag, s1_data, s2_data = get_current_batch_data(start_time, time_window, data[metric][label]['time'], data[metric][label]['Sequence'], start_point)
                if flag == False: # if the current batch data is invalid
                    continue
                metric_batch_data[metric].append(s1_data)
                metric_batch_data[metric].append(s2_data)
                if len(metric_total_data[metric])==0:
                    metric_total_data[metric] = torch.cat((torch.from_numpy(s1_data).float(),torch.from_numpy(s2_data).float()))
                else:
                    metric_total_data[metric] = torch.cat((metric_total_data[metric],torch.from_numpy(s2_data).float()))

                ind_scores[metric] = np.array(individual_root_cause_analysis(metric_total_data[metric], sensors, args.compressed_data_size)['ind_score'])
                metric_weights.append([np.max(ind_scores[metric], axis=0)])
                valid_metrics.append(metric)

            if len(metric_weights) <= 0: #no any valid metrics
                break

            weight_metrics_results = calculate_prioritized_metric_weights(metric_weights, valid_metrics)

            '''
            Topological Causal Discovery
            1. Load the incremental causal discovery model of different metrics
            2. Update each model based on new batch data
            '''
            for metric in metrics:
                dj_model = ''
                if model_name == 'NOTEARS':
                    dj_model = Notears(max_iter=300)
                elif model_name == 'GOLEM':
                    dj_model = GOLEM(num_iter=300, device_type=device)

                metric_models[metric] = dj_model


            for metric in metrics:
                dj_model = metric_models[metric]
                s1_batch_data, s2_batch_data = get_current_top_batch_data(metric_batch_data, metric, patch, device)
                combine_data = torch.cat((s1_batch_data,s2_batch_data)).cpu().numpy()
                dj_model.learn(combine_data)
                causal_graph = generate_graph_by_thre(dj_model.causal_matrix, 0.99)
                top_scores[metric] = np.array(topological_root_cause_analysis(metric_total_data[metric].numpy(), causal_graph, sensors)['top_score'])
                metric_dags[metric] = causal_graph

            final_results = integrate_ind_top_causal_score(weight_metrics_results, ind_scores,top_scores, alpha, sensors)
            final_results.to_csv(save_path+"{}_final_causal_score_batch_{}.csv".format(model_name,str(count)),index=False)
            logger.info("current {} batch, top 10 result is {}".format(count,final_results.iloc[:10]))
            rank_per = evaluation_percentile(final_results, real_rc)
            logger.info("rank percentile:{}%".format(rank_per))
            ranked_list.append(np.array(final_results['sensor']))
            if count > 1:
                sim = rbo.RankingSimilarity(ranked_list[count-2],ranked_list[count-1]).rbo()
                logger.info("The similarity of predicted root cause lists between current and previous batches is {}".format(sim))

            logger.info("current {} batch has been done!".format(count))
            if math.isclose(rank_per, 0.0):
                converge_count += 1
                if converge_count >= 3:
                    break
            start_time += time_window
            logger.info("The time cost of the whole process {} is {} min!".format(data_sample_name,(time.time()-t0)/60))
        logger.info("{} model {} data sample has finished!".format(model_name,data_sample_name))
    logger.info("{} model has finished!".format(model_name))







