import warnings
warnings.filterwarnings('ignore')
import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from utility import *
import pandas as pd
import time
import rbo
import argparse
from baro_algorithm import bocpd, robust_scorer
from pyrca.analyzers.ht import HT, HTConfig
from pyrca.analyzers.epsilon_diagnosis import EpsilonDiagnosis, EpsilonDiagnosisConfig
from pyrca.analyzers.rcd import RCD, RCDConfig
from sklearn.model_selection import train_test_split
import networkx as nx
from causalnex.structure.notears import from_pandas


def remove_cycles_from_adjacency_matrix(adj_matrix: pd.DataFrame) -> pd.DataFrame:
    G = nx.from_pandas_adjacency(adj_matrix, create_using=nx.DiGraph())
    try:
        cycle = nx.find_cycle(G, orientation='original')
        G.remove_edge(*cycle[0][:2])
    except nx.NetworkXNoCycle:
        print("No cycle detected")
    adj_matrix_no_cycles = nx.to_pandas_adjacency(G, dtype=int)
    adj_matrix_no_cycles = adj_matrix_no_cycles.reindex_like(adj_matrix)
    print("Now, the adjacency matrix does not have cycles.")
    return adj_matrix_no_cycles


if __name__ == "__main__":
    # Initialize the parser
    parser = argparse.ArgumentParser(description='Specify analysis parameters')
    parser.add_argument('--dataset', type=str, default='20211203',
                        help='default: 20211203')
    parser.add_argument('--time_window', type=int, default=3000, help='Time window for analysis')
    parser.add_argument('--rca_stride', type=int, default=400, help='Stride for Root Cause Analysis')
    parser.add_argument('--initial_compressed_data_size', type=int, default=500, help='Initial compressed data size')
    parser.add_argument('--topology_compressed_data_size', type=int, default=500,
                        help='Individual log compressed data size')
    parser.add_argument('--individual_log_compressed_data_size', type=int, default=500,
                        help='Individual metric compressed data size')
    parser.add_argument('--individual_metric_compressed_data_size', type=int, default=500,
                        help='Topology compressed data size')
    parser.add_argument('--topK_metrics_num', type=int, default=3, help='Number of top K metrics')
    parser.add_argument('--alpha', type=float, default=0.3, help='individual causal weight')
    parser.add_argument('--dag_thres', type=int, default=90, help='initial causal graph threshold')
    parser.add_argument('--offline_length', type=int, default=57600,
                        help='offline length of data used for individual causal discovery')
    parser.add_argument('--modality', type=str, default='metric-only',
                        help='metric-only | log-only | multimodality')
    parser.add_argument('--method', type=str, default='baro',
                        help='default: baro (baro | epsilon_diagnosis)')

    args = parser.parse_args()
    modality = args.modality
    data_name = args.dataset
    method = args.method
    print('\n\nStarting analysis of {}...'.format(data_name))
    if data_name in ['20220606', '20210517', '20210524', '20211203']:
        if modality == 'metric-only':
            POD_METRIC_FILE = {'cpu_usage': 1, 'memory_usage': 1, 'rate_received_packets': 1, 'rate_transmitted_packets': 1,
                               'received_bandwidth': 1, 'transmit_bandwidth': 1}
        elif modality == 'log-only':
            POD_METRIC_FILE = {'golden_signal': 1, 'frequency': 1}
        else:
            POD_METRIC_FILE = {'cpu_usage': 1, 'memory_usage': 1, 'rate_received_packets': 1, 'rate_transmitted_packets': 1,
                               'received_bandwidth': 1, 'transmit_bandwidth': 1, 'golden_signal': 1, 'frequency': 1}
        log_label = 'ratings.book-info.svc.cluster.local:9080/*'
    elif data_name in ['20240207']:
        POD_METRIC_FILE = {'cpu_usage': 1, 'memory_usage': 1, 'rate_received_packets': 1, 'rate_transmitted_packets': 1,
                           'rate_storage_iops': 1, 'received_bandwidth': 1, 'transmit_bandwidth': 1, }
        if modality != 'metric-only':
            modality = 'metric-only'
            print('Changing modality to metric-only as there is no log data available!')
    elif data_name in ['20231207']:
        if modality == 'metric-only':
            POD_METRIC_FILE = {'cpu_usage': 1, 'memory_usage': 1, 'rate_received_packets': 1, 'rate_transmitted_packets': 1,
                               'rate_storage_iops': 1, 'received_bandwidth': 1, 'transmit_bandwidth': 1, }
        elif modality == 'log-only':
            POD_METRIC_FILE = {'golden_signal': 1, 'frequency': 1}
        else:
            POD_METRIC_FILE = {'cpu_usage': 1, 'memory_usage': 1, 'rate_received_packets': 1, 'rate_transmitted_packets': 1,
                               'rate_storage_iops': 1, 'received_bandwidth': 1, 'transmit_bandwidth': 1, 'golden_signal': 1, 'frequency': 1
                               }
        log_label = 'book_info'
    elif data_name in ['20240124']:
        if modality == 'metric-only':
            POD_METRIC_FILE = {'cpu_usage': 1, 'disk_used': 1, 'diskio_reads': 1, 'diskio_writes': 1, 'memory_used': 1,
                               'netstat_established': 1, 'swap_used': 1}
        elif modality == 'log-only':
            POD_METRIC_FILE = {'golden_signal': 1, 'frequency': 1}
        else:
            POD_METRIC_FILE = {'cpu_usage': 1, 'disk_used': 1, 'diskio_reads': 1, 'diskio_writes': 1, 'memory_used': 1,
                               'netstat_established': 1, 'swap_used': 1, 'golden_signal': 1, 'frequency': 1}
        log_label = 'ratings.book-info.svc.cluster.local:9080/*'
    elif data_name in ['20240215']:
        if modality == 'metric-only':
            POD_METRIC_FILE = {'pod_cpu_limit': 1, 'pod_cpu_usage_total': 1, 'pod_cpu_utilization_over_pod_limit': 1,
                               'pod_memory_limit': 1, 'pod_memory_utilization_over_pod_limit': 1, 'pod_memory_working_set': 1,
                               'pod_network_rx_bytes': 1, 'pod_network_tx_bytes': 1}
        elif modality == 'log-only':
            POD_METRIC_FILE = {'golden_signal': 1, 'frequency': 1}
        else:
            POD_METRIC_FILE = {'pod_cpu_usage_total': 1, 'pod_memory_limit': 1, 'pod_cpu_limit': 1,
                                'pod_memory_utilization_over_pod_limit': 1,  'pod_cpu_utilization_over_pod_limit': 1,
                               'pod_memory_working_set': 1, 'pod_network_rx_bytes': 1, 'pod_network_tx_bytes': 1,
                               'golden_signal': 1, 'frequency': 1}
        log_label = 'book_info'
    elif data_name in ['20240115']:
        if modality == 'metric-only':
            POD_METRIC_FILE = {'cpu_usage': 1, 'memory_usage': 1, 'rate_received_packets': 1, 'rate_transmitted_packets': 1,
                               'rate_storage_iops': 1, 'received_bandwidth': 1, 'transmit_bandwidth': 1}
        elif modality == 'log-only':
            POD_METRIC_FILE = {'golden_signal': 1, 'frequency': 1}
        else:
            POD_METRIC_FILE = {'cpu_usage': 1, 'memory_usage': 1, 'rate_received_packets': 1, 'rate_transmitted_packets': 1,
                               'rate_storage_iops': 1, 'received_bandwidth': 1, 'transmit_bandwidth': 1, 'golden_signal': 1,
                               'frequency': 1}
        log_label = 'ratings.book-info.svc.cluster.local:9080/*'
    elif data_name in ['20231221']:
        if modality == 'metric-only':
            POD_METRIC_FILE = {'cpu_usage': 1, 'memory_usage': 1, 'rate_received_packets': 1, 'rate_transmitted_packets': 1,
                               'rate_storage_iops': 1, 'received_bandwidth': 1, 'transmit_bandwidth': 1, }
        elif modality == 'log-only':
            POD_METRIC_FILE = {'golden_signal': 1, 'frequency': 1}
        else:
            POD_METRIC_FILE = {'cpu_usage': 1, 'memory_usage': 1, 'rate_received_packets': 1, 'rate_transmitted_packets': 1,
                               'rate_storage_iops': 1, 'received_bandwidth': 1, 'transmit_bandwidth': 1, 'golden_signal': 1, 'frequency': 1,
                               }
        log_label = 'ratings.book-info.svc.cluster.local:9080/*'
    else:
        raise ValueError('Invalid data_name')

    dag_thres = args.dag_thres

    data_sample_name = data_name[-4:]
    config_path = f'./config/data_config_{data_sample_name}_freq_gs.yaml'
    node_data_path = f"./node_data/{data_sample_name}/node_level_data_cpu_usage.npy"

    data, metrics, real_rc, label, online_start, mul_kpi, failure_time, time_stamp, node_time_seq = system_data_load(data_sample_name,
                                                                                   config_path, POD_METRIC_FILE, modality)

    start_time = online_start

    cpd_window_size = 3000
    cpd_stride = 150
    cpd_manifold_stride = 5

    dist_list, dist_list_offline = [], []
    old_ind = 0
    cpd_start = 0

    offline_length = args.offline_length

    converge_thre = 3
    converge_sim = 0.9
    topk = 5
    count = 0
    converge_count = 1
    metric_total_data = {x: '' for x in metrics}
    metric_iters = {x: 0 for x in metrics}
    max_iter_graph = 20
    lag = 30
    log_name = './runs/{}/'.format(data_sample_name)
    exp_name = log_name + 'vrca_' + str(time.time()).split('.')[0]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    metric_models = {}
    metric_optimizer = {}
    learning_rate = 0.001
    max_iteration_overall = 50

    # Use arguments
    time_window = args.time_window
    rca_stride = args.rca_stride
    individual_log_compressed_data_size = args.individual_log_compressed_data_size
    individual_metric_compressed_data_size = args.individual_metric_compressed_data_size
    initial_compressed_data_size = args.initial_compressed_data_size
    topology_compressed_data_size = args.topology_compressed_data_size
    topK_metrics_num = args.topK_metrics_num
    alpha = args.alpha

    # parameter for trigger point detection: e.g., beta =2 gamma = 5 means 2*mean + 5*standard_deviation
    beta = 1
    gamma = 4

    patch = 50

    ranked_list = []
    is_cpd = False
    propagate_steps = 20000
    is_output_early = True
    tolerance_time = 600
    padding_way = 'zero'  # zero/mean

    is_save = True
    ind_saving_path = f'./results/{data_sample_name}/individual/'
    top_saving_path = f'./results/{data_sample_name}/toplogical/'
    inte_saving_path = f'./results/{data_sample_name}/integrate/'
    if not os.path.exists(ind_saving_path):
        os.makedirs(ind_saving_path)
    if not os.path.exists(top_saving_path):
        os.makedirs(top_saving_path)
    if not os.path.exists(inte_saving_path):
        os.makedirs(inte_saving_path)

    run_time0 = time.time()
    metric_sensors = {}
    for metric in metrics:
        metric_sensors[metric] = data[metric][label]['Pod_Name']

    ts, cpd_target, time_len, time_stamps = get_cpd_time_seq(node_time_seq, time_stamp, data_sample_name, mul_kpi, label, start_time)

    print("System starts!")

    print("The maximum timestamp for the entire time series of {} is (JST): {}".format(data_sample_name,
                                                                                       datetime.utcfromtimestamp(int(np.max(
                                                                                           time_stamps)) + 3600 * 9).strftime(
                                                                                           '%Y-%m-%d %H:%M:%S')))
    print(
        f"The real failure time of the system failure is (JST): {datetime.utcfromtimestamp(failure_time + 3600 * 9).strftime('%Y-%m-%d %H:%M:%S')}")

    iteration = 0

    while cpd_start + cpd_window_size <= time_len and iteration < max_iteration_overall:

        dist0 = distance_calculation_for_conceutive_iterations(ts, cpd_start, cpd_window_size, cpd_stride,
                                                               cpd_manifold_stride)

        dist_list.append(dist0)
        dist_list_offline.append(dist0)

        alarm_time = change_point_detection_cusum(dist_list, beta, gamma)

        count += 1
        metric_weights = []
        valid_metrics = []
        ind_scores = {}
        top_scores = {}
        metric_batch_data = {x: [] for x in metrics}
        individual_results = {}
        ind_t0 = time.time()

        if is_cpd:
            metric_total_data, metric_batch_data = update_total_data(metrics, start_time, time_window, data, label,
                                                                     metric_batch_data, metric_total_data, online_start,
                                                                     offline_length, is_cpd, cpd_window_size, rca_stride)
            for metric in metrics:
                print('Identifying root cause for metric: {}'.format(metric))
                s1_batch_data, s2_batch_data = get_current_top_batch_data(metric_batch_data, metric, device, patch)
                batch_data = np.concatenate([s1_batch_data, s2_batch_data])
                sensors = data[metric][label]['Pod_Name']
                ## add latency to sensors
                if 'Latency' not in sensors:
                    sensors = np.append(sensors, 'Latency')

                batch_data = pd.DataFrame(batch_data, columns=sensors)
                # anomalies = bocpd(batch_data)
                # # print("Anomalies are detected at timestep:", anomalies[0])
                # results = robust_scorer(batch_data, anomalies=anomalies)

                if method == 'baro':
                    anomalies = bocpd(batch_data)
                    print("Anomalies are detected at timestep:", anomalies[0])
                    results = robust_scorer(batch_data, anomalies=anomalies)
                elif method == 'rcd':
                    X_train, X_test = train_test_split(batch_data, test_size=0.6, shuffle=False)
                    model = RCD(config=RCDConfig(k=3, alpha_limit=0.5))
                    results = model.find_root_causes(X_train, X_test).to_dict()['root_cause_nodes']
                    f_results = []
                    for i in range(len(results)):
                        if results[i][1] is None:
                            f_results.append((results[i][0], 0.01))
                        else:
                            f_results.append(results[i])
                    results = f_results
                    # print(results)
                elif method == 'circa':
                    sm = from_pandas(batch_data)
                    estimated_matrix = nx.to_pandas_adjacency(sm)
                    quantile_value = np.quantile(estimated_matrix.values.flatten(), 0.95)
                    estimated_matrix = (estimated_matrix > quantile_value).astype(int)

                    estimated_matrix = remove_cycles_from_adjacency_matrix(estimated_matrix)

                    estimated_matrix.to_csv("{}_adjacency.csv".format(metric))
                    model = HT(config=HTConfig(graph=estimated_matrix, root_cause_top_k=10))
                    X_train, X_test = train_test_split(batch_data, test_size=0.6, shuffle=False)
                    model.train(X_train)
                    results = model.find_root_causes(X_test, X_test.columns[-1], True).to_list()
                    f_results = []
                    for i in range(len(results)):
                        if results[i]['score'] is None:
                            f_results.append((results[i][0], 0.01))
                        else:
                            f_results.append((results[i]['root_cause'], results[i]['score']))
                    results = f_results
                elif method == 'epsilon_diagnosis':
                    model = EpsilonDiagnosis(config=EpsilonDiagnosisConfig(root_cause_top_k=10))
                    model.train(batch_data)
                    results = model.find_root_causes(batch_data).to_list()
                    f_results = []
                    for i in range(len(results)):
                        if results[i]['score'] is None:
                            f_results.append((results[i][0], 0.01))
                        else:
                            f_results.append((results[i]['root_cause'], results[i]['score']))
                    results = f_results
                else:
                    raise ValueError('Invalid model_name')
                root_causes = []
                for result in results:
                    (root_cause, score) = result
                    root_causes.append([root_cause, score])
                if not os.path.exists('./{}_results'.format(method)):
                    os.mkdir('./{}_results'.format(method))
                if not os.path.exists('./{}_results/{}'.format(method, data_name)):
                    os.mkdir('./{}_results/{}'.format(method, data_name))

                root_causes = pd.DataFrame(root_causes)
                root_causes.columns = [['root_cause', 'score']]
                root_causes.to_csv(
                    "./{}_results/{}/{}_{}_root_cause.csv".format(method, data_name, metric, data_name),
                    index=False)

                individual_results[metric] = root_causes

            final_results = integrate_ind_top_causal_score(individual_results)
            print("current {} batch, top 10 result is {}".format(count, final_results.iloc[:10]))

            confidence_score = 0

            if len(list(final_results['final_score'])) <= 3:
                print("The number of entities are too few to estimate the confidence")
            else:
                confidence_score = confidence_generation(list(final_results['final_score']))
                print('Root cause confidence: ', confidence_score)

            confidence = pd.DataFrame([confidence_score], columns=['confidence'])

            if is_save:
                final_results.to_csv(f"{inte_saving_path}_integrate_causal_score_batch_{count}.csv", index=False)
                confidence.to_csv(f"{inte_saving_path}_integrate_confidence_score_batch_{count}.csv", index=False)

            rank_per = evaluation_percentile(final_results, real_rc)
            print("rank percentile:{}%".format(rank_per))
            ranked_list.append(np.array(final_results['sensor']))

            iteration += 1

            if is_output_early and (start_time + rca_stride) >= (failure_time + tolerance_time):
                early_output_integration_result(individual_results, count, real_rc, is_save, inte_saving_path, start_time)
                is_output_early = False

        if len(alarm_time):
            change_point_timestep = cpd_start + old_ind + alarm_time[0]
            start_time = time_stamps.iloc[change_point_timestep]
            online_start = start_time
            if change_point_timestep > cpd_target:
                is_cpd = True
                print("Online CPD module has detected a trigger point.")
                print("The timestamp for detect change point of {} is (JST): {}".format(data_sample_name,
                                                                                        datetime.utcfromtimestamp(int(
                                                                                            time_stamps.iloc[
                                                                                                change_point_timestep]) + 3600 * 9).strftime(
                                                                                            '%Y-%m-%d %H:%M:%S')))
                print("Start Root Cause Analysis Process.")
                count = 0
            cpd_start += (2 * cpd_window_size)
            dist_list = []
            old_ind = len(dist_list_offline)
        else:
            cpd_start += cpd_stride

        if count > 1 and is_cpd:
            sim = rbo.RankingSimilarity(ranked_list[count - 2][:topk], ranked_list[count - 1][:topk]).rbo()

            sim_top_one = rbo.RankingSimilarity(ranked_list[count - 2][:1], ranked_list[count - 1][:1]).rbo()
            print(
                f"The localization time of this batch is {datetime.utcfromtimestamp(start_time + 3600 * 9).strftime('%Y-%m-%d %H:%M:%S')}")

            print("The similarity of predicted root cause lists between current and previous batches is {}".format(sim))
            if sim >= converge_sim:
                converge_count += 1
                if converge_count >= converge_thre:
                    print(
                        f"The localization time of current batch is {datetime.utcfromtimestamp(start_time + 3600 * 9).strftime('%Y-%m-%d %H:%M:%S')}")
                    if is_output_early:
                        print(
                            f"The early localization time of the root cause is the same, since the algorithm converges within the tolerance time of early analysis.")
                    print(f"Total running time for online root cause analysis is {(time.time() - run_time0) / 60} minutes!")
                    break
            else:
                converge_count = 1

        if is_cpd:
            start_time += rca_stride

