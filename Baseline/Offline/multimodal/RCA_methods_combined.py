import numpy as np
import pandas as pd
from sklearn import preprocessing
import os
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from causalnex.structure.notears import from_pandas
from pyrca.analyzers.ht import HT, HTConfig
from pyrca.analyzers.epsilon_diagnosis import EpsilonDiagnosis, EpsilonDiagnosisConfig
from pyrca.analyzers.rcd import RCD, RCDConfig
# from pyrca.analyzers
import networkx as nx
import argparse


def remove_cycles_from_adjacency_matrix(adj_matrix: pd.DataFrame) -> pd.DataFrame:
    G = nx.from_pandas_adjacency(adj_matrix, create_using=nx.DiGraph())
    while True:
        try:
            cycle = nx.find_cycle(G, orientation='original')
            G.remove_edge(*cycle[0][:2])

        except nx.NetworkXNoCycle:
            break
    adj_matrix_no_cycles = nx.to_pandas_adjacency(G, dtype=int)
    adj_matrix_no_cycles = adj_matrix_no_cycles.reindex_like(adj_matrix)

    print("Now, the adjacency matrix does not have cycles.")
    return adj_matrix_no_cycles



def main(args):
    model_name = args.model
    data_name = args.case
    metric_data = {}
    columns_common = {}
    metric_path = '../data/{}'.format(data_name)
    if data_name == '20220606':
        label = 'reviews-v3'
    elif data_name == '20210517' or data_name == '20210524':
        label = 'Book_Info_product'
    elif data_name == '20211203':
        label = 'ratings.book-info.svc.cluster.local:9080/*'
    elif data_name == '20240215':
        label = 'pod usage'
    elif data_name == '20240124':
        label = 'scenario8_app_request'
    elif data_name == '20231207':
        label = 'book_info'
    elif data_name == '20231221':
        label = 'book_info'
    elif data_name == '20240115':
        label = 'book_info'
    else:
        raise ValueError('Invalid data_name')

    if data_name in ['20220606', '20210517', '20210524', '20211203', '20231207']:
        POD_METRIC_FILE = {'log_golden_signal': 1, 'log_frequency': 1, 'cpu_usage': 1, 'memory_usage': 1, 'rate_received_packets': 1, 'rate_transmitted_packets': 1,
                           'received_bandwidth': 1, 'transmit_bandwidth': 1}
        log_label = 'ratings.book-info.svc.cluster.local:9080/*'
    elif data_name in ['20231207']:
        POD_METRIC_FILE = {'cpu_usage': 1, 'memory_usage': 1, 'rate_received_packets': 1, 'rate_transmitted_packets': 1, 'rate_storage_iops': 1,
                           'received_bandwidth': 1, 'transmit_bandwidth': 1, 'log_golden_signal': 1,
                           'log_frequency': 1}
        log_label = 'book_info'
    elif data_name in ['20240124']:
        POD_METRIC_FILE = {'log_golden_signal': 1, 'log_frequency': 1, 'cpu_usage': 1, 'disk_used': 1, 'diskio_reads': 1, 'diskio_writes': 1, 'memory_used': 1,
                           'netstat_established': 1, 'swap_used': 1}
        log_label = 'ratings.book-info.svc.cluster.local:9080/*'
    elif data_name in ['20240215']:
        POD_METRIC_FILE = {'pod_cpu_limit': 1, 'pod_cpu_usage_total': 1, 'pod_cpu_utilization_over_pod_limit': 1, 'pod_memory_limit': 1,
                           'pod_memory_utilization_over_pod_limit': 1, 'pod_memory_working_set': 1, 'pod_network_rx_bytes': 1, 'pod_network_tx_bytes':1,
                           'log_golden_signal': 1, 'log_frequency': 1}
        log_label = 'book_info'
    elif data_name in ['20240115']:
        POD_METRIC_FILE = {'cpu_usage': 1, 'memory_usage': 1, 'rate_received_packets': 1, 'rate_transmitted_packets': 1,
                           'rate_storage_iops': 1, 'received_bandwidth': 1, 'transmit_bandwidth': 1, 'log_golden_signal': 1,
                           'log_frequency': 1}
        log_label = 'ratings.book-info.svc.cluster.local:9080/*'
    elif data_name in ['20231221']:
        POD_METRIC_FILE = {'cpu_usage': 1, 'memory_usage': 1, 'rate_received_packets': 1, 'rate_transmitted_packets': 1, 'rate_storage_iops': 1,
                           'received_bandwidth': 1, 'transmit_bandwidth': 1, 'log_golden_signal': 1, 'log_frequency': 1}
        log_label = 'ratings.book-info.svc.cluster.local:9080/*'
    else:
        raise ValueError('Invalid data_name')

    pathset = "./output/"
    if not (os.path.exists(pathset)):
        os.mkdir(pathset)

    for metric, weight in POD_METRIC_FILE.items():
        if metric in ['log_PCA', 'log_golden_signal', 'log_frequency']:
            metric_file = '{}/pod_level_{}.npy'.format(metric_path, metric)
            metric_data[metric] = np.load(metric_file, allow_pickle=True).item()
            # log_label = 'ratings.book-info.svc.cluster.local:9080/*'
            if len(metric_data[metric].keys()) == 1:
                if log_label != label:
                    metric_data[metric][label] = metric_data[metric][log_label]
                    del metric_data[metric][log_label]
            else:
                 metric_data[metric][label] = metric_data[metric]
            metric_data[metric][label]['Pod_Name'] = metric_data[metric][label]['Node_Name']
            del metric_data[metric][label]['Node_Name']
            metric_data[metric][label]['Sequence'] = metric_data[metric][label]['Sequence'].squeeze().T
            metric_data[metric][label]['KPI_Feature'] = [metric_data[metric][label]['KPI_Feature']]
            if columns_common:
                columns_common = list(set(metric_data[metric][label]['Pod_Name']).intersection(columns_common))
            else:
                columns_common = list(metric_data[metric][label]['Pod_Name'])
        else:
            metric_file = '{}/pod_level_data_{}.npy'.format(metric_path, metric)
            metric_data[metric] = np.load(metric_file, allow_pickle=True).item()
            if columns_common:
                columns_common = list(set(metric_data[metric][label]['Pod_Name']).intersection(columns_common))
            else:
                columns_common = list(metric_data[metric][label]['Pod_Name'])

    index_data = {}
    metric_names = []
    metric_weight_assigned = []
    for metric, weight in POD_METRIC_FILE.items():
        index_data[metric] = [metric_data[metric][label]['Pod_Name'].index(x) for x in columns_common]
        metric_names = metric_names + [metric]
        metric_weight_assigned = metric_weight_assigned + [weight]

    metric_weight = np.zeros((len(POD_METRIC_FILE), 1))
    metric_id = 0
    final_root_results = {}

    for metric in metric_names:
        print('For metric:', metric)
        data = metric_data[metric]
        X = data[label]['Sequence']
        index = index_data[metric]
        # Preprocessing to reduce the redundant samples
        if X.shape[0] // 100 < 100:
            patch = 20
        else:
            patch = 100
        sample = X.shape[0] // patch
        X = X[:patch * sample, :]
        X = np.sum(X.reshape((-1, patch, X.shape[1])), axis=1)
        X_metric = X[:, index]
        X_metric = preprocessing.normalize(X_metric, axis=0, norm='l1')
        X = np.append(X_metric, X[:, -1].reshape(-1, 1), axis=1)
        columns = list(columns_common) + data[label]['KPI_Feature']

        std = np.std(X[:, :-1], axis=0)
        idx_std = [i for i, x in enumerate(std > 1e-5) if x]
        if len(idx_std) == 0:
            metric_weight[metric_id] = 0
            metric_id = metric_id + 1
            print(metric, ' all pods are all constant or quasi-constant')
            continue

        selector = VarianceThreshold(threshold=0)
        X_var = selector.fit_transform(X[:, :-1])
        idx = selector.get_support(indices=True)
        # print('X shape after variance: ', X_var.shape)
        if X_var.shape[1] < 1:
            metric_weight[metric_id] = 0
            metric_id = metric_id + 1
            print(metric, ' all pods are all constant or quasi-constant')
            continue

        mask = np.full(len(columns_common), False, dtype=bool)
        mask[idx] = True
        idx = list(idx) + [X.shape[1] - 1]
        X = X[:, idx]
        columns = [columns[i] for i in idx]
        X = pd.DataFrame(X, columns=columns)
        if model_name == 'circa':
            sm = from_pandas(X)
            estimated_matrix = nx.to_pandas_adjacency(sm)
            quantile_value = np.quantile(estimated_matrix.values.flatten(), 0.95)
            estimated_matrix = (estimated_matrix > quantile_value).astype(int)
            estimated_matrix = remove_cycles_from_adjacency_matrix(estimated_matrix)
            # estimated_matrix.to_csv("{}_adjacency.csv".format(metric))

        X_train, X_test = train_test_split(X, test_size=0.6, shuffle=False)

        X.insert(0, 'time', pd.date_range(start='2024-01-01', periods=len(X), freq='D'))

        X['time'] = X['time'].astype('int64') // 1_000_000_000
        X.columns = [f"{col}_cpu" if i < 9 else f"{col}_memory" for i, col in enumerate(X.columns)]


        X.iloc[:, 1:] = (X.iloc[:, 1:] - X.iloc[:, 1:].min()) / (X.iloc[:, 1:].max() - X.iloc[:, 1:].min())

        if model_name == 'rcd':
            model = RCD(config=RCDConfig(k=3,alpha_limit=0.5))
            results = model.find_root_causes(X_train, X_test).to_list()
            print(results)
        elif model_name == 'circa':
            model = HT(config=HTConfig(graph=estimated_matrix, root_cause_top_k=10))
            model.train(X_train)
            results = model.find_root_causes(X_test, metric_data[metric][label]['KPI_Feature'][0], True).to_list()
        elif model_name == 'epsilon_diagnosis':
            model = EpsilonDiagnosis(config=EpsilonDiagnosisConfig(root_cause_top_k=10))
            model.train(X)
            results = model.find_root_causes(X).to_list()
        else:
            raise ValueError('Invalid model_name')

        root_causes = []
        for result in results:
            root_causes.append([result['root_cause'], result['score']])
        if not os.path.exists('./{}_results'.format(model_name)):
            os.mkdir('./{}_results'.format(model_name))
        if not os.path.exists('./{}_results/{}'.format(model_name, data_name)):
            os.mkdir('./{}_results/{}'.format(model_name, data_name))

        root_causes = pd.DataFrame(root_causes)
        root_causes.columns = [['root_cause', 'score']]
        root_causes.to_csv("./{}_results/{}/{}_{}_{}_root_cause.csv".format(model_name, data_name, metric, model_name, data_name),
                           index=False)
        final_root_results[metric] = root_causes
    concatenated_df = pd.concat(final_root_results.values(), ignore_index=True)
    concatenated_df.to_csv("./{}_results/{}/concated_df.csv".format(model_name, data_name), index=False)
    concatenated_df = pd.read_csv("./{}_results/{}/concated_df.csv".format(model_name, data_name))
    aggregated_df = concatenated_df.groupby('root_cause')['score'].sum().reset_index()
    aggregated_df = aggregated_df.sort_values(by='score', ascending=False)
    aggregated_df.to_csv("./{}_results/{}/final_{}_{}_root_cause.csv".format(model_name, data_name, model_name, data_name), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Baro')
    parser.add_argument("-case", type=str, default='20240115', help="case of the dataset")
    parser.add_argument("-model", type=str, default='rcd', help="model name, [rcd, circa, epsilon_diagnosis], default is rcd")
    parser.set_defaults(validation=True)
    args = parser.parse_args()
    main(args)






