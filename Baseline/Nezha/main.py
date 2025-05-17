import os.path

from pattern_ranker import *
import argparse
from log_parsing import *

file_path = './'
# print(file_path)
log_path = file_path + '/log/' + str(datetime.datetime.now().strftime(
    '%Y-%m-%d')) + '_nezha.log'
print(log_path)
logger = Logger(log_path, logging.DEBUG, __name__).getlog()


def get_miner(ns):
    template_indir = file_path + '/log_template'
    config = TemplateMinerConfig()
    config.load(file_path + "/log_template/drain3_" + ns + ".ini")
    config.profiling_enabled = False

    path = file_path + '/log_template/' + ns + ".bin"
    persistence = FilePersistence(path)
    template_miner = TemplateMiner(persistence, config=config)

    return template_miner

# def generate_trace_id(log_dir):
#     trace_list = []
#     for file in os.listdir(log_dir):
#         if file.endswith("_messages_structured.csv"):
#             trace_list.append(file[:-24])
#     if not os.path.exists(log_dir + '../traceid/'):
#         os.mkdir(log_dir + '../traceid/')
#     pd.DataFrame(trace_list).to_csv(log_dir + '../traceid/trace_id.csv', index=False,  header=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Nezha')

    parser.add_argument('--ns', default="hipster", help='namespace')
    parser.add_argument('--level', default="service", help='service-level or inner-service level')
    parser.add_argument('--log_dir', default="./20240124/log/", help='the path to log data')
    parser.add_argument('--metric_dir', default="./20240124/Latency/", help='the path to metric data')
    parser.add_argument('--save_dir', default="./20240124/", help='the path to save preprocessed data')
    # parser.add_argument('--level', default="service", help='service-level or inner-service level')
    args = parser.parse_args()
    ns = args.ns
    level = args.level
    save_dir = args.save_dir
    log_dir = args.log_dir
    metric_dir = args.metric_dir
    kpi_file = save_dir + '/kpi_20240124_latency.csv'
    path1 = save_dir + "./20240124-fault_list.json"
    kpi_data = pd.read_csv(kpi_file)
    normal_time1 = str(pd.to_datetime(kpi_data['timeStamp'].iloc[0], unit='s'))
    time_index = int(kpi_data['timeStamp'].shape[0] * 0.6)
    preprocess(log_dir, metric_dir, save_dir)
    file_path = save_dir
    log_template_miner = get_miner(ns)
    inject_list = [path1]
    normal_time_list = [normal_time1]
    if level == "service":
        logger.info("------- Result at service level -------")
        evaluation_pod(normal_time_list, inject_list, ns, log_template_miner, file_path)
    else:
        logger.info("------- Result at inner service level -------")
        evaluation(normal_time_list, inject_list, ns, log_template_miner, file_path)
