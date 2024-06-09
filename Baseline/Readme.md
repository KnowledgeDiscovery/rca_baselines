# Baseline 

This folder contains the baseline methods for Lemma-RCA evaluation. We provide several methods for both offline and online settings.

1. **Offline Baseline**: 
    - Dynotears: 
    ```
        python test_gnn_pod.py ## for metric data only
        python test_gnn_pod_log.py  ## for log data only
        python test_gnn_pod_combine.py  ## for both metric and log data
    ```
    - FastPC: 
    ```
        python test_gnn_pod.py ## for metric data only
        python test_gnn_pod_log.py  ## for log data only
        python test_gnn_pod_combine.py  ## for both metric and log data
    ```
    - GNN: 
    ```
        python test_gnn_pod.py ## for metric data only
        python test_gnn_pod_log.py  ## for log data only
        python test_gnn_pod_combine.py  ## for both metric and log data
    ```
    - GOLEM: 
    ```
        python test_gnn_pod.py ## for metric data only
        python test_gnn_pod_log.py  ## for log data only
        python test_gnn_pod_combine.py  ## for both metric and log data
    ```
    - LSTM: 
    ```
        python test_gnn_pod.py ## for metric data only
        python test_gnn_pod_log.py  ## for log data only
        python test_gnn_pod_combine.py  ## for both metric and log data
    ```
    - Nezha:
    ```
        python main.py
    ```

2. **Online Baseline**:
    - Notears and GOLEM:
    ```
        python baseline_final.py
    ```
