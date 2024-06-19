# Baselines 

This folder contains the baseline methods for Lemma-RCA evaluation. We provide several methods for both offline and online settings.

1. **Offline Baselines**: 
    - Dynotears: 
    ```
        python test_Dynotears_pod_metric.py ## for metric data only
        python test_Dynotears_pod_log.py  ## for log data only
        python test_Dynotears_pod_combine.py  ## for both metric and log data
    ```
    - FastPC: 
    ```
        python test_FastPC_pod_metric.py ## for metric data only
        python test_FastPC_pod_log.py  ## for log data only
        python test_FastPC_pod_combine.py  ## for both metric and log data
    ```

    - GOLEM: 
    ```
        python test_GOLEM_pod_metric.py ## for metric data only
        python test_GOLEM_pod_log.py  ## for log data only
        python test_GOLEM_pod_combine.py  ## for both metric and log data
    ```
    - LSTM: 
    ```
        python test_LSTM_pod_metric.py ## for metric data only
        python test_LSTM_pod_log.py  ## for log data only
        python test_LSTM_pod_combine.py  ## for both metric and log data
    ```
    - Nezha:
    ```
        python main.py
    ```

2. **Online Baselines**:
    - NOTEARS<sup>* </sup> and GOLEM<sup>*</sup>
    ```
        python Test_notears_Golem.py
    ```
