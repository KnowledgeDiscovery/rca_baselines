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
    ## for Nezha, we provide the demo code for the case 20240124. Due to inconsistant filename for each case, you may need to change the name of the folder for each case accordingly. 
##### If you encounter the error regarding "name 'LIBSPOT' is not defined", please double-check if you are running the code in the directory of FastPC. We observe such an error if the command is 'python FastPC/test_FastPC_pod_log.py' running in the directory of './rca_baselines/Baseline/offline/'.
2. **Online Baselines**:
   #### Step 1: Run the following command to generate the intialize DAG graph:
    ```
        python main_dag.py
    ```
    Notice that you need to change the flags, including dataset name, dataset path and output directory. The default value is for case 20211203.
   #### Step 2: Run the following command to evaluate Notears or GOLEM:
    - NOTEARS<sup>* </sup> and GOLEM<sup>*</sup>
    ```
        python Test_notears_Golem.py --method NOTEARS
    ```
    Notice that you need to change the flags, including dataset name, dataset path, output directory and method. The default value is for case 20211203.
