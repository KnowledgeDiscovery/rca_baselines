# Baselines 

This folder contains the baseline methods for Lemma-RCA datasets evaluation with both single- and multi-modal settings. Note that SWAT and WADI datasets comply with single-modal setting.

    - FastPC: 
    ```
        python test_FastPC_pod_metric.py -case 20240115 ## for case 20240115 metric data only
        python test_FastPC_pod_log.py  -case 20240115  ## for case 20240115 log data only
        python test_FastPC_pod_combine.py  -case 20240115  ## for case 20240115  with both metric and log data
    ```

    - Baro: 
    ```
        cd ./metric_only
        python baro_main_metric.py -case 20240115## for case 20240115 metric data only
        cd ./log_only
        python baro_main_log.py  -case 20240115## for case 20240115 log data only
        cd ./multimodal
        python baro_main_combined.py  -case 20240115 ## case 20240115 with for both metric and log data
    ```


    - RCD: 
    ```
        cd ./metric_only
        python RCA_methods_metric.py -case 20240115 -model rcd ## for metric data only
        cd ./log_only
        python RCA_methods_log.py  -case 20240115 -model rcd ## for log data only
        cd ./multimodal
        python RCA_methods_combined.py  -case 20240115 -model rcd ## for both metric and log data
    ```

    - CIRCA: 
    ```
        cd ./metric_only
        python RCA_methods_metric.py -case 20240115 -model circa ## for metric data only
        cd ./log_only
        python RCA_methods_log.py  -case 20240115 -model circa ## for log data only
        cd ./multimodal
        python RCA_methods_combined.py  -case 20240115 -model circa ## for both metric and log data
    ```

    - epsilon_diagnosis: 
    ```
        cd ./metric_only
        python RCA_methods_metric.py -case 20240115 -model epsilon_diagnosis ## for metric data only
        cd ./log_only
        python RCA_methods_log.py  -case 20240115 -model epsilon_diagnosis ## for log data only
        cd ./multimodal
        python RCA_methods_combined.py  -case 20240115 -model epsilon_diagnosis ## for both metric and log data
    ```

    - Nezha:
    ```
        python main.py
    ```
    For Nezha, we provide the demo code for the case 20240124. Due to inconsistant filename for each case, you may need to change the name of the folder for each case accordingly. 

To run the baseline methods for SWAT and WADI datasets, the only difference is the data loader and the labels for evaluation. The labels are given in the corresponding scripts. For the Baro method:

    - Baro for SWAT & WADI: 
    ```
        python baro.py 
        python baro-evaluation.py
    ```

The RCD, epsilon_diagnosis, CIRCA methods are included in the pyrca package. Simply run:

    - Baro for SWAT & WADI: 
    ```
        python pyrca-main.py 
        python pyrca-evaluation.py
    ```

##### If you encounter the error regarding "name 'LIBSPOT' is not defined", please double-check if you are running the code in the directory of FastPC. 

#### If you fail to install pyrca package in windows, please use the following command:
#### "pip install sfr-pyrca --use-pep517 git+https://github.com/SchmollerLab/python-javabridge-windows"

