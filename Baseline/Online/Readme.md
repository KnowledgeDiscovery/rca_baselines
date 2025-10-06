# Baselines 

This folder contains the baseline methods for Lemma-RCA datasets evaluation with both single- and multi-modal settings in the online setting.

Be sure to update the data_path of each case defined in each configuration file under config folder.

    - Baro: 
    ```
        --method baro --modality metric-only --dataset 20240115 ## for case 20240115 metric data only
        --method baro --modality log-only --dataset 20240115 ## for case 20240115 log data only
        --method baro --modality multimodality --dataset 20240115 ## case 20240115 with for both metric and log data
    ```

    - RCD: 
    ```
        --method rcd --modality metric-only --dataset 20240115 ## for case 20240115 metric data only
        --method rcd --modality log-only --dataset 20240115 ## for case 20240115 log data only
        --method rcd --modality multimodality --dataset 20240115 ## case 20240115 with for both metric and log data
    ```

    - CIRCA: 
    ```
        --method circa --modality metric-only --dataset 20240115 ## for case 20240115 metric data only
        --method circa --modality log-only --dataset 20240115 ## for case 20240115 log data only
        --method circa --modality multimodality --dataset 20240115 ## case 20240115 with for both metric and log data
    ```

    - epsilon_diagnosis: 
    ```
        --method epsilon_diagnosis --modality metric-only --dataset 20240115 ## for case 20240115 metric data only
        --method epsilon_diagnosis --modality log-only --dataset 20240115 ## for case 20240115 log data only
        --method epsilon_diagnosis --modality multimodality --dataset 20240115 ## case 20240115 with for both metric and log data
    ```

#### If you fail to install pyrca package in windows, please use the following command:
#### "pip install sfr-pyrca --use-pep517 git+https://github.com/SchmollerLab/python-javabridge-windows"

