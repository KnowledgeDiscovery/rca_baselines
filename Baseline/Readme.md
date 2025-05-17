# Baselines 

This folder contains the baseline methods for Lemma-RCA evaluation with both single- and multi-modal settings.

    - FastPC: 
    ```
        python test_FastPC_pod_metric.py ## for metric data only
        python test_FastPC_pod_log.py  ## for log data only
        python test_FastPC_pod_combine.py  ## for both metric and log data
    ```

    - Nezha:
    ```
        python main.py
    ```
    For Nezha, we provide the demo code for the case 20240124. Due to inconsistant filename for each case, you may need to change the name of the folder for each case accordingly. 
    
##### If you encounter the error regarding "name 'LIBSPOT' is not defined", please double-check if you are running the code in the directory of FastPC. We observe such an error if the command is 'python FastPC/test_FastPC_pod_log.py' running in the directory of './rca_baselines/Baseline/offline/'.

