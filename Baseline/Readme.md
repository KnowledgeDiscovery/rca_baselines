# Baselines 

This folder contains the baseline methods for Lemma-RCA evaluation with both single- and multi-modal settings.

    - Baro: 
    ```
        cd ./metric_only
        python baro_main_metric.py ## for metric data only
        cd ./log_only
        python baro_main_log.py  ## for log data only
        cd ./multimodal
        python baro_main_combined.py  ## for both metric and log data
    ```
    

    - Nezha:
    ```
        python main.py
    ```
    For Nezha, we provide the demo code for the case 20240124. Due to inconsistant filename for each case, you may need to change the name of the folder for each case accordingly. 
    
##### If you encounter the error regarding "name 'LIBSPOT' is not defined", please double-check if you are running the code in the directory of FastPC. 

