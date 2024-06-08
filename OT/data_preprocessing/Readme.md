# OT: Data Preprocessing for Root Cause Localization

This folder contains the data preprocessing code for the SWaT and WADI datasets. These datasets were initially used for anomaly detection, but we have adapted them for root cause localization. Follow the steps below to download the original data and use our code to process it into the corresponding RCA datasets.

1. **Download Original Data**: 
    - You can download the original datasets from the [iTrust website](https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/).
    - For SWaT, use the data from December 2015.
    - For WADI, use the WADI.A1 data.

2. **Processing Strategy**:
    - Although the two datasets require slightly different processing steps, we use a consistent high-level strategy for both.

## Files and their Functions

1. **data_segment.py**
    - Segments the monitoring time series data of different sensors/actuators based on the time of the attack event. Each segment contains one attack event and is 2 hours long.

2. **node_data_cut.py**
    - Generates the monitoring data of each node (high-level stage), including timestamp, system metrics, and the corresponding label.

3. **pod_data_cut.py**
    - Generates the monitoring data of each pod (low-level sensor), including timestamp, system metrics, and the corresponding label.

4. **node_final_process.py**
    - Reorganizes each data segment into a defined data structure. For each segment, it divides the data according to the monitoring metric. Each metric includes:
        - **Sequence:** Monitoring metric of different nodes.
        - **Time:** Corresponding timestamp.
        - **Node_Name:** Name of each node.
        - **KPI_Feature:** Corresponding system KPI.

5. **pod_final_process.py**
    - Reorganizes each data segment into a defined data structure. For each segment, it divides the data according to the monitoring metric. Each metric includes:
        - **Sequence:** Monitoring metric of different pods.
        - **Time:** Corresponding timestamp.
        - **Node_Name:** Name of each pod.
        - **KPI_Feature:** Corresponding system KPI.

6. **process.sh**
    - Unifies the previous five steps into one shell script. You can run all processing steps by executing `./process.sh`.

## User Guidance

1. Download the SWaT and WADI datasets based on the provided version (i.e., SWaT Dec 2015 and WADI.A1).
2. Run the processing steps for SWaT by executing `./SWaT/process.sh`.
3. Run the processing steps for WADI by executing `./WADI/process.sh`.

By following these steps, you will be able to preprocess the SWaT and WADI datasets for root cause localization effectively.