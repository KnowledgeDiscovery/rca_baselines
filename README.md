# [LEMMA-RCA](https://lemma-rca.github.io/)
Root cause analysis (RCA) is a task of identifying the underlying causes of system faults/failures by analyzing the system monitoring data. LEMMA-RCA is a collection of multi-modal datasets with various real system faults to facilitate future research in RCA. LEMMA-RCA is multi-domain, encompassing real-world applications such as microservice and water treatment/distribution systems. The datasets are released under the CC BY-NC 4.0 license and hosted on Huggingface, the codes are available on Github.

<p align="center">
      <img src="./Other/rca_update.png" alt="drawing" style="width:600px;"/> 
</p>

### Dataset Download
We release the dataset in [Huggingface](https://huggingface.co/Lemma-RCA-NEC) and the detailed data statistics can be found in [Lemma-RCA Webpage](https://lemma-rca.github.io/docs/data.html).


### Multiple Domains
LEMMA-RCA covers two domains: 
- IT Operations (Product Review and Cloud Computing)
- OT Operations (Water Treatment/Distribution)
Each domain contains two datasets.

### Real System Faults
Each dataset contains various system faults simulated from real-world scenarios. 
For details, please check our [website](https://lemma-rca.github.io/).

### Unified Evaluation
LEMMA-RCA datasets are evaluated with eight causal learning baselines in four settings: online/offline with single/multiple modality data.

### File directory
```
Root:
      --|IT
            --|data preprocessing
                  --|json2message.py
                  --|drain3.py
                  --|drain3_parse.py
                  --|README.md
                  --|drain3.yaml
                  --|log_frequency_extraction.py
                  --|log_golden_frequency.py

      --|OT
            --|data preprocessing
                  --|SWaT
                        --|data_segment.py
                        --|node_data_cut.py
                        --|node_final_precess.py
                        --|pod_data_cut.py
                        --|pod_final_process.py
                        --|process.sh
                  --|WADI
                        --|data_segment.py
                        --|node_data_cut.py
                        --|node_final_precess.py
                        --|pod_data_cut.py
                        --|pod_final_process.py
                        --|process.sh
      --|Baseline
            --|offline
                  --|Dynotears
                  --|FastPC
                  --|GNN
                  --|GOLEM
                  --|LSTM
            --|online
                  --| baseline_final
```
### Citation
If you use LEMMA-RCA in your work, please cite our paper (Coming soon!).

