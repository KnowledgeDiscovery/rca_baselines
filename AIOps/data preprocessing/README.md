# Drain preprocessing log data

## Execution steps

### Step 1: Preprocess data from original elasticsearch log (json format) to log messages
```terminal command
python3 json2message.py
```

***Notice***: Some of the arguments may need to change
```
    --path, the input directory of the json format log data
    --output_dir, the output directory of all log messages
    --output_dir2, the output directory of pod-level log messages for each pod
    --output_dir3, the output directory of node-level log messages for each node
```

### Step 2: Usa Drain to parse both node-level and pod-level log messages

```terminal command
python3 drain3_parse.py ./output/log_prep_node/  -o "./drain3_result/node"

python3 drain3_parse.py ./output/log_prep_pod/   -o "./drain3_result/pod"

```

```
    --input_dir, default="./output/log_prep_node/" or "./output/log_prep_pod/"
    --output_dir, default="./drain3_result/node"   or "./drain3_result/pod"
  
```
