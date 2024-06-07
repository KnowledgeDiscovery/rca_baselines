#!/bin/bash

# Ensure the script stops if any command fails
set -e

# Execute the Python scripts in order
python data_segment.py
python node_data_cut.py
python pod_data_cut.py
python node_final_process.py
python pod_final_process.py

echo "All scripts executed successfully!"
