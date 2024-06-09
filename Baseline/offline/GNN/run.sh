echo "Start running separate program!"
timer_start=`date "+%Y-%m-%d %H:%M:%S"`
python 0606/test_gnn_pod.py
wait
python 0517/test_gnn_pod.py
wait
python 0524/test_gnn_pod.py
wait
python 0901/test_gnn_pod.py
wait
python 1203/test_gnn_pod.py
wait
python baseline_evaluation.py
timer_end=`date "+%Y-%m-%d %H:%M:%S"`
echo $timer_start
echo $timer_end