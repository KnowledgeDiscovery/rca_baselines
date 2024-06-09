echo "Start running separate program!"
timer_start=`date "+%Y-%m-%d %H:%M:%S"`
python test_gnn_node.py
wait
python test_gnn_pod.py
wait
python interdependent.py
timer_end=`date "+%Y-%m-%d %H:%M:%S"`
echo $timer_start
echo $timer_end