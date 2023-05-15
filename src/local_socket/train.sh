for ((i=1; i<=2; i++))
do
    taskset -c $i python3 ./src/local_socket/client_process.py --client_id $i & 
done
sleep 10
taskset -c 0 python3 ./src/local_socket/server_process.py --client_num 2 --num_epoch 2