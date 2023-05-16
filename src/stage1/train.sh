home='/home/ubuntu/FL_Project'
read -p "Client server number: " n_clients
read -p "Epoch number: " n_epoch

trial_no=$(ls $home/Logs/stage1 | wc -l)
log_dir=$home/Logs/stage1/${trial_no}
model_dir=$home/model/stage1/${trial_no}
mkdir -p $log_dir
mkdir -p $model_dir

echo "start server"
taskset -c 0 python3 ./src/stage1/server_process.py --client_num $n_clients --num_epoch $n_epoch --trial $trial_no & 
sleep 1
for ((i=1; i<=$n_clients; i++))
do
    echo "start client "$i
    taskset -c $i python3 ./src/stage1/client_process.py --client_id $i --num_epoch $n_epoch --trial $trial_no & 
done