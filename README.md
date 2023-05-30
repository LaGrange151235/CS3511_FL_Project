[中文版本](README_zh.md)

This repository is for the solo project of CS3511 2023 Spring. I have implemented 3 different FL systems for 3 Stages requirement.
1. $N(N=20)$ client processes with 1 server process, all the $N$ clients participate the aggregation of each round, communicate with a shared folder
2. $N(N=20)$ client processes with 1 server process, $M$ out of $N$ clients participate the aggregation of each round, communicate with a shared folder
3. $N(N=20)$ client processes with 1 server process, all the $N$ clients participate the aggregation of each round, communicate with sockets

## Install
```bash
pip install -r requirements.txt
```

## Run
Before start training, you have to put training datasets at folder `data`, you can refer to the file tree record at [tree.txt](tree.txt) to know how I arrange files. Besides, you should better ensure you have more than 5GB free memory to avoid failure.

For each Stage, I provide a script `train.sh` for experiment.
- Stage 1
    Run the `train.sh` to start training:
    ```bash
    bash ./src/stage1/train.sh
    ```
    Then input the client server number and epoch number as below:
    ```bash
    Client server number: 20
    Epoch number: 50
    ```
    If training processes start successfully, you will see as below:
    ![](example.gif)
- Stage 2
    Run the `train.sh` to start training:
    ```bash
    bash ./src/stage2/train.sh
    ```
    Then input the client server number and epoch number as below:
    ```bash
    Client server number: 20
    Epoch number: 50
    m: 0.8
    ```
    Here the `m` is selection ratio ($m\in(0,1]$), so if you want to choose $M$ out of $N$ clients for each round to participate aggregation, please set $m=\frac{M}{N}$.
- Stage 3
    Run the `train.sh` to start training:
    ```bash
    bash ./src/stage3/train.sh
    ```
    Then input the client server number and epoch number as below:
    ```bash
    Client server number: 20
    Epoch number: 50
    ```
When experiment finish, you can find log files at dir `Logs` and `final_global_model.pth` at dir `model`. You can also download `Logs` and `model` of my experiments from jbox([here](https://jbox.sjtu.edu.cn/l/D1m5hr)).