[English Version](README.md)

本仓库是2023年春季学期CS3511课程单人作业，我根据3阶段不同要求实现了3个联邦学习系统
1. $N(N=20)$个客户端进程和1个服务器进程，所有的$N$个客户端进程都参与每轮同步聚合，通过共享文件路径通信
2. $N(N=20)$个客户端进程和1个服务器进程，$N$个中选$M$个客户端进程参与每轮同步聚合，通过共享文件路径通信
3. $N(N=20)$个客户端进程和1个服务器进程，所有的$N$个客户端进程都参与每轮同步聚合，通过socket通信

## 安装
```bash
pip install -r requirements.txt
```

## 运行
开始训练前，需要将训练数据集放在路径`data`下, 你可以参考文件树记录[tree.txt](tree.txt)来了解我如何安排文件. 此外，你应当保证有多于5GB空闲内存来避免崩溃。

对于每个阶段，我提供一个`train.sh`脚本启动实验。
- 阶段1
    运行`train.sh`启动训练：
    ```bash
    bash ./src/stage1/train.sh
    ```
    输入客户端个数和训练轮数：
    ```bash
    Client server number: 20
    Epoch number: 50
    ```
    如果训练过程成功启动，你会看到如下结果：
    ![](example.gif)
- Stage 2
    运行`train.sh`启动训练：
    ```bash
    bash ./src/stage2/train.sh
    ```
    输入客户端个数和训练轮数：
    ```bash
    Client server number: 20
    Epoch number: 50
    m: 0.8
    ```
    这里的 `m`是选择率($m\in(0,1]$),如果你想从$N$个客户端进程中每轮选择$M$个客户端参与更新聚合，请设置$m=\frac{M}{N}$.
- Stage 3
    运行`train.sh`启动训练：
    ```bash
    bash ./src/stage3/train.sh
    ```
    输入客户端个数和训练轮数：
    ```bash
    Client server number: 20
    Epoch number: 50
    ```
## 文件
对于每个阶段，我们的代码文件都在`src/stage*`。在每个阶段，我们都有`client_process.py`，`dataset_manager.py`，`model.py`和`server_process.py`。
- `client_process.py`: 客户端进程代码
- `dataset_manager.py`: 读取所需数据集代码
- `model.py`: 训练所需CNN模型代码
- `server_process.py`: 服务器进程代码

实验结束后，你可以在路径 `Logs`下找到日志文件并在路径`model`找到最终的全局模型。你也可以从交大云盘（[这里](https://jbox.sjtu.edu.cn/l/D1m5hr)）下载我的实验结果 `Logs`和`model`文件。