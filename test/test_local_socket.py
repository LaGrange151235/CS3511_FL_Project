import os
import threading

for i in range(1, 3):
    t = threading.Thread(target=os.system, args=("python ./src/local_socket/client_process.py --client_id "+ str(i), ))
    t.start()

t = threading.Thread(target=os.system, args=("python ./src/local_socket/server_process.py", ))
t.start()