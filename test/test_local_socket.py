import os
import threading
import time

for i in range(20):
    t = threading.Thread(target=os.system, args=("python3 ./src/local_socket/client_process.py --client_id "+ str(i+1), ))
    t.start()

time.sleep(10)

t = threading.Thread(target=os.system, args=("python3 ./src/local_socket/server_process.py", ))
t.start()