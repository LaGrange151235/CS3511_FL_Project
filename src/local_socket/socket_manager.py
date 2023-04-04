import socket
import torch
import json
import copy

class server_socket():
    def __init__(self, device, port=8080, ip="127.0.0.1"):
        self.device = device
        self.s = socket.socket()
        self.ip = ip
        self.host = socket.gethostbyaddr(self.ip)[0]
        self.port = port
        print("bind: %s, %d" % (self.host, self.port))
        

    def rec_tensor_list(self):
        self.s.bind((self.host, self.port))
        self.s.listen(20)
        c, c_addr = self.s.accept()
        tensor_list = c.recv(102400000)
        tensor_list = json.loads(tensor_list)
        for tensor_id, tensor_content in enumerate(tensor_list):
            tensor_list[tensor_id] = torch.tensor(tensor_list[tensor_id]).to(self.device)
        self.s.close()
        self.s = socket.socket()
        return tensor_list
    
class client_socket():
    def __init__(self, port=8080, ip="127.0.0.1"):
        self.s = socket.socket()
        self.ip = ip
        self.host = socket.gethostbyaddr(self.ip)[0]
        self.port = port

    def send_tensor_list(self, sent_tensor_list):
        tensor_list = copy.deepcopy(sent_tensor_list)
        print("conncet: %s, %d" % (self.host, self.port))
        self.s.connect((self.host, self.port))
        device = torch.device("cpu")
        for tensor_id, tensor_content in enumerate(tensor_list):
            tensor_list[tensor_id] = tensor_list[tensor_id].to(device).tolist()
        self.s.send(json.dumps(tensor_list).encode())
        self.s.close()
        self.s = socket.socket()