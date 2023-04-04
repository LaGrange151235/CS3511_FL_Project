import threading
import datetime
import time
import torch
import torch.utils.data as Data
import torch.nn as nn

import client_process
import dataset_manager
import socket_manager
import model

class server:
    def __init__(self, client_number):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.client_number = client_number
        self.global_model = model.Net()
        self.global_model = self.global_model.to(self.device)
        self.batch_size = 10
        self.criterion = nn.CrossEntropyLoss()
        self.test_dataset = dataset_manager.get_test_dataset()
        self.test_dataloader = Data.DataLoader(dataset=self.test_dataset, batch_size=self.batch_size, shuffle=True)
        self.arrived = [0] * client_number
        self.arrived_tensor_list = [None] * client_number
        self.aggregated_tensor_list = []
        self.server_sockets = []
        self.client_sockets = []
        self.log_file = open("./Logs/local_socket/server_log.txt", "w")
        for i in range(client_number):
            self.server_sockets.append(socket_manager.server_socket(device=self.device, port=(8121+i)))
            self.client_sockets.append(socket_manager.client_socket(port=(8081+i)))
        for param in self.global_model.parameters():
            self.aggregated_tensor_list.append(param.data)
    
    def logging(self, string):
        print('['+str(datetime.datetime.now())+'] [Server] '+str(string))
        self.log_file.write('['+str(datetime.datetime.now())+'] [Server] '+str(string)+'\n')

    def start_client_socket_listening_process(self, client_id):
        self.client_sockets[client_id-1].send_tensor_list(self.aggregated_tensor_list)

        for epoch in range(1,5):
            tensor_list = self.server_sockets[client_id-1].rec_tensor_list()
            self.logging("Recieve tensor_list")
            self.arrived[client_id-1] = 1
            self.arrived_tensor_list[client_id-1] = tensor_list
            while self.arrived[client_id-1] == 1:
                time.sleep(0.01)
            self.client_sockets[client_id-1].send_tensor_list(self.aggregated_tensor_list)
            self.logging("Send tensor_list")
        
    def aggregation_process(self):
        for epoch in range(1,5):
            while sum(self.arrived) != self.client_number:                
                time.sleep(0.01)
            aggregated_tensor_list = []
            for tensor_id, tensor_content in enumerate(self.arrived_tensor_list[0]):
                sum_tensor = torch.zeros(tensor_content.size()).to(self.device)
                for rank in range(self.client_number):
                    sum_tensor += self.arrived_tensor_list[rank][tensor_id]
                aggregated_tensor = sum_tensor / self.client_number
                aggregated_tensor_list.append(aggregated_tensor)
            
            self.aggregated_tensor_list = aggregated_tensor_list
            for i, param in enumerate(self.global_model.parameters()):
                param.data = self.aggregated_tensor_list[i]

            self.logging("Tensors aggregated")
            self.test()
    
            self.arrived = [0] * self.client_number

    def start_server_process(self):
        for i in range(1, self.client_number+1):
            t = threading.Thread(target=self.start_client_socket_listening_process, args=(i, ))
            t.start()
        t = threading.Thread(target=self.aggregation_process, args=())
        t.start()
    
    def test(self):
        self.global_model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_dataloader:
                data = data.to(self.device)
                target = target.to(self.device)
                output = self.global_model(data)
                test_loss += self.criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(self.test_dataloader.dataset)
        self.logging("Test set: Average loss: %.4f, Accuracy: %d/%d (%.0f%s)" % (test_loss, correct, len(self.test_dataloader.dataset), 100. * correct / len(self.test_dataloader.dataset), "%"))

if __name__=="__main__":
    server_process = server(2)
    server_process.start_server_process()
    server_process.test()