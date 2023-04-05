import threading
import datetime
import time
import torch
import torch.utils.data as Data
import torch.nn as nn
import socket

import client_process
import dataset_manager
import model

class server:
    def __init__(self, client_number):
        self.client_number = client_number
        
        """define model"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
        self.global_model = model.Net().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.batch_size = 10
        self.num_epochs = 3
        
        """define data"""
        self.test_dataset = dataset_manager.get_test_dataset()
        self.test_dataloader = Data.DataLoader(dataset=self.test_dataset, batch_size=self.batch_size, shuffle=True)
        
        """define aggregation tools"""
        self.arrived = [0] * client_number
        self.arrived_tensor_list = [None] * client_number
        self.aggregated_tensor_list = []
        for param in self.global_model.parameters():
            self.aggregated_tensor_list.append(param.data)
        
        """define log"""
        self.log_file = open("./Logs/local_socket/server_log.txt", "w")
        
    def logging(self, string):
        print('['+str(datetime.datetime.now())+'] [Server] '+str(string))
        self.log_file.write('['+str(datetime.datetime.now())+'] [Server] '+str(string)+'\n')

    def send_global_model(self, i):
        """send global model"""
        torch.save(self.global_model, "send_global_model.pt")
        global_model = open("send_global_model.pt", "rb")
        client_socket = socket.socket()
        client_socket.connect(("127.0.0.1", 8081+i))
        line = global_model.read(1024)
        while(line):
            client_socket.send(line)
            line = global_model.read(1024)
        client_socket.close()
        global_model.close()

    def aggregation_process(self):
        server_socket = socket.socket()
        server_socket.bind(("127.0.0.1", 8080))
        server_socket.listen()

        for epoch in range(0, self.num_epochs):
            """send global model"""
            for i in range(self.client_number):
                #t = threading.Thread(target=self.send_global_model, args=(i, ))
                #t.start()
                self.send_global_model(i)

            """wait for client model"""
            for i in range(self.client_number):
                con, addr = server_socket.accept()
                client_model = open("rec_client_"+str(i+1)+"_model.pt", "wb")
                line = con.recv(1024)
                while(len(line) == 0):
                    line = con.recv(1024)
                self.logging("waited")
                while(line):
                    client_model.write(line)
                    line = con.recv(1024)
                client_model.close()
                con.close()

                rec_model = torch.load("rec_client_"+str(i+1)+"_model.pt")
                tensor_list = []
                for param in rec_model.parameters():
                    tensor_list.append(param)
                self.arrived_tensor_list[i] = tensor_list
                self.arrived[i] = 1


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
    server_process = server(20)
    server_process.aggregation_process()
    torch.save(server_process.global_model, "final_global_model_local_socket.pt")