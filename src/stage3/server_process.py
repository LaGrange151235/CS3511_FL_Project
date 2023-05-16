import torch
import torch.utils.data as Data
import torch.nn as nn

import argparse
import os
import time
import datetime
import random
import socket
import warnings
import numpy as np

import dataset_manager
import model

class server:
    def __init__(self, client_number, num_epoch, trial):
        self.set_seed()
        self.client_number = client_number
        
        """define model"""
        self.device = torch.device("cpu")
        self.global_model = model.Net().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.batch_size = 10
        self.num_epochs = num_epoch
        
        """define data"""
        self.test_dataset = dataset_manager.get_test_dataset()
        self.test_dataloader = Data.DataLoader(dataset=self.test_dataset)
        
        """define aggregation tools"""
        self.arrived = [0] * client_number
        self.arrived_tensor_list = [None] * client_number
        self.aggregated_tensor_list = []
        for param in self.global_model.parameters():
            self.aggregated_tensor_list.append(param.data)
        
        """define log"""
        self.log_file_path = "./Logs/stage3/"+str(trial)+"/"
        self.log_file = None

        """define file path"""
        self.file_path = "./model/stage3/"+str(trial)+"/server/"
        if os.path.exists(self.file_path) == False:
            os.makedirs(self.file_path)
        
    def set_seed(self, Seed=100):    
        random.seed(Seed)    
        np.random.seed(Seed)    
        torch.manual_seed(Seed)    
        torch.cuda.manual_seed(Seed)    
        torch.cuda.manual_seed_all(Seed)

    def logging(self, string):
        print('['+str(datetime.datetime.now())+'] [Server] '+str(string))
        self.log_file = open(self.log_file_path+"server_log.txt", "a")
        self.log_file.write('['+str(datetime.datetime.now())+'] [Server] '+str(string)+'\n')
        self.log_file.close()

    def send_global_model(self, i, epoch):
        """send global model"""
        torch.save(self.global_model, self.file_path+"send_global_model_epoch"+str(epoch)+".pth")
        global_model = open(self.file_path+"send_global_model_epoch"+str(epoch)+".pth", "rb")
        client_socket = socket.socket()
        self.logging("send global model to client %d" % (i+1))
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

        for epoch in range(1, self.num_epochs+1):
            """send global model"""
            for i in range(self.client_number):
                self.send_global_model(i, epoch)

            """wait for client model"""
            for i in range(self.client_number):
                '''receive client update'''
                con, addr = server_socket.accept()
                client_model = open(self.file_path+"rec_client_"+str(i+1)+"_model_epoch"+str(epoch)+".pth", "wb")
                line = con.recv(1024)
                while(len(line) == 0):
                    line = con.recv(1024)
                self.logging("waiting for client %d" % (i+1))
                while(line):
                    client_model.write(line)
                    line = con.recv(1024)
                client_model.close()
                con.close()
                
                '''load client model'''
                rec_model = torch.load(self.file_path+"rec_client_"+str(i+1)+"_model_epoch"+str(epoch)+".pth")
                tensor_list = []
                for param in rec_model.parameters():
                    tensor_list.append(param)
                self.arrived_tensor_list[i] = tensor_list
                self.arrived[i] = 1

            '''BSP aggregation scheme'''
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
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument('--client_num', type=int, default=20)
    parser.add_argument('--num_epoch', type=int, default=5)
    parser.add_argument('--trial', type=int, default=0)
    args = parser.parse_args()
    server_process = server(args.client_num, args.num_epoch, args.trial)
    server_process.logging(str(args))
    server_process.aggregation_process()
    torch.save(server_process.global_model, server_process.file_path+"final_global_model.pth")