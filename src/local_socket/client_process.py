import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import numpy as np
import random
import argparse
import datetime
import socket

import dataset_manager
import model

import warnings


def set_seed(Seed=100):    
    random.seed(Seed)    
    np.random.seed(Seed)    
    torch.manual_seed(Seed)    
    torch.cuda.manual_seed(Seed)    
    torch.cuda.manual_seed_all(Seed)

class client:
    def __init__(self, client_id, path):
        self.client_id = client_id

        """define model"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
        self.model = model.Net().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.0001, momentum=0.5)
        self.batch_size = 10
        self.num_epochs = 3
        self.epoch = 0

        """define data"""
        self.train_datapath = path
        self.train_dataset = dataset_manager.get_train_dataset(self.train_datapath)
        self.test_dataset = dataset_manager.get_test_dataset()
        self.train_dataloader = Data.DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_dataloader = Data.DataLoader(dataset=self.test_dataset, batch_size=self.batch_size, shuffle=True)

        """define log"""
        self.log_file = open("./Logs/local_socket/client_"+str(self.client_id)+"_log.txt", "w")
        
    def logging(self, string):
        log_info = '['+str(datetime.datetime.now())+'] [Client '+str(self.client_id)+'] '+str(string)
        print(log_info)
        self.log_file.write(log_info)

    def train(self):
        self.model.train()
        for batch_idx, (data, target) in enumerate(self.train_dataloader):
            data = data.to(self.device)
            target = target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            if batch_idx % 100 == 0:
                self.logging("Train Epoch: %d [%d/%d (%.0f%s)]\tLoss:%.6f" % (self.epoch, batch_idx*len(data), len(self.train_dataloader.dataset), 100. * batch_idx/len(self.train_dataloader), "%", loss.item()))
                self.test()

    def test(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_dataloader:
                data = data.to(self.device)
                target = target.to(self.device)
                output = self.model(data)
                test_loss += self.criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(self.test_dataloader.dataset)
        self.logging("Test set: Average loss: %.4f, Accuracy: %d/%d (%.0f%s)" % (test_loss, correct, len(self.test_dataloader.dataset), 100. * correct / len(self.test_dataloader.dataset), "%"))

if __name__=="__main__":
    warnings.filterwarnings("ignore")

    """parameters"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--client_id', type=int, default=1)
    args = parser.parse_args()

    """define client process"""
    path = "./data/Client"+str(args.client_id)+".pkl"
    client_process = client(args.client_id, path)
    set_seed()

    """main process"""
    server_socket = socket.socket()
    server_socket.bind(("127.0.0.1", 8080+client_process.client_id))
    server_socket.listen()

    for epoch in range(0, client_process.num_epochs):
        client_process.epoch = epoch
        con, addr = server_socket.accept()
        client_process.logging("rec from %s" % (str(addr)))
        global_model = open("rec_global_model"+str(client_process.client_id)+".pt", "wb")
        line = con.recv(1024)
        while(len(line) == 0):
            line = con.recv(1024)
        client_process.logging("waited")
        while(line):
            global_model.write(line)
            line = con.recv(1024)
        global_model.close()
        con.close()

        """load global model"""
        rec_model = torch.load("rec_global_model"+str(client_process.client_id)+".pt")
        tensor_list = []
        for param in rec_model.parameters():
            tensor_list.append(param)
        for i, param in enumerate(client_process.model.parameters()):
            param.data = tensor_list[i].to(client_process.device)
        
        """start train"""
        client_process.train()
        torch.save(client_process.model, "send_client_model"+str(client_process.client_id)+".pt")

        """send client model"""
        client_model = open("send_client_model"+str(client_process.client_id)+".pt", "rb")
        client_socket = socket.socket()
        client_socket.connect(("127.0.0.1", 8080))
        client_process.logging("send to server")
        line = client_model.read(1024)
        while(line):
            client_socket.send(line)
            line = client_model.read(1024)
        client_model.close()
        client_socket.close()