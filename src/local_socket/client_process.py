import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import numpy as np
import random
import argparse
import datetime


import dataset_manager
import socket_manager
import model

def set_seed(Seed=100):    
    random.seed(Seed)    
    np.random.seed(Seed)    
    torch.manual_seed(Seed)    
    torch.cuda.manual_seed(Seed)    
    torch.cuda.manual_seed_all(Seed)

class client:
    def __init__(self, client_id, path, device):
        self.device = device
        self.client_id = client_id
        self.num_epochs = 10
        self.model = model.Net()
        self.model = self.model.to(self.device)
        self.batch_size = 10
        self.epoch = 0
        self.train_datapath = path
        self.train_dataset = dataset_manager.get_train_dataset(self.train_datapath)
        self.test_dataset = dataset_manager.get_test_dataset()
        self.train_dataloader = Data.DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_dataloader = Data.DataLoader(dataset=self.test_dataset, batch_size=self.batch_size, shuffle=True)
        self.log_file = open("./Logs/local_socket/client_"+str(self.client_id)+"_log.txt", "w")
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.5)
    
    def logging(self, string):
        print('['+str(datetime.datetime.now())+'] [Client '+str(self.client_id)+'] '+str(string))
        self.log_file.write('['+str(datetime.datetime.now())+'] [Client '+str(self.client_id)+'] '+str(string)+'\n')

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--client_id', type=int, default=1)
    args = parser.parse_args()
    path = "./data/Client"+str(args.client_id)+".pkl"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    client_process = client(args.client_id, path, device)
    set_seed()
    server_socket = socket_manager.server_socket(device=client_process.device, port=8080+args.client_id)
    client_socket = socket_manager.client_socket(port=8120+args.client_id)

    tensor_list = server_socket.rec_tensor_list()
    for i, param in enumerate(client_process.model.parameters()):
        param.data = tensor_list[i].to(client_process.device)

    for epoch in range(1,5):
        client_process.epoch = epoch
        client_process.train()
        tensor_list = []
        for param in client_process.model.parameters():
            tensor_list.append(param.data)
        client_socket.send_tensor_list(tensor_list)
        tensor_list = server_socket.rec_tensor_list()
        for i, param in enumerate(client_process.model.parameters()):
            param.data = tensor_list[i].to(client_process.device)
        