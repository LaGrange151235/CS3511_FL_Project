import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import numpy as np
import random
import os
import argparse
import datetime

import dataset_manager
import model

class client:
    def __init__(self, client_id, path, device):
        self.set_seed()
        self.client_id = client_id
        self.epoch = 0

        self.device = device
        self.model = model.Net()
        self.model = self.model.to(self.device)
        self.batch_size = 10
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.0001, momentum=0.5)

        self.train_datapath = path
        self.train_dataset = dataset_manager.get_train_dataset(self.train_datapath)
        self.train_dataloader = Data.DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_dataset = dataset_manager.get_test_dataset()
        self.test_dataloader = Data.DataLoader(dataset=self.test_dataset)
        
        self.log_file_path = "./Logs/local/"
        self.log_file = None
        if os.path.exists(self.log_file_path) == False:
            os.makedirs(self.log_file_path)
    
    def set_seed(self, Seed=100):    
        random.seed(Seed)    
        np.random.seed(Seed)    
        torch.manual_seed(Seed)    
        torch.cuda.manual_seed(Seed)    
        torch.cuda.manual_seed_all(Seed)

    def logging(self, string):
        print('['+str(datetime.datetime.now())+'] [Client '+str(self.client_id)+'] '+str(string))
        self.log_file = open("./Logs/local/client_"+str(self.client_id)+"_log.txt", "a")
        self.log_file.write('['+str(datetime.datetime.now())+'] [Client '+str(self.client_id)+'] '+str(string)+'\n')
        self.log_file.close()

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
            if batch_idx % 10 == 0:
                self.logging("Train Epoch: %d [%d/%d (%.0f%s)]\tLoss:%.6f" % (self.epoch, batch_idx*len(data), len(self.train_dataloader.dataset), 100. * batch_idx/len(self.train_dataloader), "%", loss.item()))
        #self.test()

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
