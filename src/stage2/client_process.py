import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

import argparse
import os
import time
import datetime
import random
import numpy as np

import dataset_manager
import model

class client:
    def __init__(self, client_id, path, trial):
        self.set_seed()
        self.client_id = client_id
        self.epoch = 0

        self.device = torch.device("cpu")
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
        
        self.log_file_path = "./Logs/stage2/"+str(trial)+"/"
        self.log_file = None

        self.file_path = "./model/stage2/"+str(trial)+"/"

    def set_seed(self, seed=100):    
        random.seed(seed)    
        np.random.seed(seed)    
        torch.manual_seed(seed)    
        torch.cuda.manual_seed(seed)    
        torch.cuda.manual_seed_all(seed)

    def logging(self, string):
        print('['+str(datetime.datetime.now())+'] [Client '+str(self.client_id)+'] '+str(string))
        self.log_file = open(self.log_file_path+"client_"+str(self.client_id)+"_log.txt", "a")
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

if __name__=="__main__":
    """parameters"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--client_id', type=int, default=1)
    parser.add_argument('--num_epoch', type=int, default=5)
    parser.add_argument('--trial', type=int, default=0)
    args = parser.parse_args()

    path = "./data/Client"+str(args.client_id)+".pkl"
    client_process = client(args.client_id, path, args.trial)
    for epoch in range(1,args.num_epoch+1):
        client_process.epoch = epoch
        global_model_path = client_process.file_path+"global_model_epoch_"+str(epoch)+".pth"
        
        while os.path.exists(global_model_path) == False:
            time.sleep(0.01)
        time.sleep(1)

        global_model = torch.load(global_model_path)
        tensor_list = []
        for param in global_model.parameters():
            tensor_list.append(param)
        for i, param in enumerate(client_process.model.parameters()):
            param.data = tensor_list[i].to(client_process.device)
        
        client_process.train()
        
        client_model_path = client_process.file_path+"client_"+str(client_process.client_id)+"_model_epoch_"+str(epoch)+".pth"
        torch.save(client_process.model, client_model_path)
        
        while os.path.exists(client_model_path):
            time.sleep(0.01)
