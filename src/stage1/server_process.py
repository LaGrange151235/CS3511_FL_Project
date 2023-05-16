import torch
import torch.utils.data as Data
import torch.nn as nn

import argparse
import os
import time
import datetime
import random
import warnings
import numpy as np

import dataset_manager
import model

class server:
    def __init__(self, client_number, num_epoch, trial):
        self.set_seed()
        self.client_number = client_number
        self.num_epoch = num_epoch
        
        self.device = torch.device("cpu")
        self.global_model = model.Net()
        self.global_model = self.global_model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        
        self.test_dataset = dataset_manager.get_test_dataset()
        self.test_dataloader = Data.DataLoader(dataset=self.test_dataset)
        
        self.arrived_tensor_list = [None] * client_number
        self.aggregated_tensor_list = []
        
        self.log_file_path = "./Logs/stage1/"+str(trial)+"/"
        self.log_file = None

        self.file_path = "./model/stage1/"+str(trial)+"/"
    
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

    def aggregation_process(self, epoch):
        for client_id in range(1, self.client_number+1):
            client_model_path = self.file_path+"client_"+str(client_id)+"_model_epoch_"+str(epoch)+".pth"
            model = torch.load(client_model_path)
            tensor_list = []
            for param in model.parameters():
                tensor_list.append(param.data)
            self.arrived_tensor_list[client_id-1] = tensor_list

        aggregated_tensor_list = []
        for tensor_id, tensor_content in enumerate(self.arrived_tensor_list[0]):
            sum_tensor = torch.zeros(tensor_content.size()).to(self.device)
            for rank in range(self.client_number):
                sum_tensor += self.arrived_tensor_list[rank][tensor_id]
            aggregated_tensor = sum_tensor / self.client_number
            aggregated_tensor_list.append(aggregated_tensor)
            
        '''save global model'''
        self.aggregated_tensor_list = aggregated_tensor_list
        for i, param in enumerate(self.global_model.parameters()):
            param.data = self.aggregated_tensor_list[i]

        self.logging("Tensors aggregated")
        self.test()

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
    for epoch in range(1,args.num_epoch+1):
        torch.save(server_process.global_model,server_process.file_path+"global_model_epoch_"+str(epoch)+".pth")
        while len(os.listdir(server_process.file_path)) < args.client_num + 1:
            time.sleep(0.01)
        time.sleep(1)
        server_process.aggregation_process(epoch)
        for file in os.listdir(server_process.file_path):
            os.system("rm "+server_process.file_path+str(file))
    torch.save(server_process.global_model, server_process.file_path+"final_global_model.pth")