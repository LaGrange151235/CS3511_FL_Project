import argparse
import threading
import datetime
import time
import os
import random
import torch
import torch.utils.data as Data
import torch.nn as nn

import client_process
import dataset_manager
import model

class server:
    def __init__(self, client_number, num_epoch, ratio):
        self.portion_number = int(ratio*client_number)
        self.client_number = client_number
        self.num_epoch = num_epoch
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.global_model = model.Net()
        self.global_model = self.global_model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        
        self.test_dataset = dataset_manager.get_test_dataset()
        self.test_dataloader = Data.DataLoader(dataset=self.test_dataset)
        
        self.arrived = [0] * client_number
        self.arrived_tensor_list = [None] * client_number
        self.aggregated_tensor_list = []
        
        self.log_file_path = "./Logs/local_portion/"
        self.log_file = None
        if os.path.exists(self.log_file_path) == False:
            os.makedirs(self.log_file_path)
        for param in self.global_model.parameters():
            self.aggregated_tensor_list.append(param.data)
    
    def logging(self, string):
        print('['+str(datetime.datetime.now())+'] [Server] '+str(string))
        self.log_file = open("./Logs/local_portion/server_log.txt", "a")
        self.log_file.write('['+str(datetime.datetime.now())+'] [Server] '+str(string)+'\n')
        self.log_file.close()

    def aggregation_process(self):
        for epoch in range(1,self.num_epoch+1):
            '''BSP aggregation scheme'''
            while sum(self.arrived) != self.client_number:
                time.sleep(0.01)
            aggregated_tensor_list = []
            client_portion_id_list = []
            while len(client_portion_id_list) < self.portion_number:
                random_id = int(random.randint(0, self.client_number) % self.client_number)
                if random_id not in client_portion_id_list:
                    client_portion_id_list.append(random_id)
            self.logging("aggregation with client %s" % (str(client_portion_id_list)))

            for tensor_id, tensor_content in enumerate(self.arrived_tensor_list[0]):
                sum_tensor = torch.zeros(tensor_content.size()).to(self.device)
                for rank in client_portion_id_list:
                    sum_tensor += self.arrived_tensor_list[rank][tensor_id]
                aggregated_tensor = sum_tensor / self.client_number
                aggregated_tensor_list.append(aggregated_tensor)
            
            '''save global model'''
            self.aggregated_tensor_list = aggregated_tensor_list
            for i, param in enumerate(self.global_model.parameters()):
                param.data = self.aggregated_tensor_list[i]

            self.logging("Tensors aggregated")
            self.test()
    
            '''reset'''    
            self.arrived = [0] * self.client_number

    def client_process_(self, client_id):
        '''init client process'''
        path = "./data/Client"+str(client_id)+".pkl"
        client = client_process.client(client_id, path, self.device)
        client_process.set_seed()

        '''load global model'''
        tensor_list = []
        for param in client.model.parameters():
            tensor_list.append(param.data)
        self.arrived_tensor_list[client_id-1] = tensor_list

        '''start training'''
        for epoch in range(1,self.num_epoch+1):
            for i, param in enumerate(client.model.parameters()):
                param.data = self.aggregated_tensor_list[i].to(self.device)

            client.epoch = epoch
            client.train()
            self.arrived[client_id-1] = 1
            
            tensor_list = []
            for param in client.model.parameters():
                tensor_list.append(param.data)
            self.arrived_tensor_list[client_id-1] = tensor_list

            '''waiting for aggregation'''
            while self.arrived[client_id-1]:
                time.sleep(0.01)

    def server_process_(self):
        for i in range(1, self.client_number+1):
            t = threading.Thread(target=self.client_process_, args=(i, ))
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--client_num', type=int, default=20)
    parser.add_argument('--num_epoch', type=int, default=5)
    parser.add_argument('--m', type=float, default=0.8)
    args = parser.parse_args()
    server_process = server(args.client_num, args.num_epoch, args.m)
    server_process.server_process_()
    torch.save(server_process.global_model, "final_global_model_local.pt")