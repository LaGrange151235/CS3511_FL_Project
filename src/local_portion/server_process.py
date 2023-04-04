import threading
import datetime
import time
import random
import torch
import torch.utils.data as Data
import torch.nn as nn

import client_process
import dataset_manager
import model

class server:
    def __init__(self, client_number, portion_number):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.client_number = client_number
        self.portion_number = portion_number
        self.global_model = model.Net()
        self.global_model = self.global_model.to(self.device)
        self.batch_size = 10
        self.criterion = nn.CrossEntropyLoss()
        self.test_dataset = dataset_manager.get_test_dataset()
        self.test_dataloader = Data.DataLoader(dataset=self.test_dataset, batch_size=self.batch_size, shuffle=True)
        self.arrived = [0] * client_number
        self.arrived_tensor_list = [None] * client_number
        self.aggregated_tensor_list = []
        for param in self.global_model.parameters():
            self.aggregated_tensor_list.append(param.data)
    
    def logging(self, string):
        print('['+str(datetime.datetime.now())+'] [Server] '+str(string))


    def start_client_process(self, client_id):
        path = "./data/Client"+str(client_id)+".pkl"
        client = client_process.client(client_id, path, self.device)
        client_process.set_seed()
        tensor_list = []
        for param in client.model.parameters():
            tensor_list.append(param.data)
        self.arrived_tensor_list[client_id-1] = tensor_list

        for epoch in range(1,5):
            for i, param in enumerate(client.model.parameters()):
                param.data = self.aggregated_tensor_list[i].to(self.device)

            client.epoch = epoch
            client.train()
            client.test()
            self.arrived[client_id-1] = 1
            
            tensor_list = []
            for param in client.model.parameters():
                tensor_list.append(param.data)
            self.arrived_tensor_list[client_id-1] = tensor_list
            while self.arrived[client_id-1]:
                time.sleep(0.01)

    def aggregation_process(self):
        for epoch in range(1,5):
            while sum(self.arrived) != self.client_number:
                time.sleep(0.01)
            aggregated_tensor_list = []
            client_portion_id_list = []
            while len(client_portion_id_list) < self.portion_number:
                random_id = int(random.randint(0, self.client_number) % self.client_number)
                if random_id not in client_portion_id_list:
                    client_portion_id_list.append(random_id)

            for tensor_id, tensor_content in enumerate(self.arrived_tensor_list[0]):
                sum_tensor = torch.zeros(tensor_content.size()).to(self.device)
                for rank in client_portion_id_list:
                    sum_tensor += self.arrived_tensor_list[rank][tensor_id]
                aggregated_tensor = sum_tensor / self.client_number
                aggregated_tensor_list.append(aggregated_tensor)
            
            self.aggregated_tensor_list = aggregated_tensor_list
            for i, param in enumerate(self.global_model.parameters()):
                param.data = self.aggregated_tensor_list[i]

            self.logging("Tensors aggregated")
            self.logging("Client portion idx list: %s" % (str(client_portion_id_list)))
            self.test()
    
            self.arrived = [0] * self.client_number

      
    def start_server_process(self):
        for i in range(1, self.client_number+1):
            t = threading.Thread(target=self.start_client_process, args=(i, ))
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

server_process = server(4, 2)
server_process.start_server_process()