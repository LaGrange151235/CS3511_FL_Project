import torchvision
import torch.utils.data as Data
from torchvision import transforms

import dill

class Train_dataset(Data.Dataset):
    def __init__(self, data_path):
        f = open(data_path, 'rb')
        train_dataset = dill.load(f)
        img_list = []
        label_list = []
        for i in range(len(train_dataset)):
            img, label = train_dataset[i]
            img_list.append(img)
            label_list.append(label)
        self.data = img_list
        self.label = label_list
    def __getitem__(self, index):
        data = self.data[index]
        label = self.label[index]
        return data, label
    def __len__(self):
        return len(self.data)
    
def get_train_dataset(data_path):
    return Train_dataset(data_path)

def get_test_dataset():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = torchvision.datasets.MNIST('./data', train=False, download=True, transform=transform)
    return test_dataset
