import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.dataset import ConcatDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

home_dir = '../frcnn_csi/'
noise = False

def load_dataset(dataset_name, data_type):
    data_dir = home_dir+"data/"+dataset_name+"/"
    
    data = sio.loadmat(data_dir + data_type + '_data.mat')[data_type + '_data']
    if dataset_name=="TEMPORAL" and data_type=="train":
        location = sio.loadmat(data_dir + data_type + '_grountruth.mat')[data_type + '_grountruth']
    else:
        location = sio.loadmat(data_dir + data_type + '_groundtruth.mat')[data_type + '_groundtruth']
    location = location / 1.0

    data_class = sio.loadmat(data_dir + data_type + '_class.mat')[data_type+'_class']
    num_instances = len(data)

    # 类型转换 data type and location type
    data = torch.from_numpy(data).type(torch.FloatTensor)
    location = torch.from_numpy(location).type(torch.FloatTensor)
    data_class = torch.from_numpy(data_class).type(torch.FloatTensor)

    dataset = TensorDataset(data, location, data_class)

    return dataset, num_instances


def get_data_loader(dataset_name, data_type, batch_size, shuffle=False):
    if dataset_name == "S":
        return _get_data_S_loader(data_type, batch_size, shuffle)
        
    dataset, num_instances = load_dataset(dataset_name, data_type)

    # 数据增强：噪声
    if noise:
        dataset, num_instances = data_argumentation(dataset, num_instances)
    
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
    return num_instances, data_loader


def _get_data_S_loader(data_type, batch_size, shuffle=False):

    dataset_S1, num_instances_S1 = load_dataset("192S1ALL", data_type)

    dataset_S2, num_instances_S2 = load_dataset("192S2", data_type)

    concat_data = ConcatDataset([dataset_S1, dataset_S2])
    num_instances = num_instances_S1+num_instances_S2

    # 数据增强：噪声
    #concat_data, num_instances = data_argumentation(concat_data, num_instances)
    
    data_loader = DataLoader(dataset=concat_data, batch_size=batch_size, shuffle=shuffle)
    return num_instances, data_loader


def data_argumentation(dataset, num_instances):
    datas = []
    bboxs = []
    labels = [] 
    for (data, bbox, label) in dataset:
        tmp = AddGaussianNoise(amplitude=0.3)(data.clone())
        datas.append(tmp)
        bboxs.append(bbox)
        labels.append(label)

        # tmp2 = salt_and_pepper(data.clone(), 0.03)
        # datas.append(tmp2)
        # bboxs.append(bbox)
        # labels.append(label)
    datas = torch.cat(datas).unsqueeze(1)
    bboxs = torch.cat(bboxs).view(-1,2)
    labels = torch.cat(labels).view(-1,1)

    new_dataset = TensorDataset(datas, bboxs, labels)

    con = ConcatDataset([dataset, new_dataset])

    return con, num_instances + datas.shape[0]


# 椒盐噪声   
# input：复制后的tensor
# prob: 噪声出现的概率
def salt_and_pepper(input,prob):
    noise_tensor=torch.rand(input.shape)
    salt=torch.max(input)
    pepper=torch.min(input)
    input[noise_tensor<prob/2]=salt
    input[noise_tensor>1-prob/2]=pepper
    return input

# 高斯噪声
# input：复制后的tensor
class AddGaussianNoise(object):

    def __init__(self, mean=0.0, variance=1.0, amplitude=1.0):

        self.mean = mean
        self.variance = variance
        self.amplitude = amplitude

    def __call__(self, input):
        input = np.array(input)
        N = self.amplitude * np.random.normal(loc=self.mean, scale=self.variance, size=input.shape)
        input = N + input
        input[input > 255] = 255                       # 避免有值超过255而反转
        return torch.from_numpy(input).float()

if __name__ == "__main__":
    os.system("rm predict/*.png")
    num_test_instances, test_data_loader = get_data_loader("TEMPORAL", "test", batch_size=1, shuffle = False)
    i=0
    for (data, bbox, label) in tqdm(test_data_loader):
        if i==0 or i==215 or i==216:
            X=np.linspace(0,192,192,endpoint=True)
            plt.contourf(data[0][0].cpu())
            plt.colorbar()
            currentAxis=plt.gca()
            plt.savefig("predict/orignal%d.png"%(i), dpi=520)
            plt.show()
            plt.close()

        # tmp = AddGaussianNoise(amplitude=0.3)(data.clone())
        # X=np.linspace(0,192,192,endpoint=True)
        # plt.contourf(tmp[0][0].cpu())
        # plt.colorbar()
        # currentAxis=plt.gca()
        # plt.savefig("predict/gauss%d.png"%(i), dpi=520)
        # plt.show()
        # plt.close()

        # tmp = salt_and_pepper(data.clone(), 0.03)
        # X=np.linspace(0,192,192,endpoint=True)
        # plt.contourf(tmp[0][0].cpu())
        # plt.colorbar()
        # currentAxis=plt.gca()
        # plt.savefig("predict/salt%d.png"%(i), dpi=520)
        # plt.show()
        # plt.close()

        i+=1
    print(i)




# #test_phase = sio.loadmat('data\\192S2\\test_phase.mat')
# #test_phase = test_phase['test_phase']
# #test_phase = torch.from_numpy(test_phase).type(torch.FloatTensor)
# #test_data = torch.cat((test_data,test_phase),1)