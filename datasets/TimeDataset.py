import torch
from torch.utils.data import Dataset, DataLoader

import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np


class TimeDataset(Dataset):
    def __init__(self, raw_data, edge_index, mode='train', config = None):
        self.raw_data = raw_data # length is number of sensors (for each sensor) plus the attack column. each index is the time-series data for the sensors.

        self.config = config
        self.edge_index = edge_index
        self.mode = mode

        x_data = raw_data[:-1] # excludes the labels column so that it's only the sensor data
        labels = raw_data[-1]

        data = x_data

        data = torch.tensor(data).double()
        labels = torch.tensor(labels).double()
        print('length of labels before process: ', len(labels), '\n')
        self.x, self.y, self.labels = self.process(data, labels) # self.y has the dimensions of self.x tranposed
        print('length of labels after process: ', len(self.labels), '\n')
    
    def __len__(self):
        return len(self.x)


    def process(self, data, labels): # labels is number of timestamps X 1
        print('DATA')
        print(data) # dimensions of data are number of sensors X number of timestamps
        print('LABELS')
        print(labels) 
        x_arr, y_arr = [], []
        labels_arr = []

        # by default, the slide_win is 5 and the slide_stride is 1
        slide_win, slide_stride = [self.config[k] for k
            in ['slide_win', 'slide_stride']
        ]
        is_train = self.mode == 'train'

        node_num, total_time_len = data.shape

        rang = range(slide_win, total_time_len, slide_stride) if is_train else range(slide_win, total_time_len)
        
        print("DATA Here")
        print(len(list(data)))
        for i in rang:

            ft = data[:, i-slide_win:i] # the window of inputs for each sensor. ft is the number of sensors X the sliding window
            tar = data[:, i] # tar has dimensions 1 X number of sensors. Each value in tar represents the value that comes right after the sliding window for each sensor
            x_arr.append(ft)
            y_arr.append(tar) 

            labels_arr.append(labels[i]) # labels[i] is the attack value (1 or 0) that comes right after the sliding window.
        print('IN TimeDataset.process')
        print(len(x_arr)) # 12656
        print(len(y_arr)) # 12656
        x = torch.stack(x_arr).contiguous()
        y = torch.stack(y_arr).contiguous()

        labels = torch.Tensor(labels_arr).contiguous()
        
        return x, y, labels # x, y, and labels are both (5 shorter with sliding window of 5) shorter than the original length of the data because of the sliding window. They are all the same length

    def __getitem__(self, idx):

        feature = self.x[idx].double()
        y = self.y[idx].double()

        edge_index = self.edge_index.long()

        label = self.labels[idx].double()

        return feature, y, label, edge_index





