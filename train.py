import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import time
from util.time import *
from util.env import *
from sklearn.metrics import mean_squared_error
from test import *
import torch.nn.functional as F
import numpy as np
from evaluate import get_best_performance_data, get_val_performance_data, get_full_err_scores
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score
from torch.utils.data import DataLoader, random_split, Subset
from scipy.stats import iqr
import tqdm
import time
import csv

loss_list = []
ACU_loss_list = []

def loss_func(y_pred, y_true):
    #print('y_pred shape: ', y_pred.shape)
    #print('y_true shape: ', y_true.shape)
    loss = F.mse_loss(y_pred, y_true, reduction='mean')

    return loss


def recompute_labels(labels, slide_win, slide_stride):
    original_lables = labels
    new_labels = []

    #for i in range(0, len(labels)-slide_stride):
    #    for i in range(i+)



def train(model = None, save_path = '', config={},  train_dataloader=None, val_dataloader=None, feature_map={}, test_dataloader=None, test_dataset=None, dataset_name='swat', train_dataset=None):

    time_each_epoch_takes = list()

    seed = config['seed']

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=config['decay'])

    now = time.time()
    
    train_loss_list = []
    cmp_loss_list = []

    device = get_device()


    acu_loss = 0
    min_loss = 1e+8
    min_f1 = 0
    min_pre = 0
    best_prec = 0

    i = 0
    print('in train.py')
    epoch = config['epoch']
    print(epoch)
    early_stop_win = 15

    model.train()

    log_interval = 1000
    stop_improve_count = 0

    dataloader = train_dataloader

    epoch_progress = tqdm.tqdm(total=epoch, desc='Epoch')
    for i_epoch in range(epoch):
        start = time.time()

        acu_loss = 0 
        model.train() # simply puts the model in training mode

        for x, labels, attack_labels, edge_index in dataloader: # size of x: [32, 17, 5]; size of labels: [32, 17]; attack_labels: [32]
            # length of x (and labels and attack_labels since they are all the same length) is reduced since some of the training dataset was split to become the validation and because data is loaded in batches
            _start = time.time()

            x, labels, edge_index = [item.float().to(device) for item in [x, labels, edge_index]]

            optimizer.zero_grad()
            out = model(x, edge_index).float().to(device) # does forward propagation

            loss = loss_func(out, labels) # measures how well model predicted the sensor data (labels) at the current timestamp
            
            loss.backward()
            optimizer.step()

            
            train_loss_list.append(loss.item())
            acu_loss += loss.item()
                
            i += 1

        loss_list.append(acu_loss/len(dataloader))
        ACU_loss_list.append(acu_loss)

        # each epoch
        print('epoch ({} / {}) (Loss:{:.8f}, ACU_loss:{:.8f})'.format(
                        i_epoch, epoch, 
                        acu_loss/len(dataloader), acu_loss), flush=True
            )


        # use val dataset to judge
        if val_dataloader is not None:

            val_loss, val_result = test(model, val_dataloader)

            if val_loss < min_loss:
                torch.save(model.state_dict(), save_path)

                min_loss = val_loss
                stop_improve_count = 0
            else:
                stop_improve_count += 1

            epoch_progress.update(1)

            if stop_improve_count >= early_stop_win:
                break

        else:
            if acu_loss < min_loss :
                torch.save(model.state_dict(), save_path)
                min_loss = acu_loss

        time_each_epoch_takes.append(time.time()-start)

    with open(f'C:\\Users\\yasi4\\OneDrive\\Documents\\GitHub\\GDN\\TimeDifference_epoch_train_losses.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Loss', 'ACU_loss'])
        for i in range(0, len(loss_list)):
            writer.writerow([i, loss_list[i], ACU_loss_list[i]])
    
    with open(f'C:\\Users\\yasi4\\OneDrive\\Documents\\GitHub\\GDN\\TimeDifference_training_duration.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Duration in seconds, Total Time in minutes'])

        total_time = 0
        for i in range(0, len(time_each_epoch_takes)):
            total_time += time_each_epoch_takes[i]
            writer.writerow([i+1, time_each_epoch_takes[i], total_time/60])


    return train_loss_list
