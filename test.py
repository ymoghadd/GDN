import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import time
from util.time import *
from util.env import *

import argparse
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch.nn.functional as F


from util.data import *
from util.preprocess import *

import pickle
import copy

def test(model, dataloader):
    # test
    loss_func = nn.MSELoss(reduction='mean')
    device = get_device()

    test_loss_list = []
    now = time.time()

    test_predicted_list = []
    test_ground_list = []
    test_labels_list = []

    t_test_predicted_list = []
    t_test_ground_list = []
    t_test_labels_list = []

    test_len = len(dataloader)

    model.eval()

    i = 0
    acu_loss = 0


    # ORIGINAL
    for x, y, labels, edge_index in dataloader: # x dim: 32 X 5 X 5; y dim: 32 X 5; labels dim: 32 X 5
        x, y, labels, edge_index = [item.to(device).float() for item in [x, y, labels, edge_index]]
        
        with torch.no_grad():
            predicted = model(x, edge_index).float().to(device)
            
            
            loss = loss_func(predicted, y)
            

            labels = labels.unsqueeze(1).repeat(1, predicted.shape[1])

            if len(t_test_predicted_list) <= 0:
                t_test_predicted_list = predicted
                t_test_ground_list = y
                t_test_labels_list = labels
            else:
                t_test_predicted_list = torch.cat((t_test_predicted_list, predicted), dim=0)
                t_test_ground_list = torch.cat((t_test_ground_list, y), dim=0)
                t_test_labels_list = torch.cat((t_test_labels_list, labels), dim=0)
        
        test_loss_list.append(loss.item())
        acu_loss += loss.item()
        
        i += 1

        if i % 10000 == 1 and i > 1:
            print(timeSincePlus(now, i / test_len))


    test_predicted_list = t_test_predicted_list.tolist()        
    test_ground_list = t_test_ground_list.tolist()        
    test_labels_list = t_test_labels_list.tolist()      
    
    avg_loss = sum(test_loss_list)/len(test_loss_list)

    return avg_loss, [test_predicted_list, test_ground_list, test_labels_list]
    # ORIGINAL


    '''
    # since there aren't any batches, then a dataloader isn't technically needed; so x, y, and labels don't come from dataloader
    list_x = dataloader.__dict__['dataset'].__dict__['dataset'].__dict__['x'] # list of lists: each list is a window
    list_x_dynamic = copy.deepcopy(list_x) # this constantly gets updated with predictions
    list_y = [] # each 
    list_edge_index = []

    prediction_list = []
    previous_window = []

    threshold = 0.0 # some number learned from training
    with torch.no_grad():
        predicted = model(list_x[0], list_edge_index[0]).float().to(device)
        if predicted > threshold:
            predicted_binary = 1.0
        else:
            predicted_binary = 0.0
        prediction_list.append(predicted)
        list_x_dynamic[1][4] = predicted
    
    for i in range(0, len(list_x[1:])-1):
        with torch.no_grad():
            predicted = model(list_x_dynamic[i], list_edge_index[i]).float().to(device)
            if predicted > threshold:
                predicted_binary = 1.0
            else:
                predicted_binary = 0.0
            prediction_list.append(predicted)
            loss = loss_func(predicted, list_y[i])
            prediction_list.append(predicted)
            list_x_dynamic[i+1][4] = predicted
    '''

'''
# REAL TIME
    i1 = 1
    for x, y, labels, edge_index in dataloader: # x dim: 32 X 5 X 5; y dim: 32 X 5; labels dim: 32 X 5
        print('x size: ', x.size())
        print(x)
        print('y size: ', y.size())
        x, y, labels, edge_index = [item.to(device).float() for item in [x, y, labels, edge_index]]


        with torch.no_grad():
            print('HIIIIIIIIIIIIIIIIIIIIIIIIIIIII')
            predicted = model(x, edge_index).float().to(device)

            #for sensor_num in range(0, x.size(1))
            #print(predicted.size(1))
            #print(predicted[0][1])
            #print(dataloader.__dict__['dataset'].__dict__['dataset'].__dict__['x'].size())
            #print(dataloader.__dict__['dataset'].__dict__['dataset'].__dict__['x'][i].size())

            x_dataset = dataloader.__dict__['dataset'].__dict__.get('dataset', 0)
            if (x_dataset):
                for j in range(0, x_dataset.__dict__['x'][i1].size(1)):
                    print('INSIDE FOR LOOP in TEST.PY')
                    x_dataset.__dict__['x'][i1][4] = predicted[0][j]


            
            # for j in range(0, dataloader.__dict__['dataset'].__dict__['dataset'].__dict__['x'][i].size(1)):
            #    print('INSIDE FOR LOOP IN TEST.PY')
            #    dataloader.__dict__['dataset'].__dict__['dataset'].__dict__['x'][i][4] = predicted[0][j]
            
            #print(dataloader.__dict__['dataset'].__dict__['dataset'].__dict__['x'].size())
            #print(dataloader.__dict__['dataset'].__dict__['dataset'].__dict__['y'].size())
            #dataloader.__dict__['dataset'].__dict__['dataset'].__dict__['x'][i][4] = predicted
            i1 += 1
            loss = loss_func(predicted, y)
            


            labels = labels.unsqueeze(1).repeat(1, predicted.shape[1])

            if len(t_test_predicted_list) <= 0:
                t_test_predicted_list = predicted
                t_test_ground_list = y
                t_test_labels_list = labels
            else:
                t_test_predicted_list = torch.cat((t_test_predicted_list, predicted), dim=0)
                t_test_ground_list = torch.cat((t_test_ground_list, y), dim=0)
                t_test_labels_list = torch.cat((t_test_labels_list, labels), dim=0)

        test_loss_list.append(loss.item())
        acu_loss += loss.item()
        
        i += 1

        if i % 10000 == 1 and i > 1:
            print(timeSincePlus(now, i / test_len))

    print('outside of outer for loop')
    test_predicted_list = t_test_predicted_list.tolist()        
    test_ground_list = t_test_ground_list.tolist()        
    test_labels_list = t_test_labels_list.tolist()      

    print('test_predicted_list rows: ', len(test_predicted_list))
    print('test_ground_list rows: ', len(test_ground_list))
    print('test_predicted_list columns: ', len(test_predicted_list[0]))
    print('test_ground_list columns: ', len(test_ground_list[0]))

    avg_loss = sum(test_loss_list)/len(test_loss_list)

    return avg_loss, [test_predicted_list, test_ground_list, test_labels_list]

    '''


