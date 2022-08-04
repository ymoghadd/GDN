# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split, Subset

from sklearn.preprocessing import MinMaxScaler

from util.env import get_device, set_device
from util.preprocess import build_loc_net, construct_data
from util.net_struct import get_feature_map, get_fc_graph_struc
from util.iostream import printsep

from datasets.TimeDataset import TimeDataset


from models.GDN import GDN

from train import train
from test  import test
from evaluate import get_err_scores, get_best_performance_data, get_val_performance_data, get_full_err_scores

import sys
from datetime import datetime

import os
import argparse
from pathlib import Path

import matplotlib.pyplot as plt

import json
import random

import time
import csv

import pickle


class Main():
    def __init__(self, train_config, env_config, debug=False):

        self.edges = []
        self.num_nodes = 0

        self.training_time = 0
        self.testing_time1 = 0
        self.testing_time2 = 0

        self.train_config = train_config
        self.env_config = env_config
        self.datestr = None

        dataset = self.env_config['dataset'] 
        train_orig = pd.read_csv(f'./data/{dataset}/train.csv', sep=',', index_col=0)
        test_orig = pd.read_csv(f'./data/{dataset}/test.csv', sep=',', index_col=0)
       


        train = train_orig  # they're all 1.0???
        train, test = train_orig, test_orig

        if 'attack' in train.columns:
            train = train.drop(columns=['attack'])

        feature_map = get_feature_map(dataset) # all the sensor names in list.txt
        fc_struc = get_fc_graph_struc(dataset)

        set_device(env_config['device'])
        self.device = get_device()

        fc_edge_index = build_loc_net(fc_struc, list(train.columns), feature_map=feature_map)
        fc_edge_index = torch.tensor(fc_edge_index, dtype = torch.long)

        self.feature_map = feature_map

        self.original_labels = test.attack.tolist()

        train_dataset_indata = construct_data(train, feature_map, labels=0)
        self.original_attacks = test.attack.tolist()
        test_dataset_indata = construct_data(test, feature_map, labels=test.attack.tolist())

        cfg = {
            'slide_win': train_config['slide_win'],
            'slide_stride': train_config['slide_stride'],
        }

        train_dataset = TimeDataset(train_dataset_indata, fc_edge_index, mode='train', config=cfg)

        test_dataset = TimeDataset(test_dataset_indata, fc_edge_index, mode='test', config=cfg)
        print('test raw_data length: ', len(test_dataset.raw_data), '\n')

        train_dataloader, val_dataloader = self.get_loaders(train_dataset, train_config['seed'], train_config['batch'], val_ratio = train_config['val_ratio'])

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset


        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = DataLoader(test_dataset, batch_size=train_config['batch'],
                            shuffle=False, num_workers=0)

        edge_index_sets = []
        edge_index_sets.append(fc_edge_index)

        self.model = GDN(edge_index_sets, len(feature_map), dim=train_config['dim'], input_dim=train_config['slide_win'],out_layer_num=train_config['out_layer_num'],out_layer_inter_dim=train_config['out_layer_inter_dim'],topk=train_config['topk']).to(self.device)



    def run(self):

        if len(self.env_config['load_model_path']) > 0:
            model_save_path = self.env_config['load_model_path']
        else:
            model_save_path = self.get_save_path()[0]
            print('MODEL_SAVE_PATH')
            print(model_save_path)

            start = time.time()
            print('in main.run', train_config['epoch'])
            #import pdb; pdb.set_trace()
            self.train_log = train(self.model, model_save_path, config = train_config, train_dataloader=self.train_dataloader, val_dataloader=self.val_dataloader, feature_map=self.feature_map,test_dataloader=self.test_dataloader,test_dataset=self.test_dataset,train_dataset=self.train_dataset,dataset_name=self.env_config['dataset'])
            time_after_training = time.time()
            self.training_time = time_after_training-start
        # test            
        self.model.load_state_dict(torch.load(model_save_path))
        best_model = self.model.to(self.device)

        start1 = time.time()
        _, self.test_result = test(best_model, self.test_dataloader) # all three indices of self.test_result have length 12656. Need self.test_result -> test labels list as the ground values for the anomaly labels
        '''
        print(self.test_result[1])
        all_predictions = []
        for lst in self.test_result[1]:
            all_predictions.extend(lst)
        

        
        with open('C:\\Users\\yasi4\\OneDrive\\Documents\\GitHub\\GDN\\Actual_vs_Predictions.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['Actual', 'Predictions'])
            for i in range(0, len(all_predictions)):
                writer.writerow([self.original_attacks[i], all_predictions[i]])
        '''

        time_after_testing1 = time.time()
        self.testing_time1 = time_after_testing1-start

        start2 = time.time()
        _, self.val_result = test(best_model, self.val_dataloader)
        time_after_testing2 = time.time()
        self.testing_time2 = time_after_testing2-start2

        '''
        with open(f'C:\\Users\\yasi4\\OneDrive\\Documents\\GitHub\\GDN\\msl_original_testing_duration.csv', 'w') as f:
            writer = csv.writer(f)

            writer.writerow(['Duration of Testing Dataset in seconds', 'Duration of Validation Dataset in seconds'])
            writer.writerow([self.testing_time1, self.testing_time2])
        '''
        

        self.get_score(self.test_result, self.val_result)

    def get_loaders(self, train_dataset, seed, batch, val_ratio=0.1):
        dataset_len = int(len(train_dataset)) # 12656
        train_use_len = int(dataset_len * (1 - val_ratio)) # 10124
        val_use_len = int(dataset_len * val_ratio)
        val_start_index = random.randrange(train_use_len)
        indices = torch.arange(dataset_len)

        train_sub_indices = torch.cat([indices[:val_start_index], indices[val_start_index+val_use_len:]])
        train_subset = Subset(train_dataset, train_sub_indices)

        val_sub_indices = indices[val_start_index:val_start_index+val_use_len]
        val_subset = Subset(train_dataset, val_sub_indices)


        train_dataloader = DataLoader(train_subset, batch_size=batch,
                                shuffle=True)

        val_dataloader = DataLoader(val_subset, batch_size=batch,
                                shuffle=False)

        return train_dataloader, val_dataloader

    def get_score(self, test_result, val_result):

        feature_num = len(test_result[0][0]) # test_result[0] is the test_predicted_list, which has 12656 rows, each having five elements (the size of the sliding window; they are predictions for a particular timestamp)
        np_test_result = np.array(test_result)

        np_val_result = np.array(val_result)


        test_labels = np_test_result[2, :, 0].tolist() # a single-dimensional array of length 12656 of the true values for the test labels as anomalous or not 
        val_labels = np_val_result[2, :, 0].tolist()


        #import pdb; pdb.set_trace()
        test_scores, normal_scores = get_full_err_scores(test_result, val_result) # test scores will later be used to determine if there is an anomaly or not


        #print('length of test labels: ', len(test_labels))
        #print('length of test_predictions: ', len(test_result[0]))

        top1_best_info = get_best_performance_data(test_scores, test_labels, topk=1) # MUST CREATE CONFUSION MATRIX IN THIS METHOD
        top1_val_info = get_val_performance_data(test_scores, normal_scores, test_labels, topk=1)


        print('=========================** Result **============================\n')

        info = None
        if self.env_config['report'] == 'best':
            info = top1_best_info
        elif self.env_config['report'] == 'val':
            info = top1_val_info

        print(f'F1 score: {info[0]}')
        print(f'precision: {info[1]}')
        print(f'recall: {info[2]}\n')

        '''
        with open(f'C:\\Users\\yasi4\\OneDrive\\Documents\\GitHub\\GDN\\msl_original_Results.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['F1 score', 'Precision', 'Recall'])
            writer.writerow([info[0], info[1], info[2]])

        
        with open(f'C:\\Users\\yasi4\\OneDrive\\Documents\\GitHub\\GDN\\msl_original_actual_vs_predicted.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['Actual values', 'Predicted values'])
            for actual, predicted in zip(self.original_attacks, test_labels):
                writer.writerow([actual, predicted])
        '''
        
        
    def get_save_path(self, feature_name=''):

        dir_path = self.env_config['save_path']
        
        if self.datestr is None:
            now = datetime.now()
            self.datestr = now.strftime('%m-%d-%H_%M_%S') # MODIFIED: before was self.datestr = now.strftime('%m|%d-%H:%M:%S')
        datestr = self.datestr          

        paths = [
            f'C:\\Users\\yasi4\\OneDrive\\Documents\\GitHub\\GDN\\pretrained\\{dir_path}\\msl_original_best_{datestr}.pt',
            f'\\Users\\yasi4\\OneDrive\\Documents\\GitHub\\GDN\\results\\{dir_path}\\{datestr}.csv',
        ] # MODIFIED: before was paths = [f'./pretrained/{dir_path}/best_{datestr}.pt', f'./results/{dir_path}/{datestr}.csv',]

        for path in paths:
            dirname = os.path.dirname(path)
            Path(dirname).mkdir(parents=True, exist_ok=True)

        return paths

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('-batch', help='batch size', type = int, default=128)
    parser.add_argument('-epoch', help='train epoch', type = int, default=100)
    parser.add_argument('-slide_win', help='slide_win', type = int, default=15)
    parser.add_argument('-dim', help='dimension', type = int, default=64)
    parser.add_argument('-slide_stride', help='slide_stride', type = int, default=5)
    parser.add_argument('-save_path_pattern', help='save path pattern', type = str, default='')
    parser.add_argument('-dataset', help='wadi / swat', type = str, default='wadi')
    parser.add_argument('-device', help='cuda / cpu', type = str, default='cuda')
    parser.add_argument('-random_seed', help='random seed', type = int, default=0)
    parser.add_argument('-comment', help='experiment comment', type = str, default='')
    parser.add_argument('-out_layer_num', help='outlayer num', type = int, default=1)
    parser.add_argument('-out_layer_inter_dim', help='out_layer_inter_dim', type = int, default=256)
    parser.add_argument('-decay', help='decay', type = float, default=0)
    parser.add_argument('-val_ratio', help='val ratio', type = float, default=0.0)
    parser.add_argument('-topk', help='topk num', type = int, default=15)
    parser.add_argument('-report', help='best / val', type = str, default='best')
    parser.add_argument('-load_model_path', help='trained model path', type = str, default='')

    args = parser.parse_args()

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(args.random_seed)

    print('here')
    print('epochs: ', args.epoch)

    train_config = {
        'batch': 1, # originally args.batch
        'epoch': 100,
        'slide_win': args.slide_win,
        'dim': 128,
        'slide_stride': args.slide_stride,
        'comment': args.comment,
        'seed': args.random_seed,
        'out_layer_num': args.out_layer_num,
        'out_layer_inter_dim': args.out_layer_inter_dim,
        'decay': args.decay,
        'val_ratio': 0.1,
        'topk': 3,
    }
    print('train_config[topk]')
    print(train_config['topk'])


    env_config={
        'save_path': args.save_path_pattern,
        'dataset': args.dataset,
        'report': 'val', # THIS HAS BEEN CHANGED TO USE THE VALIDATION DATASET FOR EVALUATION INSTEAD OF THE TEST DATASET
        'device': args.device,
        'load_model_path': args.load_model_path
    }

    #import pdb; pdb.set_trace()

    main = Main(train_config, env_config, debug=False)

    #import pdb; pdb.set_trace()
    main.run()