from util.data import *
import numpy as np
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score, confusion_matrix, precision_recall_curve, roc_curve
import pickle
import csv


from numpy import sqrt
from numpy import argmax
from matplotlib import pyplot

def get_full_err_scores(test_result, val_result):
    np_test_result = np.array(test_result)
    np_val_result = np.array(val_result)

    all_scores =  None
    all_normals = None
    feature_num = np_test_result.shape[-1]

    labels = np_test_result[2, :, 0].tolist()

    for i in range(feature_num):
        test_re_list = np_test_result[:2,:,i]
        val_re_list = np_val_result[:2,:,i]

        scores = get_err_scores(test_re_list, val_re_list)
        normal_dist = get_err_scores(val_re_list, val_re_list)

        if all_scores is None:
            all_scores = scores
            all_normals = normal_dist
        else:
            all_scores = np.vstack((
                all_scores,
                scores
            ))
            all_normals = np.vstack((
                all_normals,
                normal_dist
            ))

    return all_scores, all_normals


def get_final_err_scores(test_result, val_result):
    full_scores, all_normals = get_full_err_scores(test_result, val_result, return_normal_scores=True)

    all_scores = np.max(full_scores, axis=0)

    return all_scores



def get_err_scores(test_res, val_res):
    test_predict, test_gt = test_res
    val_predict, val_gt = val_res

    n_err_mid, n_err_iqr = get_err_median_and_iqr(test_predict, test_gt)

    test_delta = np.abs(np.subtract(
                        np.array(test_predict).astype(np.float64), 
                        np.array(test_gt).astype(np.float64)
                    ))
    epsilon=1e-2

    err_scores = (test_delta - n_err_mid) / ( np.abs(n_err_iqr) +epsilon)

    smoothed_err_scores = np.zeros(err_scores.shape)
    before_num = 3
    for i in range(before_num, len(err_scores)):
        smoothed_err_scores[i] = np.mean(err_scores[i-before_num:i+1])

    
    return smoothed_err_scores



def get_loss(predict, gt):
    return eval_mseloss(predict, gt)

def get_f1_scores(total_err_scores, gt_labels, topk=1):
    print('total_err_scores', total_err_scores.shape)
    # remove the highest and lowest score at each timestep
    total_features = total_err_scores.shape[0]

    # topk_indices = np.argpartition(total_err_scores, range(total_features-1-topk, total_features-1), axis=0)[-topk-1:-1]
    topk_indices = np.argpartition(total_err_scores, range(total_features-topk-1, total_features), axis=0)[-topk:]
    
    topk_indices = np.transpose(topk_indices)

    total_topk_err_scores = []
    topk_err_score_map=[]
    # topk_anomaly_sensors = []

    for i, indexs in enumerate(topk_indices):
       
        sum_score = sum( score for k, score in enumerate(sorted([total_err_scores[index, i] for j, index in enumerate(indexs)])) )

        total_topk_err_scores.append(sum_score)

    final_topk_fmeas = eval_scores(total_topk_err_scores, gt_labels, 400)

    return final_topk_fmeas

def get_val_performance_data(total_err_scores, normal_scores, gt_labels, topk=1):
    total_features = total_err_scores.shape[0]

    topk_indices = np.argpartition(total_err_scores, range(total_features-topk-1, total_features), axis=0)[-topk:]

    total_topk_err_scores = []
    topk_err_score_map=[]

    total_topk_err_scores = np.sum(np.take_along_axis(total_err_scores, topk_indices, axis=0), axis=0)

    thresold = np.max(normal_scores)

    pred_labels = np.zeros(len(total_topk_err_scores))
    pred_labels[total_topk_err_scores > thresold] = 1

    for i in range(len(pred_labels)):
        pred_labels[i] = int(pred_labels[i])
        gt_labels[i] = int(gt_labels[i])

    pre = precision_score(gt_labels, pred_labels)
    rec = recall_score(gt_labels, pred_labels)

    f1 = f1_score(gt_labels, pred_labels)


    auc_score = roc_auc_score(gt_labels, total_topk_err_scores)

    return f1, pre, rec, auc_score, thresold


def get_best_performance_data(total_err_scores, gt_labels, topk=1): # total_err_scores has the dimensions number of sensors X number of timestamps (12656)

    total_features = total_err_scores.shape[0] # total number of test sensors

    # topk_indices = np.argpartition(total_err_scores, range(total_features-1-topk, total_features-1), axis=0)[-topk-1:-1]
    topk_indices = np.argpartition(total_err_scores, range(total_features-topk-1, total_features), axis=0)[-topk:] # has length 12656 and is one dimensional. It has values 0, 1, 2, 3, 4

    total_topk_err_scores = []
    topk_err_score_map=[]

    total_topk_err_scores = np.sum(np.take_along_axis(total_err_scores, topk_indices, axis=0), axis=0) # sums all the sensor errors (all 5) for each timestamp

    final_topk_fmeas ,thresolds = eval_scores(total_topk_err_scores, gt_labels, 400, return_thresold=True)

    th_i = final_topk_fmeas.index(max(final_topk_fmeas)) 
    thresold = thresolds[th_i] # thresold is actually a score in total_topk_err_scores. It is determined by the error score that results in the highest f1 score
    print('THRESHOLD: ', thresold)
    '''
    precision, recall, thresholds = precision_recall_curve(gt_labels, total_topk_err_scores)
    print(len(precision))
    print('Precision: ', precision)
    print('Recall: ', recall)
    print('Thresholds: ', thresholds)
    # convert to f score
    fscore = (2 * precision * recall) / (precision + recall)
    print('f1 score: ', fscore)
    # locate the index of the largest f score
    ix = argmax(fscore)
    print('Best Threshold=%f, F-Score=%.3f' % (thresholds[ix], fscore[ix]))

    thresold = thresholds[ix]
    '''

    ###################################################
    # THIS IS WHERE THEY DETERMINE WHETHER A TIMESTAMP IS AN ANOMALY OR NOT
    pred_labels = np.zeros(len(total_topk_err_scores))
    #thresold = 0.551697 # originally -0.27180347019679296, which resulted in 93.8% of the data being labeled as anomalous
    pred_labels[total_topk_err_scores > thresold] = 1
    ###################################################


    with open('C:\\Users\\yasi4\\OneDrive\\Documents\\GitHub\\GDN\\msl_original_INFO.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Total Topk Error Scores', 'Predicted Values', 'Ground Values'])
        for i in range(0, len(gt_labels)):
            writer.writerow([total_topk_err_scores[i], pred_labels[i], gt_labels[i]])

    
    for i in range(len(pred_labels)):
        pred_labels[i] = int(pred_labels[i]) # both pred_labels and gt_labels have a length of 12656
        gt_labels[i] = int(gt_labels[i])

    '''
    # calculate roc curves
    fpr, tpr, thresholds = roc_curve(gt_labels, total_topk_err_scores)
    # calculate the g-mean for each threshold
    gmeans = sqrt(tpr * (1-fpr))
    # locate the index of the largest g-mean
    ix = argmax(gmeans)
    print('Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix])) # Best Threshold=0.551697, G-Mean=0.504
    # plot the roc curve for the model
    pyplot.plot([0,1], [0,1], linestyle='--', label='No Skill')
    pyplot.plot(fpr, tpr, marker='.', label='Logistic')
    pyplot.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Best')
    # axis labels
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    pyplot.legend()
    # show the plot
    pyplot.show()
    '''

    confusionMatrix = confusion_matrix(list(gt_labels), list(pred_labels))

    print(confusionMatrix)
    tn, fp, fn, tp = confusionMatrix.ravel()
    print('True negative: ', tn, '\n')
    print('False postiive: ', fp, '\n')
    print("False negative: ", fn, '\n')
    print('True positive: ', tp, '\n')

    with open(f'C:\\Users\\yasi4\\OneDrive\\Documents\\GitHub\\GDN\\confusion_matrix_msl_original.pickle', 'wb') as f:
        pickle.dump(confusionMatrix, f)



    pre = precision_score(gt_labels, pred_labels)
    rec = recall_score(gt_labels, pred_labels)

    auc_score = roc_auc_score(gt_labels, total_topk_err_scores)

    return max(final_topk_fmeas), pre, rec, auc_score, thresold

