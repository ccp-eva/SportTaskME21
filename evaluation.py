'''Evaluation script by Pierre-Etienne MARTIN dedicated to the Sport Task for MediaEval 2021'''
import os
import cv2
import argparse
import numpy as np
from xml.etree import ElementTree
import operator, pdb

dict_of_moves = ['Serve Forehand Backspin',
                'Serve Forehand Loop',
                'Serve Forehand Sidespin',
                'Serve Forehand Topspin',

                'Serve Backhand Backspin',
                'Serve Backhand Loop',
                'Serve Backhand Sidespin',
                'Serve Backhand Topspin',

                'Offensive Forehand Hit',
                'Offensive Forehand Loop',
                'Offensive Forehand Flip',

                'Offensive Backhand Hit',
                'Offensive Backhand Loop',
                'Offensive Backhand Flip',

                'Defensive Forehand Push',
                'Defensive Forehand Block',
                'Defensive Forehand Backspin',

                'Defensive Backhand Push',
                'Defensive Backhand Block',
                'Defensive Backhand Backspin']

def compute_iou(gt, prediction, print_option=False):
    gt_not = list(map(operator.not_, gt))
    intersection = np.logical_and(gt, prediction)
    union = np.logical_or(gt, prediction)
    iou_score = 1.*np.sum(intersection) / np.sum(union)
    
    if print_option:
        prediction_not = list(map(operator.not_, prediction))
        intersection_not = np.logical_and(gt_not, prediction_not)
        TP = np.sum(intersection)
        TN = np.sum(intersection_not)
        FP = np.sum(np.logical_and(prediction, gt_not))
        FN = np.sum(np.logical_and(prediction_not, gt))
        print('Iou is : %.3f   TP: %d, TN: %d, FP: %d, FN: %d' % (iou_score, TP, TN, FP, FN))
        print('\trecall: %.3f, Precision: %.3f, TNR: %.3f, FPR: %.3f, FNR: %.3f' % (TP/(TP+FN), TP/(TP+FP), TN/(TN+FP), FP/(FP+TN), FN/(TP+FN)))
    return iou_score


def evaluate_classification(run_path, set_path):
    original_xml_list = os.listdir(set_path)
    numCorrectActions = 0
    numActions = 0

    numCorrectActions_h = dict()
    numActions_h = dict()
    for move in dict_of_moves:
        numCorrectActions_h[move] = 0
        numActions_h[move] = 0
    

    for file in original_xml_list:        
        gt_file_path = os.path.join(set_path, file)
        results_file_path = os.path.join(run_path, file)

        if not os.path.exists(results_file_path):
            raise ValueError('The result xml file does not exist: {}'.format(results_file_path))
        
        tree = ElementTree.parse(gt_file_path)
        root = tree.getroot()
        gt_actions = []
        for action in root:
            gt_actions.append([int(action.get('begin')), int(action.get('end')), action.get('move')])
        gt_actions.sort()

        tree = ElementTree.parse(results_file_path)
        root = tree.getroot()
        results_actions = []
        for action in root:
            results_actions.append([int(action.get('begin')), int(action.get('end')), action.get('move')])
        results_actions.sort()
        
        if len(gt_actions) != len(results_actions):
            raise ValueError('The xmls do not have the same number of actions: {}'.format(file))

        for gt_action, results_action in zip(gt_actions, results_actions):
            if gt_action[0] != results_action[0]:
                raise ValueError('The begin frame of the actions has been modified %s' % file)

            if gt_action[1] != results_action[1]:
                raise ValueError('The end frame of the actions has been modified in %s' % file)

            if results_action[2] not in dict_of_moves:
                raise ValueError('The move associated to the action in %s is not in the possible moves : %s not in : ' % (file, results_action[2]), dict_of_moves)

            if gt_action[2] not in dict_of_moves:
                raise ValueError('The move associated to the ground truth action in %s is not in the possible moves : %s not in : ' % (file, gt_action[2]), dict_of_moves)

            numActions += 1

            numActions_h[gt_action[2]] += 1
            
            if (results_action[2] == gt_action[2]):
                #print('CORRECT : res={} | gt={}'.format(results_action[2], gt_action[2]))
                numCorrectActions += 1
                numCorrectActions_h[gt_action[2]] += 1
            #else:
                #print('NOK : res={} | gt={}'.format(results_action[2], gt_action[2]))
                
                
                
    accuracy = numCorrectActions / float(numActions) 
    print('\nGlobal accuracy={}/{}={}\n'.format(numCorrectActions, numActions, accuracy))
    print('Accuracy per move class:')
    for move in dict_of_moves:
        if (numActions_h[move] != 0):
            print(' {} : accuracy={}/{}={}'.format(move, numCorrectActions_h[move], numActions_h[move], numCorrectActions_h[move]/numActions_h[move]))
        else:
            print(' {} : accuracy=N/A'.format(move))

def evaluate_detection(run_path, set_path):
    original_xml_list = os.listdir(set_path)

    num_gt_actions_total = 0
    num_submitted_actions_total = 0
    gt_array_total = []
    submitted_array_total = []
    iou_thresholds = [0.5 + idx/20 for idx in range(10)]
    TP = np.zeros(len(iou_thresholds))
    FP = np.zeros(len(iou_thresholds))
    FN = np.zeros(len(iou_thresholds))
    
    for file in original_xml_list:
        action_list_gt = []
        action_list_pred = []
        gt_file_path = os.path.join(set_path, file)
        submitted_file_path = os.path.join(run_path, file)

        video = cv2.VideoCapture(os.path.join("videos",os.path.splitext(file)[0] + ".mp4"))

        try:
            N_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        except:
            N_frames = int(video.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
        
        video.release()
        gt_array = np.zeros(N_frames)
        submitted_array = np.zeros(N_frames)
        num_gt_actions = 0
        num_submitted_actions = 0

        if not os.path.exists(submitted_file_path):
            raise ValueError('The result xml file does not exist: %s' % (submitted_file_path))
        
        tree = ElementTree.parse(gt_file_path)
        root = tree.getroot()
        for action in root:
            num_gt_actions += 1
            begin = int(action.get('begin'))
            end = int(action.get('end'))
            action_list_gt.append([begin, end])
            gt_array[begin:end] = 1

        tree = ElementTree.parse(submitted_file_path)
        root = tree.getroot()
        for action in root:
            num_submitted_actions += 1
            begin = int(action.get('begin'))
            end = int(action.get('end'))
            action_list_pred.append([begin, end])
            if (end-begin<0) or (begin<0) or (end<0):
                raise ValueError('%s: An action must have boundaries making sense - begin %d and end %d given' % (file, begin, end))
            if (end > N_frames) or (begin > N_frames):
                raise ValueError('%s: An action must have boundaries within the video - begin %d end %d given for video with %d frames.' % (file, begin, end, N_frames))
            submitted_array[begin:end] = 1

        # Compare each detected stroke with gt per video in order to get iou and accordingly increment TP, FP, FN
        check_FN = np.array([[1 for _ in iou_thresholds] for action in action_list_gt])
        for idx_pred, action_pred in enumerate(action_list_pred):
            check_FP = [1 for _ in iou_thresholds]
            vector_pred = np.zeros(N_frames)
            vector_pred[action_pred[0]:action_pred[1]] = 1
            for idx_gt, action_gt in enumerate(action_list_gt):
                vector_gt = np.zeros(N_frames)
                vector_gt[action_gt[0]:action_gt[1]] = 1
                iou = compute_iou(vector_pred, vector_gt)
                for idx_th, iou_th in enumerate(iou_thresholds):
                    # Stroke has been for the first time detected and overlaps with GT stroke
                    if iou >= iou_th and check_FN[idx_gt,idx_th]!=0:
                        TP[idx_th]+=1
                        check_FP[idx_th]=0
                        check_FN[idx_gt,idx_th]=0
            # Stroke has been detected but no GT Stroke
            FP+=check_FP
        # GT stroke has not been detected
        FN+=check_FN.sum(0)
  
        gt_array_total = np.append(gt_array_total, gt_array)
        submitted_array_total = np.append(submitted_array_total, submitted_array)
        num_gt_actions_total += num_gt_actions
        num_submitted_actions_total += num_submitted_actions

    print("\nFrame as one object to detect:\n")
    print("Number of actions detected vs GT: %d / %d" % (num_submitted_actions_total, num_gt_actions_total))
    compute_iou(gt_array_total, submitted_array_total, print_option=True)

    for idx in range(len(iou_thresholds)):
        if num_gt_actions_total != TP[idx]+FN[idx]:
            print('This should not appear. Contact the organizers.')

    print('\nStroke as one object to detect:\n')
    # Compute recall and precision (graph possible)
    recall = TP/(TP+FN)
    precision = TP/(TP+FP)

    # Max left
    recall_sorted, precision_sorted = zip(*sorted(zip(recall, precision)))
    recall_sorted = np.array(recall_sorted)
    precision_sorted_maxleft = [max(precision_sorted[idx:]) for idx in range(len(precision_sorted))]
    AP = (precision_sorted_maxleft*(np.append(recall_sorted[0],recall_sorted[1:]-recall_sorted[:-1]))).sum()
    
    for idx in range(len(iou_thresholds)):
        print("With IoU threshold of %g" % iou_thresholds[idx])
        print('\tPrecision: %.3f, Recall: %.3f, dedicated AP: %.3f' % (precision[idx], recall[idx], precision[idx]*recall[idx]))
    
    print("\nAverage Precision at IoU=.50:.05:.95 = %f" % AP)

    return 1



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Evaluation of the participant')
    parser.add_argument('path', help='Folder path in which you have the subfolders classificationTask and detectionTask containg subfolders representings the runs in which there are the xml files filled')
    parser.add_argument('set_path', nargs='?', default='testGT', help='Set on which the run has been done (per default test but you may run some checks on your valid and train sets too)')
    args = parser.parse_args()

    if args.set_path not in ['train', 'valid', 'test', 'testGT']:
        raise ValueError('Please provide a correct set name (train, valid or test')

    classification_path = os.path.join(args.path, "classificationTask")
    if os.path.isdir(classification_path):
        print('\nClassification task:')
        idx=0
        for idx, run in enumerate(os.listdir(classification_path)):
            run_path = os.path.join(classification_path, run)
            if os.path.isdir(run_path):
                idx+=1
                print('\nRun %d (%s):' % (idx, run))
                try:
                    evaluate_classification(run_path, os.path.join("classificationTask", args.set_path))
                except ValueError as error:
                    print(error)
                    continue

            
    detection_path = os.path.join(args.path, "detectionTask")
    if os.path.isdir(detection_path):
        print('\nDetection task:')
        idx=0
        for idx, run in enumerate(os.listdir(detection_path)):
            run_path = os.path.join(detection_path, run)
            if os.path.isdir(run_path):
                idx+=1
                print('\nRun %d (%s)' % (idx, run))
                try:
                    evaluate_detection(run_path, os.path.join("detectionTask", args.set_path))
                except ValueError as error:
                    print(error)
                    continue
    
    print('Test done')

    
    
    
