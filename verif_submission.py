'''Verification script by Pierre-Etienne MARTIN dedicated to the Sport Task for MediaEval 2021'''
import os
import numpy as np
import argparse
import cv2, pdb
from xml.etree import ElementTree
import operator

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

def compute_iou(gt, prediction):
    gt_not = list(map(operator.not_, gt))
    intersection = np.logical_and(gt, prediction)
    union = np.logical_or(gt, prediction)
    iou_score = 1.*np.sum(intersection) / np.sum(union)
    return iou_score

def check_classification_run(run_path, set_path):
    original_xml_list = os.listdir(set_path)
    submitted_xml_list = os.listdir(run_path)
    output = ''

    if len(original_xml_list) != len(submitted_xml_list):
        output += '\nNot the same number of xml files'

    for file in original_xml_list:

        if file not in submitted_xml_list:
            output += '\n%s not found in submission' % (file)
            continue

        tree = ElementTree.parse(os.path.join(set_path, file))
        root = tree.getroot()
        test_actions = []
        for action in root:
            test_actions.append([int(action.get('begin')), int(action.get('end'))])
        test_actions.sort()

        tree = ElementTree.parse(os.path.join(run_path, file))
        root = tree.getroot()
        output_actions = []
        for action in root:
            output_actions.append([int(action.get('begin')), int(action.get('end')), action.get('move')])
        output_actions.sort()

        if len(test_actions) != len(output_actions):
            output += '\n%s: Not the same number of actions' % (file)
            continue

        for test_action, output_action in zip(test_actions, output_actions):
            if test_action[0] != output_action[0]:
                output += '\n%s: The strating frame of the actions has been modified' % (file)

            if test_action[1] != output_action[1]:
                output += '\n%s: The ending frame of the actions has been modified' % (file)

            if output_action[2] not in dict_of_moves:
                output += '\n%s: "%s" is not in the possible moves: %s' % (file, output_action[2], dict_of_moves)

    if output == '':
        return 'OK'
    else:
        return output

def check_detection_run(run_path, set_path):
    original_xml_list = os.listdir(set_path)
    submitted_xml_list = os.listdir(run_path)
    output = ''

    if len(original_xml_list) != len(submitted_xml_list):
        output += '\nNot the same number of xml files'

    for file in original_xml_list:

        action_list = []
        video = cv2.VideoCapture(os.path.join("videos",os.path.splitext(file)[0] + ".mp4"))

        try:
            N_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        except:
            N_frames = int(video.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))

        video.release()

        if file not in submitted_xml_list:
            output += '\n%s not found in submission' % (file)
            continue

        tree = ElementTree.parse(os.path.join(run_path, file))
        root = tree.getroot()
        boundaries = []
        for action in root:
            begin = int(action.get('begin'))
            end = int(action.get('end'))
            action_list.append([begin, end])
            if (end-begin<0) or (begin<0) or (end<0):
                output += '\n%s: An action must have boundaries making sense - begin %d and end %d given' % (file, begin, end)

            if end > N_frames or begin > N_frames:
                output += '\n%s: An action must have boundaries within the video - begin %d end %d given for video with %d frames.' % (file, begin, end, N_frames)
        
        # Compare each detected stroke one by one per video in order to check if the iou is less than .5 (commutative) - may take a while
        for idx, action1 in enumerate(action_list[:-1]):
            vector1 = np.zeros(N_frames)
            vector1[action1[0]:action1[1]] = 1
            for action2 in action_list[idx+1:]:
                vector2 = np.zeros(N_frames)
                vector2[action2[0]:action2[1]] = 1
                if compute_iou(vector1, vector2)>.5:
                    output += '\n%s: Two detected strokes should have an iou < .5 or else considred has a same stroke. (stroke1 [%d,%d] stroke2 [%d,%d]' % (file, action1[0], action1[1], action2[0], action2[1])

    if output == '':
        return 'OK'
    else:
        return output

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Let\'s test your submitted files')
    parser.add_argument('path', help='Folder path in which you have the subfolders classificationTask and detectionTask containg subfolders representings your runs in which there are the xml files filled')
    parser.add_argument('set_path', nargs='?', default='test', help='Set on which the run has been done (per default test but you may run some checks on your valid and train sets too)')
    args = parser.parse_args()

    print(args.set_path)

    if args.set_path not in ['train', 'valid', 'test', 'testGT']:
        raise ValueError('Please provide a correct set name: train, valid or test. %s provided.' % (args.set_path))

    classification_path = os.path.join(args.path, "classificationTask")
    if os.path.isdir(classification_path):
        print('Classification task:')
        idx=0
        for idx, run in enumerate(os.listdir(classification_path)):
            run_path = os.path.join(classification_path, run)
            if os.path.isdir(run_path):
                idx+=1
                print('Run %d (%s): %s' % (idx, run, check_classification_run(run_path, os.path.join("classificationTask", args.set_path))))
            
    detection_path = os.path.join(args.path, "detectionTask")
    if os.path.isdir(detection_path):
        print('Detection task:')
        idx=0
        for idx, run in enumerate(os.listdir(detection_path)):
            run_path = os.path.join(detection_path, run)
            if os.path.isdir(run_path):
                idx+=1
                print('Run %d (%s): %s' % (idx, run, check_detection_run(run_path, os.path.join("detectionTask", args.set_path))))
    
    print('Test done')
    