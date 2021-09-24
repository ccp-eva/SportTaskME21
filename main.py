import cv2
import datetime
import os
import platform
import time
import torch
import xml.etree.ElementTree as ET
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from utils import *
from model import *
print('Python version : ', platform.python_version())
print('OpenCV version  : ', cv2.__version__)
print('Torch version : ', torch.__version__)
# Opencv use several cpus by default for simple operation. Using only one allows loading data in parallel much faster
cv2.setNumThreads(0)
print('Nb of threads for OpenCV : ', cv2.getNumThreads())

#################################################################
###################### Model variables ##########################
#################################################################
class my_variables():
    def __init__(self, task_path, size_data=[98, 120, 120], cuda=True, batch_size=15, workers=6, epochs=1000, lr=0.0001, nesterov=True, weight_decay=0.005, momentum=0.5):
        self.cuda = cuda
        self.workers = workers
        self.batch_size = batch_size
        self.size_data = np.array(size_data)
        self.epochs = epochs
        self.lr = lr
        self.nesterov = nesterov
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.model_name = os.path.join(task_path, 'MediaEval21_%s' % (datetime.datetime.now().strftime('%d-%m-%Y_%H-%M')))
        make_path(task_path)
        make_path(self.model_name)
        if cuda:
            self.dtype = torch.cuda.FloatTensor
            # os.environ[ 'CUDA_VISIBLE_DEVICES' ] = '0'
        else:
            self.dtype = torch.FloatTensor
        self.log = setup_logger('model_log', os.path.join(self.model_name, 'model_log.log'))

####################################################################################
################################ Get annotations ###################################
####################################################################################
''' My_dataset class which uses My_stroke class to be used in the data loader'''
class My_dataset(Dataset):
    def __init__(self, dataset_list, size_data):
        self.dataset_list = dataset_list
        self.size_data = size_data

    def __len__(self):
        return len(self.dataset_list)

    def __getitem__(self, idx):
        rgb = get_data(self.dataset_list[idx].video_path, self.dataset_list[idx].begin, self.size_data)
        sample = {'rgb': torch.FloatTensor(rgb), 'label' : self.dataset_list[idx].move, 'my_stroke' : {'video_path':self.dataset_list[idx].video_path, 'begin':self.dataset_list[idx].begin, 'end':self.dataset_list[idx].end}}
        return sample

''' My_stroke class used for encoding the annotations'''
class My_stroke:
    def __init__(self, video_path, begin, end, move):
        self.video_path = video_path
        self.begin = begin
        self.end = end
        self.move = move

    def my_print(self, log=None):
        print_and_log('Video : %s\tbegin : %d\tEnd : %d\tClass : %s' % (self.video_path, self.begin, self.end, self.move), log=log)

''' Get annotations from xml files located in one folder and produce a list of My_stroke'''
def get_annotations(xml_path, data_folder, list_of_strokes=None):
    xml_list = [os.path.join(xml_path, f) for f in os.listdir(xml_path) if os.path.isfile(os.path.join(xml_path, f)) and f.split('.')[-1]=='xml']
    strokes_list = []
    for xml_file in xml_list:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        video_path = os.path.join(data_folder, xml_file.split('/')[-1].split('.')[0])
        for action in root:
            if list_of_strokes is None:
                strokes_list.append(My_stroke(video_path, int(action.get('begin')), int(action.get('end')), 1))
            else:
                strokes_list.append(My_stroke(video_path, int(action.get('begin')), int(action.get('end')), list_of_strokes.index(action.get('move'))))
        # Case of the test set in segmentation task - build proposals of size 150
        if len(root)==0: 
            for begin in range(0,len(os.listdir(video_path))-150,150):
                strokes_list.append(My_stroke(video_path, begin, begin+150, 0))
    return strokes_list

'''Infer Negative Samples from annotation betwen strokes when there are more than length_min frames'''
def build_negative_strokes(stroke_list, length_min=200):
    video_path = 'tmp'
    for stroke in stroke_list.copy():
        if stroke.video_path != video_path:
            video_path = stroke.video_path
            begin_negative = 0
        end_negative = stroke.begin
        for end in range(begin_negative+length_min, end_negative, length_min):
            stroke_list.append(My_stroke(video_path, end-length_min, end, 0))
        begin_negative = stroke.end

''' Get the rgb frames from the annotations'''
def get_data(data_path, begin, size_data):
    rgb_data = []
    for frame_number in range(begin, begin + size_data[0]):
        try:
            rgb = cv2.imread(os.path.join(data_path, '%08d.png' % frame_number))
            rgb = cv2.resize(rgb, (size_data[1], size_data[2])).astype(float) / 255
        except:
            raise ValueError('Problem with %s begin %d size %d' % (os.path.join(data_path, '%08d.png' % frame_number), begin, size_data[0]))
        rgb_data.append(cv2.split(rgb))

    rgb_data = np.transpose(rgb_data, (1, 0, 2, 3))
    return rgb_data


##########################################################################
######################### Model Architecture #############################
##########################################################################
def make_architecture(args, output_size):
    print_and_log('Make Model', log=args.log)
    model = NetSimpleBranch(args.size_data.copy(), output_size)
    ## Use GPU
    if args.cuda:
        model.cuda()
    return model

##########################################################################
############################# Training ###################################
##########################################################################
''' Training is split in train epoch and validation epoch and produce a plot'''
def train_model(model, args, train_loader, valid_loader):
    criterion = nn.CrossEntropyLoss(reduction='sum')
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)

    begin_time = time.time()
    print_and_log('\nTraining...', log=args.log)

    # For plot
    loss_train = []
    loss_val = []
    acc_val = []
    acc_train = []

    for epoch in range(args.epochs):
        # Train and validation step and save loss and acc for plot
        loss_train_, acc_train_ = train_epoch(epoch, args, model, train_loader, optimizer, criterion)
        loss_val_, acc_val_ = validation_epoch(epoch, args, model, valid_loader, criterion)

        loss_train.append(loss_train_)
        acc_train.append(acc_train_)
        loss_val.append(loss_val_)
        acc_val.append(acc_val_)
    print_and_log('Max validation accuracy of %.2f done in %ds' % (max(acc_val), int(time.time()-begin_time)), log=args.log)
    make_train_figure(loss_train, loss_val, acc_val, acc_train, os.path.join(args.model_name, 'Train.png'))
    return 1

''' Update of the model in one epoch'''
def train_epoch(epoch, args, model, data_loader, optimizer, criterion):
    model.train()
    pid = os.getpid()
    N = len(data_loader.dataset)
    begin_time = time.time()
    aLoss = 0
    Acc = 0

    for batch_idx, batch in enumerate(data_loader):
        # Get batch tensor
        rgb, label = batch['rgb'], batch['label']

        rgb = Variable(rgb.type(args.dtype))
        label = Variable(label.type(args.dtype).long())

        optimizer.zero_grad()
        output = model(rgb)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        aLoss += loss.item()
        Acc += output.data.max(1)[1].eq(label.data).cpu().sum().numpy()
        progress_bar((batch_idx + 1) * args.batch_size, N, '%d Training - Epoch : %d - Batch Loss = %.5g' % (pid, epoch, loss.item()))

    aLoss /= N
    progress_bar(N, N, 'Train - Epoch %d - Loss = %.5g - Accuracy = %.3g (%d/%d) - Time = %ds' % (epoch, aLoss, Acc/N, Acc, N, time.time() - begin_time), 1, log=args.log)
    return aLoss, Acc/N


'''Validation of the model in one epoch'''
def validation_epoch(epoch, args, model, data_loader, criterion):
    with torch.no_grad():
        begin_time = time.time()
        pid = os.getpid()
        N = len(data_loader.dataset)
        _loss = 0
        _acc = 0

        for batch_idx, batch in enumerate(data_loader):
            progress_bar(batch_idx*args.batch_size, N, '%d - Validation' % (pid))
            rgb, label = batch['rgb'], batch['label']
            rgb = Variable(rgb.type(args.dtype))
            label = Variable(label.type(args.dtype).long())
            output = model(rgb)
            _loss += criterion(output, label).item()
            output_indexes = output.data.max(1)[1]
            _acc += output.data.max(1)[1].eq(label.data).cpu().sum().numpy()

        _loss /= N
        progress_bar(N, N, 'Validation - Loss = %.5g - Accuracy = %.3g (%d/%d) - Time = %ds' % (_loss, _acc/N, _acc, N, time.time() - begin_time), 1, log=args.log)
        return _loss, _acc/N

####################################################################
######################## Test Process ##############################
####################################################################
'''Store data for xml files from the list of stroke with predicted class - for detection it is saved when index predicted to 1'''
def store_xml_data(my_stroke_list, predicted, xml_files, list_of_strokes=None):
    for video_path, begin, end, prediction_index in zip(my_stroke_list['video_path'], my_stroke_list['begin'].tolist(), my_stroke_list['end'].tolist(), predicted):
        video_name = video_path.split('/')[-1]
        if video_name not in xml_files:
            xml_files[video_name] = ET.Element('video')
        if list_of_strokes is None:
            if prediction_index:
                stroke_xml = ET.SubElement(xml_files[video_name], 'action')
                stroke_xml.set('begin', str(begin))
                stroke_xml.set('end', str(end))
        else:
            stroke_xml = ET.SubElement(xml_files[video_name], 'action')
            stroke_xml.set('begin', str(begin))
            stroke_xml.set('end', str(end))
            stroke_xml.set('move', list_of_strokes[prediction_index])

'''Save the predictions in xml files'''
def save_xml_data(xml_files, path_xml_save):
    for video_name in xml_files:
        xml_file = open('%s.xml' % os.path.join(path_xml_save, video_name), 'wb')
        xml_file.write(ET.tostring(xml_files[video_name]))
        xml_file.close()

'''Inference on test set'''
def test_model(model, args, data_loader, list_of_strokes=None):
    with torch.no_grad():
        xml_files = {}
        path_xml_save = os.path.join(args.model_name, 'xml_test')
        make_path(path_xml_save)
        N = len(data_loader.dataset)
        
        for batch_idx, batch in enumerate(data_loader):
            # Get batch tensor
            rgb, my_stroke_list = batch['rgb'], batch['my_stroke']
            progress_bar(args.batch_size*batch_idx, N, 'Testing')

            rgb = Variable(rgb.type(args.dtype))
            output = model(rgb)
            _, predicted = torch.max(output.detach(), 1)
            store_xml_data(my_stroke_list, predicted, xml_files, list_of_strokes)

        progress_bar(N, N, 'Test done', 1, log=args.log)
        save_xml_data(xml_files, path_xml_save)


########################################
################ Dev env. ##############
########################################
'''Set up the environment and extract data'''
def make_work_tree(main_folder, source_folder, frame_width=320, extract=False):
    data_path = os.path.join(main_folder, 'data')
    video_folder = os.path.join(source_folder, 'videos')
    detection_path = os.path.join(source_folder,'detectionTask')
    classification_path = os.path.join(source_folder,'classificationTask')
    if extract:
        make_path(main_folder)
        make_path(data_path)
        video_list = [_file for _file in os.listdir(video_folder) if _file[-4:]=='.mp4' and os.path.isfile(os.path.join(video_folder, _file))]
        for idx, video in enumerate(video_list):
            save_frame_path = os.path.join(data_path, video[:-4])
            make_path(save_frame_path)
            progress_bar(idx, len(video_list), 'Frame extraction')
            frame_extractor(os.path.join(video_folder, video), frame_width, save_frame_path)
        progress_bar(len(video_list), len(video_list), 'Frame extraction done', 1)
    return main_folder, data_path, detection_path, classification_path

''' According to overview paper'''
def get_list_of_strokes():
    list_of_strokes = ['Serve Forehand Backspin',
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
                   'Defensive Backhand Backspin',
                   'Unknown']
    return list_of_strokes

'''Get the split of annotation and construct negative samples fro; it if in dectetion task'''
def get_lists_annotations(task_path, data_path, list_of_strokes=None):
    train_strokes = get_annotations(os.path.join(task_path, 'train'), data_path, list_of_strokes)
    valid_strokes = get_annotations(os.path.join(task_path, 'valid'), data_path, list_of_strokes)
    test_strokes = get_annotations(os.path.join(task_path, 'test'), data_path, list_of_strokes)
    if list_of_strokes is None:
        build_negative_strokes(train_strokes)
        build_negative_strokes(valid_strokes)
        build_negative_strokes(test_strokes)
    return train_strokes, valid_strokes, test_strokes

''' Build dataloader from list of strokes'''
def get_data_loaders(train_strokes, valid_strokes, test_strokes, size_data, batch_size, workers):
    # Build Dataset
    train_set = My_dataset(train_strokes, size_data)
    valid_set = My_dataset(valid_strokes, size_data)
    test_set = My_dataset(test_strokes, size_data)

    # Loaders of the Datasets
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=workers, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, num_workers=workers, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=workers)
    return train_loader, valid_loader, test_loader

'''Classification task'''
def classification_task(main_folder, data_path, task_path):
    print('\nClassification Task')
    # Initial list
    reset_training(1)
    list_of_strokes = get_list_of_strokes()

    # Split
    train_strokes, valid_strokes, test_strokes = get_lists_annotations(task_path, data_path, list_of_strokes)

    # Model variables
    args = my_variables('classificationTask')
    
    ## Architecture with the output of the lenght of possible classes - (Unknown not counted)
    model = make_architecture(args, len(list_of_strokes)-1)

    # Loaders
    train_loader, valid_loader, test_loader = get_data_loaders(train_strokes, valid_strokes, test_strokes, args.size_data, args.batch_size, args.workers)

    # Training process
    train_model(model, args, train_loader, valid_loader)
    
    # Test process 
    test_model(model, args, test_loader, list_of_strokes)
    return 1


'''Detection task'''
def detection_task(main_folder, data_path, task_path):
    print('\nDetection Task')
    # Initial list
    reset_training(1)
    list_of_strokes = get_list_of_strokes()

    # Split
    train_strokes, valid_strokes, test_strokes = get_lists_annotations(task_path, data_path)

    # Model variables
    args = my_variables('detectionTask')

    # Architecture with the output of the lenght of possible classes - Positive and Negative
    model = make_architecture(args, 2)

    # Loaders
    train_loader, valid_loader, test_loader = get_data_loaders(train_strokes, valid_strokes, test_strokes, args.size_data, args.batch_size, args.workers)

    # Training process
    train_model(model, args, train_loader, valid_loader)
    
    # Test process 
    test_model(model, args, test_loader)

    return 1


if __name__ == "__main__":
    # MediaEval Task source folder
    source_folder = '../data'
    
    # Prepare tree and data - To call only once with extract set to True
    main_folder, data_path, detection_path, classification_path = make_work_tree('.', source_folder, extract=False)

    # Tasks
    classification_task(main_folder, data_path, classification_path)
    detection_task(main_folder, data_path, detection_path)

    print_and_log('All Done')
