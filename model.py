import torch.nn as nn
import torch.nn.functional as F

##########################################################################
########################  Flatten Features  ##############################
##########################################################################
def flatten_features(x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features

##########################################################################
#############################  One Branch ################################
##########################################################################
class NetSimpleBranch(nn.Module):
    def __init__(self, size_data, n_classes, channels=3):
        super(NetSimpleBranch, self).__init__()

        ####################
        ####### First ######
        ####################
        self.conv1 = nn.Conv3d(channels, 30, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))
        size_data //= 2

        ####################
        ###### Second ######
        ####################
        self.conv2 = nn.Conv3d(30, 60, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)) 
        self.pool2 = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))
        size_data //= 2

        ####################
        ####### Third ######
        ####################
        self.conv3 = nn.Conv3d(60, 80, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)) 
        self.pool3 = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))
        size_data //= 2

        ####################
        ####### Last #######
        ####################
        self.linear1 = nn.Linear(80*size_data[0]*size_data[1]*size_data[2], 500)
        self.relu = nn.ReLU()

        # Fusion
        self.linear2 = nn.Linear(500, n_classes)
        self.final = nn.Softmax(1)

    def forward(self, data):

        ####################
        ####### First ######
        ####################
        data = self.pool1(F.relu(self.conv1(data)))

        ####################
        ###### Second ######
        ####################
        data = self.pool2(F.relu(self.conv2(data)))

        ####################
        ####### Third ######
        ####################
        data = self.pool3(F.relu(self.conv3(data)))


        ####################
        ####### Last #######
        ####################
        data = data.view(-1, flatten_features(data))
        data = self.relu(self.linear1(data))

        data = self.linear2(data)
        label = self.final(data)

        return label