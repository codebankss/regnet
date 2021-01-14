################################################################################
#
# IMPORT
#
################################################################################

# torch
import torch
import torch.nn       as     nn
import torch.optim    as     optim
from   torch.autograd import Function

# torch utils
import torchvision
import torchvision.transforms as transforms

# additional libraries
import os
import urllib.request
import zipfile
import time
import math
import numpy             as np
import matplotlib.pyplot as plt

################################################################################
#
# PARAMETERS
#
################################################################################

# data
DATA_DIR_1        = 'data'
DATA_DIR_2        = 'data/imagenet64'
DATA_DIR_TRAIN    = 'data/imagenet64/train'
DATA_DIR_TEST     = 'data/imagenet64/val'
DATA_FILE_TRAIN_1 = 'Train1.zip'
DATA_FILE_TRAIN_2 = 'Train2.zip'
DATA_FILE_TRAIN_3 = 'Train3.zip'
DATA_FILE_TRAIN_4 = 'Train4.zip'
DATA_FILE_TRAIN_5 = 'Train5.zip'
DATA_FILE_TEST_1  = 'Val1.zip'
DATA_URL_TRAIN_1  = 'https://github.com/arthurredfern/UT-Dallas-CS-6301-CNNs/raw/master/Data/Train1.zip'
DATA_URL_TRAIN_2  = 'https://github.com/arthurredfern/UT-Dallas-CS-6301-CNNs/raw/master/Data/Train2.zip'
DATA_URL_TRAIN_3  = 'https://github.com/arthurredfern/UT-Dallas-CS-6301-CNNs/raw/master/Data/Train3.zip'
DATA_URL_TRAIN_4  = 'https://github.com/arthurredfern/UT-Dallas-CS-6301-CNNs/raw/master/Data/Train4.zip'
DATA_URL_TRAIN_5  = 'https://github.com/arthurredfern/UT-Dallas-CS-6301-CNNs/raw/master/Data/Train5.zip'
DATA_URL_TEST_1   = 'https://github.com/arthurredfern/UT-Dallas-CS-6301-CNNs/raw/master/Data/Val1.zip'
DATA_BATCH_SIZE   = 512
DATA_NUM_WORKERS  = 4
DATA_NUM_CHANNELS = 3
DATA_NUM_CLASSES  = 100
DATA_RESIZE       = 64
DATA_CROP         = 56
DATA_MEAN         = (0.485, 0.456, 0.406)
DATA_STD_DEV      = (0.229, 0.224, 0.225)

# model
MODEL_STEM_WIDTH    = 32
MODEL_SLOPE_PARAM   = 36   
MODEL_INITIAL_WIDTH = 24        
MODEL_QUANTI_PARAM  = 2.5    
MODEL_NUM_GROUPS    = 8
MODEL_NETWORK_DEPTH = 16     


u = MODEL_INITIAL_WIDTH + MODEL_SLOPE_PARAM * np.arange(MODEL_NETWORK_DEPTH)
s = np.log(u / MODEL_INITIAL_WIDTH) / np.log(MODEL_QUANTI_PARAM)
s = np.round(s)

MODEL_STAGE_WIDTH                       = MODEL_INITIAL_WIDTH * np.power(MODEL_QUANTI_PARAM, s)
MODEL_STAGE_WIDTH                       = np.round(MODEL_STAGE_WIDTH / MODEL_NUM_GROUPS) * MODEL_NUM_GROUPS
MODEL_STAGE_WIDTH, MODEL_STAGE_DEPTH    = np.unique(MODEL_STAGE_WIDTH.astype(np.int), return_counts=True)
print("Calculated Model Stage Width is " + str(MODEL_STAGE_WIDTH))

# training (linear warm up with cosine decay learning rate)
TRAINING_LR_MAX          = 0.001
TRAINING_LR_INIT_SCALE   = 0.01
TRAINING_LR_INIT_EPOCHS  = 5
TRAINING_LR_FINAL_SCALE  = 0.01
TRAINING_LR_FINAL_EPOCHS = 55
# TRAINING_LR_FINAL_EPOCHS = 2 # uncomment for a quick test
TRAINING_NUM_EPOCHS      = TRAINING_LR_INIT_EPOCHS + TRAINING_LR_FINAL_EPOCHS
TRAINING_LR_INIT         = TRAINING_LR_MAX*TRAINING_LR_INIT_SCALE
TRAINING_LR_FINAL        = TRAINING_LR_MAX*TRAINING_LR_FINAL_SCALE

# file
FILE_NAME = 'RegNetX-200MF.pt'
FILE_SAVE = 1
FILE_LOAD = 0

################################################################################
#
# DATA
#
################################################################################

# create a local directory structure for data storage
if (os.path.exists(DATA_DIR_1) == False):
    os.mkdir(DATA_DIR_1)
if (os.path.exists(DATA_DIR_2) == False):
    os.mkdir(DATA_DIR_2)
if (os.path.exists(DATA_DIR_TRAIN) == False):
    os.mkdir(DATA_DIR_TRAIN)
if (os.path.exists(DATA_DIR_TEST) == False):
    os.mkdir(DATA_DIR_TEST)

# download data
if (os.path.exists(DATA_FILE_TRAIN_1) == False):
    urllib.request.urlretrieve(DATA_URL_TRAIN_1, DATA_FILE_TRAIN_1)
if (os.path.exists(DATA_FILE_TRAIN_2) == False):
    urllib.request.urlretrieve(DATA_URL_TRAIN_2, DATA_FILE_TRAIN_2)
if (os.path.exists(DATA_FILE_TRAIN_3) == False):
    urllib.request.urlretrieve(DATA_URL_TRAIN_3, DATA_FILE_TRAIN_3)
if (os.path.exists(DATA_FILE_TRAIN_4) == False):
    urllib.request.urlretrieve(DATA_URL_TRAIN_4, DATA_FILE_TRAIN_4)
if (os.path.exists(DATA_FILE_TRAIN_5) == False):
    urllib.request.urlretrieve(DATA_URL_TRAIN_5, DATA_FILE_TRAIN_5)
if (os.path.exists(DATA_FILE_TEST_1) == False):
    urllib.request.urlretrieve(DATA_URL_TEST_1, DATA_FILE_TEST_1)

# extract data
with zipfile.ZipFile(DATA_FILE_TRAIN_1, 'r') as zip_ref:
    zip_ref.extractall(DATA_DIR_TRAIN)
with zipfile.ZipFile(DATA_FILE_TRAIN_2, 'r') as zip_ref:
    zip_ref.extractall(DATA_DIR_TRAIN)
with zipfile.ZipFile(DATA_FILE_TRAIN_3, 'r') as zip_ref:
    zip_ref.extractall(DATA_DIR_TRAIN)
with zipfile.ZipFile(DATA_FILE_TRAIN_4, 'r') as zip_ref:
    zip_ref.extractall(DATA_DIR_TRAIN)
with zipfile.ZipFile(DATA_FILE_TRAIN_5, 'r') as zip_ref:
    zip_ref.extractall(DATA_DIR_TRAIN)
with zipfile.ZipFile(DATA_FILE_TEST_1, 'r') as zip_ref:
    zip_ref.extractall(DATA_DIR_TEST)

# transforms
transform_train = transforms.Compose([transforms.RandomResizedCrop(DATA_CROP), transforms.RandomHorizontalFlip(p=0.5), transforms.ToTensor(), transforms.Normalize(DATA_MEAN, DATA_STD_DEV)])
transform_test  = transforms.Compose([transforms.Resize(DATA_RESIZE), transforms.CenterCrop(DATA_CROP), transforms.ToTensor(), transforms.Normalize(DATA_MEAN, DATA_STD_DEV)])

# data sets
dataset_train = torchvision.datasets.ImageFolder(DATA_DIR_TRAIN, transform=transform_train)
dataset_test  = torchvision.datasets.ImageFolder(DATA_DIR_TEST,  transform=transform_test)

# data loader
dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=DATA_BATCH_SIZE, shuffle=True,  num_workers=DATA_NUM_WORKERS, pin_memory=True, drop_last=True)
dataloader_test  = torch.utils.data.DataLoader(dataset_test,  batch_size=DATA_BATCH_SIZE, shuffle=False, num_workers=DATA_NUM_WORKERS, pin_memory=True, drop_last=True)


################################################################################
#
# NETWORK BUILDING BLOCK
#
################################################################################

# Batch Normalization and Activation
class BatchNormActi(nn.Module):

    def __init__(self, C_in, C_out, size, stride, padding, groups=1, batch_normalization=True, activation=True):

        super(BatchNormActi, self).__init__()

        self.batch_normalization = batch_normalization
        self.activation          = activation
        self.conv                = nn.Conv2d(C_in, C_out, size, stride=stride, padding=padding, groups=groups, bias=False)

        if self.batch_normalization:
            self.bn = nn.BatchNorm2d(C_out)
        if self.activation:
            self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.batch_normalization:
            x = self.bn(x)
        if self.activation:
            x = self.act(x)

        return x

# X block
class XBlock(nn.Module):

    # initialization
    def __init__(self, C_in, C_out, stride, groups, bottleneck=1, downsample=False):

        # parent initialization
        super(XBlock, self).__init__()

        # add your code here
        # operations needed to create a parameterized XBlock
        self.downsample = downsample
        if self.downsample:
            self.downsample_layer = BatchNormActi(C_in, C_out,  size=1, stride=stride, padding=0, groups=1,      batch_normalization=True, activation=False)
        self.conv1 = BatchNormActi(C_in, C_out,                 size=1, stride=1,      padding=0, groups=1,      batch_normalization=True, activation=True)
        self.conv2 = BatchNormActi(C_out, C_out // bottleneck,  size=3, stride=stride, padding=1, groups=groups, batch_normalization=True, activation=True)
        self.conv3 = BatchNormActi(C_out // bottleneck, C_out,  size=1, stride=1,      padding=0, groups=1,      batch_normalization=True, activation=False)
        self.relu  = nn.ReLU(inplace=True) 

    # forward path
    def forward(self, x):

        # add your code here
        # tie together the operations to create a parameterized XBlock
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.conv3(y)

        if self.downsample:
            x = self.downsample_layer(x)

        y += x
        y = self.relu(y)

        # return
        return y

################################################################################
#
# NETWORK
#
################################################################################

# define
class Model(nn.Module):

    # initialization
    def __init__(self, data_num_channels, stem_width, per_stage_width, per_stage_depth, groups, num_classes):

        # parent initialization
        super(Model, self).__init__()

        # operations needed to create a modified RegNetX-200MF network
        # use the parameterized XBlock defined to simplify this section

        self.data_num_channels = data_num_channels
        self.stem_width        = stem_width
        self.per_stage_width   = per_stage_width
        self.per_stage_depth   = per_stage_depth
        self.groups            = groups

        # ENCODER
        # stem
        self.stem  = BatchNormActi(self.data_num_channels, self.stem_width,     size=3, stride=1, padding=1, groups=1, batch_normalization=True, activation=True)   # SIZE 56x56

        # encoder 0
        self.body_0 = self.createLayer(self.stem_width, self.per_stage_width[0],         stride=1, block_num=self.per_stage_depth[0], group_num=self.groups)         # SIZE 56x56

        # encoder 1
        self.body_1 = self.createLayer(self.per_stage_width[0], self.per_stage_width[1], stride=2, block_num=self.per_stage_depth[1], group_num=self.groups)         # SIZE 28x28

        # encoder 2
        self.body_2 = self.createLayer(self.per_stage_width[1], self.per_stage_width[2], stride=2, block_num=self.per_stage_depth[2], group_num=self.groups)         # SIZE 14x14

        # encoder 3
        self.body_3 = self.createLayer(self.per_stage_width[2], self.per_stage_width[3], stride=2, block_num=self.per_stage_depth[3], group_num=self.groups)         # SIZE 7x7

        #DECODER
        # head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.softmax = nn.Linear(self.per_stage_width[3], num_classes)


    def createLayer(self, C_in, C_out, stride, block_num, group_num):
        l = []
        for num in range(block_num):
            downsample = True if num == 0 and (
                stride != 1 or C_in != C_out) else False
            C_in = C_out if num > 0 else C_in
            stride = 1 if num > 0 else stride
            l.append(XBlock(C_in, C_out, stride=stride, groups=group_num, downsample=downsample))

        return nn.Sequential(*l)

    # forward path
    def forward(self, x):

        # add your code here
        # tie together the operations to create a modified RegNetX-200MF

        # stem
        x = self.stem(x)

        # body
        x = self.body_0(x)
        x = self.body_1(x)
        x = self.body_2(x)
        x = self.body_3(x)

        # head
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.softmax(x)

        y = x

        # return
        return y

# create
model = Model(DATA_NUM_CHANNELS, MODEL_STEM_WIDTH, MODEL_STAGE_WIDTH, MODEL_STAGE_DEPTH, MODEL_NUM_GROUPS, DATA_NUM_CLASSES)

# enable data parallelization for multi GPU systems
if (torch.cuda.device_count() > 1):
    model = nn.DataParallel(model)
print('Using {0:d} GPU(s)'.format(torch.cuda.device_count()), flush=True)

# specify the device as the GPU if present with fallback to the CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)

# transfer the network to the device
model.to(device)

################################################################################
#
# ERROR AND OPTIMIZER
#
################################################################################

# add your code here
# define the error criteria and optimizer

# start epoch
start_epoch = 0

# learning rate schedule
def lr_schedule(epoch):

    # linear warmup followed by cosine decay
    if epoch < TRAINING_LR_INIT_EPOCHS:
        lr = (TRAINING_LR_MAX - TRAINING_LR_INIT)*(float(epoch)/TRAINING_LR_INIT_EPOCHS) + TRAINING_LR_INIT
    else:
        lr = (TRAINING_LR_MAX - TRAINING_LR_FINAL)*max(0.0, math.cos(((float(epoch) - TRAINING_LR_INIT_EPOCHS)/(TRAINING_LR_FINAL_EPOCHS - 1.0))*(math.pi/2.0))) + TRAINING_LR_FINAL

    return lr

# error (softmax cross entropy)
criterion = nn.CrossEntropyLoss()

# optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

################################################################################
#
# TRAINING
#
################################################################################

# add your code here
# perform network training, validation and checkpoint saving
# see previous examples in the Code directory

training_data_loss    = []
testing_data_accuracy = []
epochs                = []

# model loading
if FILE_LOAD == 1:
    checkpoint = torch.load(FILE_NAME)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1


for epoch in range(start_epoch, TRAINING_NUM_EPOCHS):

    epochs.append(epoch + 1)
    start = time.time()

    # initialize train set statistics
    model.train()
    training_loss = 0.0
    num_batches   = 0

    # set the learning rate for the epoch
    for g in optimizer.param_groups:
        g['lr'] = lr_schedule(epoch)

    # cycle through the train set
    for data in dataloader_train:

        # extract a batch of data and move it to the appropriate device
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward pass, loss, backward pass and weight update
        outputs = model(inputs)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # update statistics
        training_loss = training_loss + loss.item()
        num_batches   = num_batches + 1
    
    model.eval()
    test_correct = 0
    test_total   = 0
    
    with torch.no_grad():
        for data in dataloader_test:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            test_total   = test_total + labels.size(0)
            test_correct = test_correct + (predicted == labels).sum().item()
    
    end = time.time()

    # checkpointing
    if FILE_SAVE == 1:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
    }, FILE_NAME)

    loss_ = (training_loss/num_batches)/DATA_BATCH_SIZE
    training_data_loss.append(training_loss)

    accu_ = (100.0*test_correct/test_total)
    testing_data_accuracy.append(accu_)

    print('Epoch {0:2d} lr = {1:8.6f} avg loss = {2:8.6f} accuracy = {3:5.2f} time = {4:5.2f}'.format(epoch + 1, lr_schedule(epoch), loss_, accu_, (end - start)))
        
################################################################################
#
# DISPLAY
#
################################################################################

plt.plot(training_data_loss, epochs, color='blue')
plt.xlabel('training_data_loss')
plt.ylabel('epochs')
plt.show()

plt.plot(testing_data_accuracy, epochs, color='blue')
plt.xlabel('testing_data_accuracy')
plt.ylabel('epochs')
plt.show()

from torchsummary import summary

print(summary(model, (3, 56, 56)))