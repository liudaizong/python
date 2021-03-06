import os
from PIL import Image
# third-party library
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
from torchvision import transforms, utils
import matplotlib.pyplot as plt

torch.manual_seed(1)    # reproducible

# Hyper Parameters
EPOCH = 6              # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 2
LR = 0.001              # learning rate



#mydataset
train_data = torchvision.datasets.ImageFolder(
        'digital_number/train',
        transform=torchvision.transforms.ToTensor(),                                        #transform=transforms.Compose(transforms.ToTensor(),)
)

# plot one example
#print(train_data.train_data.size())                 # (60000, 28, 28)
#print(train_data.train_labels.size())               # (60000)
#plt.imshow(train_data.train_data[0].numpy(), cmap='gray')
#plt.title('%i' % train_data.train_labels[0])
#plt.show()

# Data Loader for easy mini-batch return in training, the image batch shape will be (50, 1, 28, 28)
train_loader = Data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

# convert test data into Variable, pick 2000 samples to speed up testing

test_data = torchvision.datasets.ImageFolder(
        #'digital_number/test2',
       'digital_number/test1',
      transform=torchvision.transforms.ToTensor(),
      )
test_loader = Data.DataLoader(test_data, batch_size=6, shuffle=True)
for step, (x, y) in enumerate(test_loader):
    b_x = Variable(x)   # batch x
    b_y = y
    test_x =  b_x
print(test_x.size())
test_y = np.array([4,3,0,0,7,1])
#test_y = np.array([4,3,7,8,1,6])
test_y = torch.from_numpy(test_y)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=3,              # input height
                out_channels=16,            # n_filters
                kernel_size=5,              # filter size
                stride=1,                   # filter movement/step
                padding=2,                  # if want same width and length of this image after con2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (16, 28, 28)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (16, 14, 14)
        )
        self.conv2 = nn.Sequential(         # input shape (1, 14, 14)
            nn.Conv2d(16, 32, 5, 1, 2),     # output shape (32, 14, 14)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(2),                # output shape (32, 7, 7)
        )
        self.out = nn.Linear(32 * 7 * 7, 10)   # fully connected layer, output 10 classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)           # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        output = self.out(x)
        return output, x    # return x for visualization


cnn = CNN()
print(cnn)  # net architecture

optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()                       # the target label is not one-hotted

# training
for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):   # gives batch data, normalize x when iterate train_loader
        b_x = Variable(x)   # batch x
        b_y = Variable(y)   # batch y
        
        output = cnn(b_x)[0]               # cnn output
        loss = loss_func(output, b_y)   # cross entropy loss
        optimizer.zero_grad()           # clear gradients for this training step
        loss.backward()                 # backpropagation, compute gradients
        optimizer.step()                # apply gradients
        
        if step % 50 == 0:
            test_output, last_layer = cnn(test_x)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            accuracy = sum(pred_y == test_y) / float(test_y.size(0))
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data[0], '| test accuracy: %.2f' % accuracy)

# testing
test_output, _ = cnn(test_x)
pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
print(pred_y, 'prediction number')
print(test_y.numpy(), 'real number')

