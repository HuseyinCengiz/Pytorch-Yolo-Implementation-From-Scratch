from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2


def predict_transform(prediction, inp_dim, anchors, num_classes, CUDA = True):
    batch_size = prediction.size(0)
    stride = inp_dim // prediction.size(2)
    grid_size = inp_dim // stride
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)

    prediction = prediction.view(batch_size, bbox_attrs*num_anchors, grid_size*grid_size) #(Number of Images, (85x3), 13x13) ---> (Number of Images, 255, 169)
    prediction = prediction.transpose(1,2).contiguous() #(Number of Images, 13x13, (85x3)) -----> (Number of Images, 169, 255)
    prediction = prediction.view(batch_size, grid_size*grid_size*num_anchors, bbox_attrs)  #(Number of Images, (13x13x3), 85) ----> (Number of Images, 507, 85)

    anchors = [(a[0]/stride,a[1]/stride) for a in anchors]

    #Sigmoid the Centre_X, Centre_Y and Object confidence

    prediction[:,:,0] = torch.sigmoid(prediction[:,:,0]) # Centre_X
    prediction[:,:,1] = torch.sigmoid(prediction[:,:,1]) # Centre_Y
    prediction[:,:,4] = torch.sigmoid(prediction[:,:,4]) # Object confidence

    ######## I couldn't understand this part ########
    
    grid = np.arange(grid_size)
    a, b = np.meshgrid(grid, grid)

    x_offset = torch.FloatTensor(a).view(-1,1)
    y_offset = torch.FloatTensor(b).view(-1,1)

    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()
    
    x_y_offset = torch.cat((x_offset,y_offset),1).repeat(1,num_anchors).view(-1, 2).unsqueeze(0)
    prediction[:,:,:2] += x_y_offset
    anchors = torch.FloatTensor(anchors)
    
    if CUDA:
        anchors = anchors.cuda()

    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
    prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4]) * anchors

    ########################################################

    #Apply sigmoid activation to the class scores
    prediction[:,:,5: 5+num_classes] = torch.sigmoid((prediction[:,:,5: 5+num_classes]))
    prediction[:,:,:4] *= stride

    return prediction






