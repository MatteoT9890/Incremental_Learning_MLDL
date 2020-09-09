from Incremental_Learning_MLDL.src.resnet import resnet32
from Incremental_Learning_MLDL.src.resnet_cosine import resnet32 as resnet32_cosine
import torch
from torch import nn
import torch.nn.functional as F

import copy
import torch.optim as optim
import math

class MyNet():
    def __init__(self,n_classes,type='normal'):
        self.type = type
        if(type == 'normal'):
          self.net = resnet32(num_classes=10)
          self.net.linear = nn.Linear(64,n_classes)
        elif type == 'cosine':
            self.net = resnet32_cosine(num_classes=10)
            self.net.linear = CosineLinear(64,n_classes)
        self.net.linear.weight = torch.nn.init.kaiming_normal_(self.net.linear.weight)
        self.init_weights = copy.deepcopy(self.net.linear.weight)
        self.batch_classes = 10
        self.prev_net = None
        self.balancing_net = None

    
    def update_network(self,best_net,n_classes,init_weights,type='not_cosine'):
        self.prev_net = copy.deepcopy(best_net)
        prev_weights = copy.deepcopy(best_net.linear.weight)
        if type == 'not_cosine':
            prev_bias = copy.deepcopy(best_net.linear.bias)
            self.net.linear = nn.Linear(64,n_classes)
            self.net.linear.weight.data[:n_classes-self.batch_classes] = prev_weights
            self.net.linear.weight.data[n_classes-self.batch_classes:n_classes] = init_weights
            self.net.linear.bias.data[:n_classes-self.batch_classes] = prev_bias
        else:
            prev_sigma = copy.deepcopy(self.net.linear.sigma)
            self.net.linear = CosineLinear(64,n_classes)
            self.net.linear.weight.data[:n_classes-self.batch_classes] = prev_weights
            self.net.linear.weight.data[n_classes-self.batch_classes:n_classes] = init_weights
            self.net.linear.sigma.data = prev_sigma
        return self.prev_net,self.net

    def get_old_outputs(self,images,labels,n_old_classes = None,type='not_cosine'):
        self.prev_net.train(False)
        if type == 'cosine':
            features,output = self.prev_net(images)
            return features[:],output
        elif type == 'rebalancing':
            self.balancing_net.train(False)
            output = self.balancing_net(images)
            return output[:,n_old_classes:]
        else:
            output = self.prev_net(images)
            return output
    
    def get_old_features_cosine(self,images,labels):
        self.prev_net.train(False)
        feature_map,_ = self.prev_net(images)
        return feature_map

    def prepare_training(self,LR,MOMENTUM,WEIGHT_DECAY,STEP_SIZE,GAMMA,typeScheduler,type='normal'):    
        parameters_to_optimize = self.net.parameters()
        if type == 'cosine':
            optimizer = optim.SGD(parameters_to_optimize,lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
        elif type == 'normal':
            optimizer = optim.SGD(parameters_to_optimize,lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
        if typeScheduler == 'multistep':
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, STEP_SIZE, gamma=GAMMA)
        else:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min')

        return (optimizer,scheduler)
    
    def freeze_conv(self):
        for i,child in enumerate(self.net.children()):
            if(i==5):
                break
            for param in child.parameters():
                param.requires_grad = False
        
        return self.net

    def freeze_neurons(self,n_old_classes):
        for param in self.net.linear.parameters():
            param.grad[:n_old_classes]=0
        
        return self.net

    def unfreeze_conv(self):
        for i,child in enumerate(self.net.children()):
            if(i==5):
                break
            for param in child.parameters():
                param.requires_grad = True
        
        return self.net
    
class CosineLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma=True):
        super(CosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if sigma:
            self.sigma = nn.Parameter(torch.Tensor(1))
        else:
            self.register_parameter('sigma', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.sigma is not None:
            self.sigma.data.fill_(1) #for initializaiton of sigma

    def forward(self, input):

        out = F.linear(F.normalize(input, p=2,dim=1), \
                F.normalize(self.weight, p=2, dim=1))
        if self.sigma is not None:
            out = self.sigma * out
        return out