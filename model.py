import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
import torchvision
from commons import *


class VGGBase(nn.Module):
    def __init__(self):
        super(VGGBase, self).__init__()
        
        self.conv11 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv12 = nn.Conv2d(3, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv21 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv22 = nn.Conv2d(128, 128, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv31 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv32 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv33 = nn.Conv2d(256, 256, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2, ceil_mode=True)
        
        self.conv41 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv42 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv43 = nn.Conv2d(512, 512, 3, padding=1)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        self.conv51 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv52 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv53 = nn.Conv2d(512, 512, 3, padding=1)
        self.pool5 = nn.MaxPool2d(3, 1, padding=1)
        
        
        self.conv6 = nn.Conv2d(512, 1024, 3, padding=6, dilation=6)
        self.conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
        
        
        self.load_pretrained_layers()
        
    
    def forward(self, image):
        out = torch.relu(self.conv11(image))
        out = torch.relu(self.conv12(out))
        out = self.pool1(out))
        
        out = torch.relu(self.conv21(out))
        out = torch.relu(self.conv22(out))
        out = self.pool2(out))
        
        out = torch.relu(self.conv31(out))
        out = torch.relu(self.conv32(out))
        out = torch.relu(self.conv33(out))
        out = self.pool3(out))
        
        out = torch.relu(self.conv41(out))
        out = torch.relu(self.conv42(out))
        out = torch.relu(self.conv43(out))
        out = self.pool4(out))
        
        conv43_feats = out
        
        out = torch.relu(self.conv51(out))
        out = torch.relu(self.conv52(out))
        out = torch.relu(self.conv53(out))
        out = self.pool5(out))   
        
        out = torch.relu(self.conv6(out))
        out = torch.relu(self.conv7(out))
        
        
        conv7_feats = out
        
        
        return conv43_feats, conv7_feats
        
    
    def load_pretrained_layers(self):
        state_dict = self.state_dict()
        param_names = list(state_dict.keys())
        
        pretrained_state_dict = torchvision.models.vgg16(pretrained=True).state_dict()
        pretrained_param_names = list(pretrained_state_dict.keys())
        
        for i, param in enumerate(param_names[:-4]):
            state_dict[param] = pretrained_state_dict[pretrained_param_names[i]]
        
        conv_fc6_weights = pretrained_state_dict['classifier.0.weight'].view(4096, 512, 7, 7)
        conv_fc6_biases = pretrained_state_dict['classifier.0.bias']
        state_dict['conv6.weight'] = decimate(conv_fc6_weights, m=[4, None, 3, 3])
        state_dict['conv6.bias'] = decimate(conv_fc6_biases,m=[4])

        conv_fc7_weights = pretrained_state_dict['classifier.3.weight'].view(4096, 512, 7, 7)
        conv_fc7_biases = pretrained_state_dict['classifier.3.bias']
        state_dict['conv7.weight'] = decimate(conv_fc7_weights, m=[4, 4, None, None])
        state_dict['conv7.bias'] = decimate(conv_fc7_biases,m=[4])
        
        
        self.load_state_dict(state_dict)
        
        print("Pretrained Model Loaded!")


class AuxConvs(nn.Module):
    def __init__(self):
        super(AuxConvs, self).__init__()
        
        self.conv81 = nn.Conv2d(1024, 256, 1, padding=0)
        self.conv82 = nn.Conv2d(256, 512, 3, 2, padding=1)
        
        self.conv91 = nn.Conv2d(512, 128, 1, padding=0)
        self.conv92 = nn.Conv2d(128, 256, 3, 2, padding=1)
        
        self.conv101 = nn.Conv2d(256, 128, 1, padding=0)
        self.conv102 = nn.Conv2d(128, 256, 3, padding=0)
        
        self.conv111 = nn.Conv2d(256, 128, 1, padding=0)
        self.conv112 = nn.Conv2d(128, 256, 1, padding=0)
        
        self.init_conv2d()
    
    def init_conv2d(self):
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_normal_(c.weight)
                c.bias.data.fill_(0.01)
    
    def forward(self, conv7_feats):
        out = torch.relu(self.conv81(conv7_feats))
        out = torch.relu(self.conv82(out))
        conv82_feats = out

        out = torch.relu(self.conv91(out))
        out = torch.relu(self.conv92(out))
        conv92_feats = out
        
        out = torch.relu(self.conv101(out))
        out = torch.relu(self.conv102(out))
        conv102_feats = out
        
        
        out = torch.relu(self.conv111(out))
        out = torch.relu(self.conv112(out))
        conv112_feats = out
        
        return conv82_feats, conv92_feats, conv102_feats, conv112_feats
