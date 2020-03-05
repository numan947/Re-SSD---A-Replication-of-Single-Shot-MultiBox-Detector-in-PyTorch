# TORCH
import torch
import torchvision.transforms.functional as FT
# GENERAL
import os
import json
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


voc_labels = ["background",
              "aeroplane", "bicycle", "bird", "boat", "bottle","bus", "car","cat","chair","cow","diningtable",
              "dog","horse","motorbike","person","pottedplant","sheep","sofa","train", "tvmonitor"
              ]

label_map = {k:v for v,k in enumerate(voc_labels)}
rev_label_map = {v:k for k,v in label_map.items()}

distinct_colors = ["#FFFFFF",
                   '#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', 
                   '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabebe',
                   '#469990', '#e6beff', '#9A6324', '#fffac8', '#800000',
                   '#aaffc3', '#808000', '#ffd8b1', '#000075', '#a9a9a9']
label_color_map = {k:distinct_colors[i] for i,k in enumerate(voc_labels)}


def find_intersection(set_1, set_2):
    lower_bounds = torch.max(set_1[:,:2].unsqueeze(1), set_2[:,:2].unsqueeze(0))
    upper_bounds = torch.min(set_1[:,2:].unsqueeze(1), set_2[:,2:].unsqueeze(0))
    intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0) # find widths and heights
    return intersection_dims[:,:, 0] * intersection_dims[:,:, 1] # widths * heights

def find_jaccard_overlap(set_1, set_2):
    intersection = find_intersection(set_1, set_2)
    
    set_1_A = (set_1[:,2] - set_1[:,0])*(set_1[:,3] - set_1[:,1]) # n1
    set_2_A = (set_2[:,2] - set_2[:,0])*(set_2[:,3] - set_2[:,1]) # n2
    
    union = set_1_A.unsqueeze(1) + set_2_A.unsqueeze(0) - intersection # (n1, n2)
    
    return intersection/union
