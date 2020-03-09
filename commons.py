# TORCH
import torch
import torchvision.transforms.functional as FT
# GENERAL
import os
import json
import random
from math import sqrt

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


def get_priors_cxcy():
    fmap_dims = {
        "conv43":38,
        'conv7':19,
        'conv82':10,
        'conv92':5,
        'conv102':3,
        'conv112':1
    }
    obj_scales = {
        'conv43':0.1,
        'conv7':0.2,
        'conv82':0.375,
        'conv92':0.55,
        'conv102':0.725,
        'conv112':0.9
    }

    aspect_ratios = {
        'conv43': [1.0, 2.0, 0.5],
        'conv7': [1.0, 2.0, 3.0, 0.5, 0.333],
        'conv82':[1.0, 2.0, 3.0, 0.5, 0.333],
        'conv92':[1.0, 2.0, 3.0, 0.5, 0.333],
        'conv102': [1.0, 2.0, 0.5],
        'conv112':[1.0, 2.0, 0.5]
    }

    fmaps = list(fmap_dims.keys())

    prior_boxes = []

    for k, fmap in enumerate(fmaps):
        for i in range(fmap_dims[fmap]):
            for j in range(fmap_dims[fmap]):
                cx = (i+0.5)/fmap_dims[fmap]
                cy = (j+0.5)/fmap_dims[fmap]

                for ratio in aspect_ratios[fmap]:
                    prior_boxes.append([cx, cy, obj_scales[fmap]*sqrt(ratio), obj_scales[fmap]/sqrt(ratio)])

                    if ratio == 1.0:
                        try:
                            additional_scale = sqrt(obj_scales[fmap]*obj_scales[fmaps[k+1]])
                        except IndexError:
                            additional_scale = 1.0

                        prior_boxes.append([cx, cy, additional_scale, additional_scale])

    prior_boxes = torch.FloatTensor(prior_boxes).to(device)
    prior_boxes.clamp_(0.0, 1.0)

    return prior_boxes


def cxcy_to_xy(cxcy):
    return torch.cat([
        cxcy[:,:2] - (cxcy[:,2:]/2.0),
        cxcy[:,:2] + (cxcy[:,2:]/2.0)]
        ,dim=1)

def xy_to_cxcy(xy):
    return torch.cat([
        (xy[:,:2] + xy[:,2:])/2.0,
        (xy[:,2:] - xy[:,:2])]
        ,dim=1)

def cxcy_to_gcxgcy(cxcy, priors_cxcy):
    return torch.cat([
        (cxcy[:,:2]-priors_cxcy[:,:2])/(priors_cxcy[:,2:]/10.0),
        torch.log(cxcy[:,2:]/priors_cxcy[:,2:]) * 5.0]
        ,dim=1
    )

def gcxgcy_to_cxcy(gcxgcy, priors_cxcy):
    return torch.cat([
        (gcxgcy[:,:2] * (priors_cxcy[:,2:]/10.0))+priors_cxcy[:,:2],
        torch.exp(gcxgcy[:,2:]/5.0) * priors_cxcy[:,2:]]
        ,dim=1
        )