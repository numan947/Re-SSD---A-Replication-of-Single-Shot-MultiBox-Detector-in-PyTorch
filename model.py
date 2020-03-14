import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
import torchvision
from commons import *


def decimate(tensor, m):
    assert tensor.dim() == len(m)

    for d in range(tensor.dim()):
        if m[d] is not None:
            tensor = tensor.index_select(dim=d, index=torch.arange(start=0, end=tensor.size(d), step=m[d]).long())
    return tensor

class VGGBase(nn.Module):
    def __init__(self):
        super(VGGBase, self).__init__()

        self.conv11 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv12 = nn.Conv2d(64, 64, 3, padding=1)
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
        out = self.pool1(out)

        out = torch.relu(self.conv21(out))
        out = torch.relu(self.conv22(out))
        out = self.pool2(out)

        out = torch.relu(self.conv31(out))
        out = torch.relu(self.conv32(out))
        out = torch.relu(self.conv33(out))
        out = self.pool3(out)

        out = torch.relu(self.conv41(out))
        out = torch.relu(self.conv42(out))
        out = torch.relu(self.conv43(out))
        conv43_feats = out
        out = self.pool4(out)

        out = torch.relu(self.conv51(out))
        out = torch.relu(self.conv52(out))
        out = torch.relu(self.conv53(out))
        out = self.pool5(out)

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

        conv_fc7_weights = pretrained_state_dict['classifier.3.weight'].view(4096, 4096, 1, 1)
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
        self.conv112 = nn.Conv2d(128, 256, 3, padding=0)

        self.init_conv2d()

    def init_conv2d(self):
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.kaiming_normal_(c.weight, nonlinearity='relu')
                c.bias.data.fill_(0.0001)

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

class PredictionConvs(nn.Module):
    def __init__(self, n_classes):
        super(PredictionConvs, self).__init__()

        self.n_classes = n_classes

        n_boxes = {
            'conv43':4,
            'conv7':6,
            'conv82':6,
            'conv92':6,
            'conv102':4,
            'conv112':4
        }

        self.lconv43 = nn.Conv2d(512, n_boxes['conv43']*4, 3, padding=1)
        self.lconv7 = nn.Conv2d(1024, n_boxes['conv7']*4, 3, padding=1)
        self.lconv82 = nn.Conv2d(512, n_boxes['conv82']*4, 3, padding=1)
        self.lconv92 = nn.Conv2d(256, n_boxes['conv92']*4, 3, padding=1)
        self.lconv102 = nn.Conv2d(256, n_boxes['conv102']*4, 3, padding=1)
        self.lconv112 = nn.Conv2d(256, n_boxes['conv112']*4, 3, padding=1)

        self.cconv43 = nn.Conv2d(512, n_boxes['conv43']*n_classes, 3, padding=1)
        self.cconv7 = nn.Conv2d(1024, n_boxes['conv7']*n_classes, 3, padding=1)
        self.cconv82 = nn.Conv2d(512, n_boxes['conv82']*n_classes, 3, padding=1)
        self.cconv92 = nn.Conv2d(256, n_boxes['conv92']*n_classes, 3, padding=1)
        self.cconv102 = nn.Conv2d(256, n_boxes['conv102']*n_classes, 3, padding=1)
        self.cconv112 = nn.Conv2d(256, n_boxes['conv112']*n_classes, 3, padding=1)

        self.init_conv2d()

    def init_conv2d(self):
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.kaiming_normal_(c.weight, nonlinearity='relu')
                c.bias.data.fill_(0.0001)

    def forward(self, conv43_feats, conv7_feats, conv82_feats, conv92_feats, conv102_feats, conv112_feats):
        batch_size = conv43_feats.size(0)

        loc43 = self.lconv43(conv43_feats).permute(0,2,3,1).contiguous().view(batch_size, -1, 4)
        loc7 = self.lconv7(conv7_feats).permute(0,2,3,1).contiguous().view(batch_size, -1, 4)
        loc82 = self.lconv82(conv82_feats).permute(0,2,3,1).contiguous().view(batch_size, -1, 4)
        loc92 = self.lconv92(conv92_feats).permute(0,2,3,1).contiguous().view(batch_size, -1, 4)
        loc102 = self.lconv102(conv102_feats).permute(0,2,3,1).contiguous().view(batch_size, -1, 4)
        loc112 = self.lconv112(conv112_feats).permute(0,2,3,1).contiguous().view(batch_size, -1, 4)

        pred43 = self.cconv43(conv43_feats).permute(0,2,3,1).contiguous().view(batch_size, -1, self.n_classes)
        pred7 = self.cconv7(conv7_feats).permute(0,2,3,1).contiguous().view(batch_size, -1, self.n_classes)
        pred82 = self.cconv82(conv82_feats).permute(0,2,3,1).contiguous().view(batch_size, -1, self.n_classes)
        pred92 = self.cconv92(conv92_feats).permute(0,2,3,1).contiguous().view(batch_size, -1, self.n_classes)
        pred102 = self.cconv102(conv102_feats).permute(0,2,3,1).contiguous().view(batch_size, -1, self.n_classes)
        pred112 = self.cconv112(conv112_feats).permute(0,2,3,1).contiguous().view(batch_size, -1, self.n_classes)

        # print(loc43.shape)
        # print(loc7.shape)
        # print(loc82.shape)
        # print(loc92.shape)
        # print(loc102.shape)
        # print(loc112.shape)

        locs = torch.cat([loc43, loc7, loc82, loc92, loc102, loc112], dim=1)
        preds = torch.cat([pred43, pred7, pred82, pred92, pred102, pred112], dim=1)

        # print(locs.shape)
        # print(preds.shape)

        return locs, preds


class SSD300(nn.Module):

    def __init__(self, n_classes):

        super(SSD300, self).__init__()

        self.n_classes = n_classes
        self.base = VGGBase()
        self.aux = AuxConvs()
        self.pred = PredictionConvs(self.n_classes)
        self.rescale_factors = nn.Parameter(torch.FloatTensor(1, 512, 1, 1))
        nn.init.constant_(self.rescale_factors, 20)

        self.priors_cxcy = get_priors_cxcy()

    def forward(self, image):

        conv43_feats, conv7_feats = self.base(image)
        norm = conv43_feats.norm(dim=1, keepdim=True)+1e-16
        conv43_feats = conv43_feats/norm
        conv43_feats = conv43_feats*self.rescale_factors

        conv82_feats, conv92_feats, conv102_feats, conv112_feats = self.aux(conv7_feats)

        locs, preds = self.pred(conv43_feats, conv7_feats, conv82_feats, conv92_feats, conv102_feats, conv112_feats)

        return locs, preds


class MultiBoxLoss(nn.Module):
    def __init__(self, priors_cxcy, overlap_threshold=0.5, neg_pos_ratio=3, alpha=1.0):
        super(MultiBoxLoss, self).__init__()

        self.priors_cxcy = priors_cxcy
        self.priors_xy = cxcy_to_xy(priors_cxcy)
        self.overlap_threshold = overlap_threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.alpha = alpha

        self.smooth_l1 = nn.SmoothL1Loss()
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')


    def forward(self, predicted_locs, predicted_scores, boxes, labels):
        batch_size = predicted_locs.size(0)
        n_priors = self.priors_cxcy.size(0)
        n_classes = predicted_scores.size(2)


        assert n_priors == predicted_scores.size(1) == predicted_locs.size(1)


        true_locs = torch.zeros((batch_size, n_priors, 4), dtype=torch.float).to(device)
        true_classes = torch.zeros((batch_size, n_priors), dtype=torch.long).to(device)


        for i in range(batch_size):
            n_objects = boxes[i].size(0)

            overlap = find_jaccard_overlap(boxes[i], self.priors_xy)

            overlap_for_each_prior, object_for_each_prior = overlap.max(dim=0)

            _, prior_for_each_object = overlap.max(dim=1)

            object_for_each_prior[prior_for_each_object] = torch.LongTensor(range(n_objects)).to(device)
            overlap_for_each_prior[prior_for_each_object] = 1.0

            labels_for_each_prior = labels[i][object_for_each_prior]
            labels_for_each_prior[overlap_for_each_prior<self.overlap_threshold] = 0.0

            true_classes[i] = labels_for_each_prior
            true_locs[i] = cxcy_to_gcxgcy(xy_to_cxcy(boxes[i][object_for_each_prior]), self.priors_cxcy)


        positive_priors = true_classes!=0

        loc_loss = self.smooth_l1(predicted_locs[positive_priors], true_locs[positive_priors])

        n_positives = positive_priors.sum(dim=1)
        n_hard_negatives = self.neg_pos_ratio*n_positives

        conf_loss_all = self.cross_entropy(predicted_scores.view(-1, n_classes), true_classes.view(-1)).view(batch_size, n_priors)
        conf_loss_pos = conf_loss_all[positive_priors]

        conf_loss_neg = conf_loss_all.clone()
        conf_loss_neg[positive_priors] = 0.0
        conf_loss_neg,_ = conf_loss_neg.sort(dim=1, descending=True)

        hard_negative_mask = torch.LongTensor(range(n_priors)).unsqueeze(0).expand_as(conf_loss_neg).to(device)
        hard_negative_mask = hard_negative_mask<n_hard_negatives.unsqueeze(dim=1)

        conf_loss_hard_neg = conf_loss_neg[hard_negative_mask]


        conf_loss = (conf_loss_hard_neg.sum() + conf_loss_pos.sum())/n_positives.sum().float()

        return conf_loss+self.alpha*loc_loss

