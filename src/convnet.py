import logging
import torch
import torch.nn as nn
from collections import OrderedDict
from tdcnn import _TDCNN
from torch.hub import load_state_dict_from_url


def make_layers(block, no_relu_layers):
    layers = []
    for layer_name, v in block.items():
        if 'pool' in layer_name:
            layer = nn.MaxPool2d(kernel_size=v[0], stride=v[1],
                                    padding=v[2])
            layers.append((layer_name, layer))
        else:
            conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1],
                               kernel_size=v[2], stride=v[3],
                               padding=v[4])
            layers.append((layer_name, conv2d))
            if layer_name not in no_relu_layers:
                layers.append(('relu_'+layer_name, nn.ReLU(inplace=True)))

    return nn.Sequential(OrderedDict(layers))


class ConvNet(_TDCNN):
    def __init__(self, char_list, num_class=2, ctc_type='warp', rd_iou_thr=0.5, num_concat=1):
        feat_dim = 512
        fsr_hidden_dim = 512
        anchor_scales = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 8, 12, 16, 24, 32, 40]
        neg_thr = 0.3
        pos_thr = 0.6
        super().__init__(num_class, feat_dim, fsr_hidden_dim, anchor_scales, neg_thr, pos_thr, char_list, ctc_type=ctc_type, rd_iou_thr=rd_iou_thr, num_concat=num_concat)
        no_relu_layers = []
        block1_0 = OrderedDict([
                ('conv1_1', [3, 64, 3, 1, 1]),
                ('conv1_2', [64, 64, 3, 1, 1]),
                ('pool1_stage1', [2, 2, 0]),
                ('conv2_1', [64, 128, 3, 1, 1]),
                ('conv2_2', [128, 128, 3, 1, 1]),
                ('pool2_stage1', [2, 2, 0]),
                ('conv3_1', [128, 256, 3, 1, 1]),
                ('conv3_2', [256, 256, 3, 1, 1]),
                ('conv3_3', [256, 256, 3, 1, 1]),
                ('conv3_4', [256, 256, 3, 1, 1]),
                ('pool3_stage1', [2, 2, 0]),
                ('conv4_1', [256, 512, 3, 1, 1]),
                ('conv4_2', [512, 512, 3, 1, 1]),
                ('conv4_3', [512, 512, 3, 1, 1]),
                ('conv4_4', [512, 512, 3, 1, 1]),
                ('conv5_1', [512, 512, 3, 1, 1]),
                ('conv5_2', [512, 512, 3, 1, 1])
            ])
        self.features = make_layers(block1_0, no_relu_layers)


def convNet(**kwargs):
    vgg19_url = 'http://download.pytorch.org/models/vgg19-dcbb9e9d.pth'
    model = ConvNet(**kwargs)
    std_vgg19 = load_state_dict_from_url(vgg19_url)
    keys_vgg19 = list(std_vgg19.keys())
    keys_mdl = list(model.state_dict().keys())
    key_index = 0
    for key_mdl in keys_mdl:
        if key_mdl.split('.')[0] == 'features':
            key_vgg19 = keys_vgg19[key_index]
            logging.info(f"{key_vgg19} -> {key_mdl}")
            model.state_dict()[key_mdl].copy_(std_vgg19[key_vgg19])
            key_index += 1
    return model
