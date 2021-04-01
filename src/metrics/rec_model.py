import torch
import torch.nn as nn
import torch.nn.functional as functional
from collections import OrderedDict

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


class handpose_model(nn.Module):
    def __init__(self):
        super(handpose_model, self).__init__()

        # these layers have no relu layer
        no_relu_layers = ['conv6_2_CPM', 'Mconv7_stage2', 'Mconv7_stage3',\
                          'Mconv7_stage4', 'Mconv7_stage5', 'Mconv7_stage6']
        # stage 1
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
                ('conv5_2', [512, 512, 3, 1, 1]),
                ('conv5_3_CPM', [512, 128, 3, 1, 1])
            ])

        block1_1 = OrderedDict([
            ('conv6_1_CPM', [128, 512, 1, 1, 0]),
            ('conv6_2_CPM', [512, 22, 1, 1, 0])
        ])

        blocks = {}
        blocks['block1_0'] = block1_0
        blocks['block1_1'] = block1_1

        for k in blocks.keys():
            blocks[k] = make_layers(blocks[k], no_relu_layers)

        self.model1_0 = blocks['block1_0']
        self.model1_1 = blocks['block1_1']

    def forward(self, x):
        out1_0 = self.model1_0(x)
        return out1_0

class ConvLstm(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers, bid=False):
        super(ConvLstm, self).__init__()
        self.backbone = handpose_model()
        input_size = 128
        self.lstm = nn.LSTM(input_size, hidden_size, n_layers, bidirectional=bid)
        self.lt = nn.Linear(hidden_size*(bid+1), output_size)
        self.classifier = nn.LogSoftmax(dim=-1)
        self.n_layers, self.nhid, self.nin, self.nout, self.num_direction = n_layers, hidden_size, input_size, output_size, int(bid)+1

    def forward(self, Xs):
        """
        Xs: (B, L, C, H, W)
        output: (L, B, V), (L, B, V)
        """
        h0 = self.init_hidden(len(Xs))
        xsz = list(Xs.size())
        B, L = xsz[0], xsz[1]
        Xs = Xs.view(*([-1] + xsz[2:]))
        Fs = self.backbone.model1_0(Xs)
        Fs = functional.adaptive_avg_pool2d(Fs, (1, 1)).view(B*L, -1)
        Fs = Fs.view(B, L, -1).transpose(0, 1) # [B, L, F]
        output, ht = self.lstm(Fs, h0) # [L, B, H]
        logits = self.lt(output)
        log_probs = self.classifier(logits)
        return logits, log_probs

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(self.num_direction*self.n_layers, bsz, self.nhid),
                weight.new_zeros(self.num_direction*self.n_layers, bsz, self.nhid))
