import time
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class PoseEstimator(nn.Module):
    def __init__(self, in_channel):
        super(PoseEstimator, self).__init__()
        num_body_kps, num_hand_kps = 25, 42
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channel, 128, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=num_body_kps+num_hand_kps, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def forward(self, feat_map, label_map, label_mask, sample_rate):
        # feat_map: [B, F, h, w], label_map: [B, N, h, w], label_mask: [B, N]
        pred = self.conv(feat_map[::sample_rate])
        loss = get_pose_loss(pred, label_map, label_mask)
        return loss


def get_pose_loss(pred, label, label_mask):
    # pred: [B, N, h, w], label: [B, N, h, w], label_mask: [B, N]
    label_mask_ = label_mask.unsqueeze(dim=-1).unsqueeze(dim=-1)
    loss = ((pred - label).pow(2) * label_mask_)
    loss = loss.view(loss.size(0), loss.size(1), -1)
    loss = loss.mean(dim=-1).sum(dim=-1).mean()
    return loss
