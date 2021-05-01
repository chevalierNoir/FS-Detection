import logging
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utils import get_scale_range
from r3d_utils import generate_proposal, generate_label_rpn, log_loss, smooth_l1_loss

class RPN(nn.Module):
    """ region proposal network """
    def __init__(self, din, anchor_scales, feat_stride, neg_thr, pos_thr, nms_top_n, nms_thresh, min_size, out_scores=False, device=torch.device('cuda')):
        super(RPN, self).__init__()
        self.din = din
        self.anchor_scales = np.array(anchor_scales)
        self.feat_stride = feat_stride
        self.out_scores = out_scores
        self.neg_thr = neg_thr
        self.pos_thr = pos_thr
        self.nms_top_n = nms_top_n
        self.nms_thresh = nms_thresh
        self.min_size = min_size
        self.mask_upsample_rate = 1
        self.device = device

        logging.info('neg thr: %.3f, pos thr: %.3f' % (neg_thr, pos_thr))
        self.feature = nn.Sequential(
            nn.Conv3d(self.din, 512, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1), bias=True),
            nn.Conv3d(512, 512, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1), bias=True),
            nn.AdaptiveMaxPool3d(output_size=(None, 1, 1)))
            # nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)))

        self.anchor_scorer = nn.Conv3d(512, len(self.anchor_scales) * 2, 1, 1, 0)
        self.delta_predictor = nn.Conv3d(512, len(self.anchor_scales) * 2, 1, 1, 0)

        self.anchor_scales = get_scale_range(base_size=feat_stride, scales=self.anchor_scales).to(self.device)

    def forward(self, base_feat, gt_chunks, video_len, training=True):
        # base_feat: [B, F, L, H, W], labels: [B, N, 3]
        batch_size, num_anchors = base_feat.size(0), len(self.anchor_scales)
        feat_vol = self.feature(base_feat)
        feat_len = feat_vol.size(2)
        prob_vol = self.anchor_scorer(feat_vol)
        prob_vol = prob_vol.squeeze(dim=-1).squeeze(dim=-1).view(batch_size, 2, num_anchors, feat_len) # [B, 2, A, L]
        prob_vol = F.softmax(prob_vol, dim=1)
        delta_vol = self.delta_predictor(feat_vol)
        delta_vol = delta_vol.squeeze(dim=-1).squeeze(dim=-1).view(batch_size, 2, num_anchors, feat_len)  # [B, 2, A, L]
        prop_coord, prop_score = generate_proposal(self.anchor_scales, prob_vol, delta_vol, self.feat_stride, self.nms_top_n, self.nms_thresh)

        if training:
            assert gt_chunks is not None
            cls_label, reg_label = generate_label_rpn(self.anchor_scales, self.feat_stride, prob_vol.size(-1), video_len, gt_chunks, high_iou_thr=self.pos_thr, low_iou_thr=self.neg_thr)
            prob_vol, delta_vol = prob_vol.view(batch_size, 2, -1).transpose(1, 2).contiguous(), delta_vol.view(batch_size, 2, -1).transpose(1, 2).contiguous()
            cls_loss = log_loss(prob_vol, cls_label)
            reg_loss = smooth_l1_loss(delta_vol, reg_label, cls_label)
        else:
            cls_loss, reg_loss = 0, 0
        return prop_coord, prop_score, cls_loss, reg_loss
