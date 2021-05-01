import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from rpn import RPN
import time
from utils import smooth_l1_loss
from twin_transform import twin_transform_inv
from r3d_utils import roi_pooling, generate_label_roi, log_loss, smooth_l1_loss
from fsr import FSR
from pose_estimate import PoseEstimator


class _TDCNN(nn.Module):
    def __init__(self, num_class, feat_dim, fsr_hidden_dim, anchor_scales,
                 neg_thr, pos_thr, char_list, ctc_type, rd_iou_thr, 
                 rcnn_pooling_length=4, rcnn_pooling_width=1, 
                 rcnn_pooling_height=1, temporal_scale=1, 
                 nms_top_n=50, nms_thresh=0.7, min_size=8, num_concat=1):
        super(_TDCNN, self).__init__()
        self.rcnn_pooling_length, self.rcnn_pooling_width, self.rcnn_pooling_height = rcnn_pooling_length, rcnn_pooling_width, rcnn_pooling_height
        self.num_class = num_class
        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_twin = 0
        print('Number of class: %d' % (self.num_class))
        # define rpn
        self.features = None
        self.rpn_only = True
        self.feat_stride = 8
        rd_iou_thr, rd_n_top = rd_iou_thr, nms_top_n
        self.RCNN_3d = nn.Sequential(
            nn.Conv3d(feat_dim, feat_dim//num_concat, kernel_size=(8, 1, 1), stride=(1, 1, 1), padding=(4, 0, 0)),
            nn.MaxPool3d(kernel_size=(8, 1, 1), padding=(4, 0, 0)))
        self.RCNN_rpn = RPN(feat_dim, anchor_scales, self.feat_stride, neg_thr, pos_thr, nms_top_n, nms_thresh, min_size)
        self.fsr = FSR(fsr_hidden_dim, feat_dim, n_layers=1, char_list=char_list, ctc_type=ctc_type, iou_thr=rd_iou_thr, n_top=rd_n_top)
        self.RCNN_cls_score, self.RCNN_twin_pred = None, None
        self.pose_estimate = PoseEstimator(feat_dim)

    def forward(self, sample, training=True, pose_on=False, fsr_on=False, pose_sample_rate=1, reward_on=True, stage=1):
        if stage == 1:
            return self.fwd_s1(sample, training, pose_on, fsr_on, pose_sample_rate, reward_on)
        elif stage == 2:
            return self.fwd_s2(sample, training, pose_on, fsr_on, pose_sample_rate, reward_on)
        else:
            raise ValueError(f'Stage: {1, 2}')

    def fwd_s1(self, sample, training=True, pose_on=False, fsr_on=False, pose_sample_rate=1, reward_on=True):
        # retrieve data
        video_data, opt_data, det_label, fs_label, fs_mask = sample['img_global'], sample['opt_global'], sample['det_label'], sample['fs_label'], sample['fs_mask']
        batch_size, L = video_data.size(0), video_data.size(1)
        base_feat_flat = self.features(video_data.view(-1, *(list(video_data.size())[2:])))
        base_feat_unflat = base_feat_flat.view(batch_size, L, *(list(base_feat_flat.size())[1:]))
        base_feat_det = self.RCNN_3d(base_feat_unflat.transpose(1, 2)) # [B, C, L, H, W]
        rois_coord, rois_score, rpn_cls_loss, rpn_reg_loss = self.RCNN_rpn(base_feat_det, det_label, L, training)
        # opt --> prior
        with torch.no_grad():
            prior_data = F.interpolate(opt_data.view(-1, opt_data.size(-2), opt_data.size(-1)).unsqueeze(dim=1), (base_feat_unflat.size(-2), base_feat_unflat.
size(-1)), mode='bilinear').squeeze(dim=1).view(batch_size, L, base_feat_unflat.size(-2), base_feat_unflat.size(-1))
            prior_data = prior_data / prior_data.max(dim=-1)[0].max(dim=-1)[0].view(batch_size, L, 1, 1).clamp_(min=1)
        if training:
            if pose_on:
                origin_hw = [video_data.size(3), video_data.size(4)]
                # assert batch_size == 1
                pose_heatmap, pose_mask = sample['pose_heatmap_global'], sample['pose_mask_global']
                h, w, N = pose_heatmap.size(2), pose_heatmap.size(3), pose_heatmap.size(4)
                pose_heatmap, pose_mask = pose_heatmap.view(-1, h, w, N).permute(0, 3, 1, 2), pose_mask.view(-1, pose_mask.size(-1))
                pose_loss = self.pose_estimate(base_feat_flat, pose_heatmap, pose_mask, pose_sample_rate)
            else:
                pose_loss = rpn_cls_loss.new_zeros(1)
            if fsr_on:
                fsr_reg_loss = rpn_cls_loss.new_zeros(1)
                fsr_cls_loss, fsr_pred, fsr_label, fsr_betas, fsr_probs = self.fsr(base_feat_unflat, fs_label, fs_mask, det_label, Ms=prior_data, digit=False)
            else:
                fsr_reg_loss, fsr_cls_loss = rpn_cls_loss.new_zeros(1), rpn_cls_loss.new_zeros(1)
                fsr_pred, fsr_label = ['x'], ['y']
            if reward_on:
                reward_loss = self.fsr.forward_reward(base_feat_unflat, prior_data, det_label, fs_label, fs_mask, rois_coord, rois_score)
            else:
                reward_loss = rpn_cls_loss.new_zeros(1)
            return rpn_cls_loss.cpu(), rpn_reg_loss.cpu(), fsr_cls_loss.cpu(), fsr_reg_loss.cpu(), fsr_pred, fsr_label, pose_loss.cpu(), reward_loss.cpu()
        else:
            return rois_score, rois_coord

    def fwd_s2(self, sample, training=True, pose_on=False, fsr_on=False, pose_sample_rate=1, reward_on=True):
        # video_data: [B, L, C, H, W], det_label: [B, N, 3], fs_label: [B, N]
        # retrieve data
        video_data_local, opt_data_local, video_data_global, opt_data_global, det_label, fs_label, fs_mask = sample['img_local'], sample['opt_local'], sample['img_global'], sample['opt_global'], sample['det_label'], sample['fs_label'], sample['fs_mask']
        batch_size, L = video_data_local.size(0), video_data_local.size(1)
        video_data = torch.cat([video_data_global, video_data_local], dim=0)
        base_feat_flat = self.features(video_data.view(-1, *(list(video_data.size())[2:]))) # [2*B*L, C, h, w]
        base_feat_unflat = base_feat_flat.view(2*batch_size, L, *(list(base_feat_flat.size())[1:])) # [2*B, L, C, h, w]
        base_feat_det = self.RCNN_3d(base_feat_unflat.transpose(1, 2)) # [2*B, C, L, h, w]
        feat_dim = base_feat_det.size(1)
        base_feat_det = base_feat_det.view([2, batch_size]+list(base_feat_det.size()[1:]))
        base_feat_det = base_feat_det.transpose(0, 1).view([batch_size, 2*feat_dim]+list(base_feat_det.size()[3:]))
        rois_coord, rois_score, rpn_cls_loss, rpn_reg_loss = self.RCNN_rpn(base_feat_det, det_label, L, training)
        # opt --> prior
        with torch.no_grad():
            prior_data_local, prior_data_global = self.make_prior(opt_data_local, base_feat_unflat.size(-2), base_feat_unflat.size(-1)), self.make_prior(opt_data_global, base_feat_unflat.size(-2), base_feat_unflat.size(-1))
        if training:
            if pose_on:
                origin_hw = [video_data.size(3), video_data.size(4)]
                # assert batch_size == 1
                pose_heatmap, pose_mask = torch.cat([sample['pose_heatmap_global'], sample['pose_heatmap_local']], dim=0), torch.cat([sample['pose_mask_global'], sample['pose_mask_local']], dim=0)
                h, w, N = pose_heatmap.size(2), pose_heatmap.size(3), pose_heatmap.size(4)
                pose_heatmap, pose_mask = pose_heatmap.view(-1, h, w, N).permute(0, 3, 1, 2), pose_mask.view(-1, pose_mask.size(-1))
                pose_feats = base_feat_flat.view([2, batch_size*L]+list(base_feat_flat.size()[1:]))
                pose_feats = torch.cat([pose_feats[0][::pose_sample_rate], pose_feats[1][::pose_sample_rate]], dim=0)
                pose_loss = self.pose_estimate(pose_feats, pose_heatmap, pose_mask, 1)
            else:
                pose_loss = rpn_cls_loss.new_zeros(1)
            if fsr_on:
                fsr_reg_loss = rpn_cls_loss.new_zeros(1)
                base_feat_unflat = base_feat_unflat.view([2, batch_size] + list(base_feat_unflat.size()[1:]))
                fsr_cls_loss_g, fsr_pred_g, fsr_label_g, fsr_betas_g, fsr_probs_g = self.fsr(base_feat_unflat[0], fs_label, fs_mask, det_label, Ms=prior_data_global, digit=False)
                fsr_cls_loss_l, fsr_pred_l, fsr_label_l, fsr_betas_l, fsr_probs_l = self.fsr(base_feat_unflat[1], fs_label, fs_mask, det_label, Ms=prior_data_local, digit=False)
                fsr_pred, fsr_label = fsr_pred_l, fsr_label_l
                fsr_cls_loss = fsr_cls_loss_g + fsr_cls_loss_l
            else:
                fsr_reg_loss, fsr_cls_loss = rpn_cls_loss.new_zeros(1), rpn_cls_loss.new_zeros(1)
                fsr_pred, fsr_label = ['x'], ['y']
            if reward_on:
                reward_loss = self.fsr.forward_reward(base_feat_unflat[1], prior_data_local, det_label, fs_label, fs_mask, rois_coord, rois_score)
            else:
                reward_loss = rpn_cls_loss.new_zeros(1)
            return rpn_cls_loss.cpu(), rpn_reg_loss.cpu(), fsr_cls_loss.cpu(), fsr_reg_loss.cpu(), fsr_pred, fsr_label, pose_loss.cpu(), reward_loss.cpu()
        else:
            return rois_score, rois_coord

    def make_prior(self, opt_data, h, w):
        batch_size, L = opt_data.size(0), opt_data.size(1)
        prior_data = F.interpolate(opt_data.view(-1, opt_data.size(-2), opt_data.size(-1)).unsqueeze(dim=1), (h, w), mode='bilinear').squeeze(dim=1).view(batch_size, L, h, w)
        prior_data = prior_data / prior_data.max(dim=-1)[0].max(dim=-1)[0].view(batch_size, L, 1, 1).clamp_(min=1)
        return prior_data

    def get_beta(self, sample):
        video_data, opt_data, det_label, fs_label, fs_mask = sample['img_global'], sample['opt_global'], sample['det_label'], sample['fs_label'], sample['fs_mask']
        batch_size, L = video_data.size(0), video_data.size(1)
        base_feat_flat = self.features(video_data.view(-1, *(list(video_data.size())[2:])))
        base_feat_unflat = base_feat_flat.view(batch_size, L, *(list(base_feat_flat.size())[1:]))
        base_feat_det = self.RCNN_3d(base_feat_unflat.transpose(1, 2)) # [B, C, L, H, W]
        # opt --> prior
        with torch.no_grad():
            prior_data = F.interpolate(opt_data.view(-1, opt_data.size(-2), opt_data.size(-1)).unsqueeze(dim=1), (base_feat_unflat.size(-2), base_feat_unflat.
size(-1)), mode='bilinear').squeeze(dim=1).view(batch_size, L, base_feat_unflat.size(-2), base_feat_unflat.size(-1))
            prior_data = prior_data / prior_data.max(dim=-1)[0].max(dim=-1)[0].view(batch_size, L, 1, 1).clamp_(min=1)

        fs_label = ['x' for _ in range(batch_size)]
        det_label = torch.LongTensor([[[0, L, 1]] for _ in range(batch_size)])
        fs_mask = det_label[:, :, -1].clone()
        fs_mask[fs_mask == 0] = -1
        fsr_cls_loss, fsr_pred, fsr_label, fsr_betas, fsr_probs = self.fsr(base_feat_unflat, fs_label, fs_mask, det_label, Ms=prior_data, digit=False)
        fsr_betas = [fsr_betas[b][0] for b in range(batch_size)]
        return fsr_betas
