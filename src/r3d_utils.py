import torch
import numpy as np
from twin_transform import twin_transform, twin_transform_inv, twin_transform_batch, twins_overlaps_batch
from nms import nms_cpu
from torch.nn import functional as F 

def get_anchor_coords(anchor_scales, feat_stride, vol_len):
    # return: [A, L, 2]
    roots = (torch.arange(vol_len, device=anchor_scales.device) * feat_stride).unsqueeze(dim=-1).type_as(anchor_scales)
    anchors = torch.stack((roots + anchor_scales[:, 0].unsqueeze(dim=0), roots + anchor_scales[:, 1].unsqueeze(dim=0)), dim=-1) # [L, A, 2]
    anchors = anchors.transpose(0, 1)
    return anchors

def generate_proposal(anchor_scales, prob_vol, delta_vol, feat_stride, nms_top_n, nms_thr):
    batch_size, vol_len = prob_vol.size(0), prob_vol.size(-1)
    anchors = get_anchor_coords(anchor_scales, feat_stride, vol_len)
    anchors = anchors.unsqueeze(dim=0).expand(batch_size, -1, -1, -1).contiguous() # [B, A, L, 2]
    anchors = anchors.view(batch_size, -1, 2) # [B, A*L, 2]
    prob_vol, delta_vol = prob_vol.view(batch_size, 2, -1).transpose(1, 2), delta_vol.view(batch_size, 2, -1).transpose(1, 2) # [B, A*L, 2]
    prob_vol = prob_vol[:, :, 1:2] # [B, A*L, 1]
    proposals = twin_transform_inv(anchors, delta_vol, batch_size)
    mask = (proposals[:, :, 0] >= 0) & (proposals[:, :, 1] < vol_len * feat_stride)

    # if batch_size > 1:
    #     raise ValueError('Flatten proposal[mask] if batch_size > 1')
    # proposals, prob_vol, delta_vol = proposals[mask].view(batch_size, -1, 2), prob_vol[mask].view(batch_size, -1, 1), delta_vol[mask].view(batch_size, -1, 2)
    mask = mask.type_as(proposals).unsqueeze(dim=-1)
    proposals, prob_vol, delta_vol = proposals*mask, prob_vol*mask, delta_vol*mask 
    _, indexes = torch.sort(prob_vol, dim=1, descending=True)
    indexes = indexes.squeeze(dim=-1)
    props_coord, props_score = anchors.new_zeros(batch_size, nms_top_n, 2), prob_vol.new_zeros(batch_size, nms_top_n, 1)
    for i in range(batch_size):
        proposal_i, prob_vol_i, delta_vol_i = proposals[i][indexes[i]], prob_vol[i][indexes[i]], delta_vol[i][indexes[i]]
        index_nms= nms_cpu(torch.cat((proposal_i, prob_vol_i), dim=-1), nms_thr).long().squeeze(dim=-1)

        if len(index_nms) >= nms_top_n:
            props_coord[i] = proposal_i[index_nms][:nms_top_n]
            props_score[i] = prob_vol_i[index_nms][:nms_top_n]
        else:
            props_coord[i, :len(index_nms)] = proposal_i[index_nms]
            props_score[i, :len(index_nms)] = prob_vol_i[index_nms]
    props_coord = props_coord.type(torch.LongTensor)
    return props_coord, props_score


def generate_label_rpn(anchor_scales, feat_stride, vol_len, video_len, gt_labels, high_iou_thr=0.7, low_iou_thr=0.3, neg_to_pos_ratio=3.0, min_neg_num=10):
    # gt_labels: [B, K, 3]
    # return: [B, N], [B, N, 2]
    batch_size = len(gt_labels)
    anchors = get_anchor_coords(anchor_scales, feat_stride, vol_len)
    anchors = anchors.unsqueeze(dim=0).expand(batch_size, -1, -1, -1).contiguous() # [B, A, L, 2]
    anchors = anchors.view(batch_size, -1, 2) # [B, A*L, 2]
    N = anchors.size(1)
    overlaps = twins_overlaps_batch(anchors, gt_labels) # [B, N, K]
    anchor_indexes = torch.arange(N).type(torch.LongTensor)

    boundary_mask = (anchors[0, :, 0] >= -float('inf')) & (anchors[0, :, 1] < float('inf'))
    anchor_cls_label, anchor_reg_label = -anchors.new_ones(batch_size, N).type(torch.LongTensor), -anchors.new_ones(batch_size, N, 2, dtype=torch.float32)
    for b in range(batch_size):
        max_overlap, max_gt_id = torch.max(overlaps[b], dim=-1)
        pos_mask, neg_mask = max_overlap >= high_iou_thr, max_overlap < low_iou_thr
        pos_indexes, neg_indexes = anchor_indexes[pos_mask & boundary_mask], anchor_indexes[neg_mask & boundary_mask]
        if len(pos_indexes) == 0:
            pos_mask = max_overlap >= low_iou_thr
            pos_indexes = anchor_indexes[pos_mask & boundary_mask]
            # print('indexes with highest IoU: %d' % (len(pos_indexes)))
        num_pos, num_neg = len(pos_indexes), int(len(pos_indexes) * neg_to_pos_ratio) if len(pos_indexes) > 0 else min_neg_num
        if len(neg_indexes) > num_neg:
            npm = torch.randperm(len(neg_indexes))[:num_neg]
            neg_indexes = neg_indexes[npm]
        if len(pos_indexes) == 0:
            non_zero_gt = gt_labels[b, gt_labels[b].sum(dim=-1) != 0]
            # print('gt label: {}'.format(non_zero_gt))
        pos_anchors, neg_anchors = anchors[b, pos_indexes], anchors[b, neg_indexes]
        anchor_cls_label[b, pos_indexes] = 1
        anchor_cls_label[b, neg_indexes] = 0
        pos_gts = gt_labels[b, max_gt_id[pos_indexes], :2]
        if len(pos_indexes) > 0:
            delta = twin_transform(pos_anchors, pos_gts)
            anchor_reg_label[b, pos_indexes] = delta
    return anchor_cls_label, anchor_reg_label


def generate_label_roi(rois, video_len, gt_labels, high_iou_thr=0.5, low_iou_thr=0.5, neg_to_pos_ratio=3.0, min_neg_num=10):
    # rois: [B, N, 2], gt_labels: [B, K, 3]
    # return: [B, N+K], [B, N+K, 2]
    # all rois 0-overlap with padded ground-truth
    batch_size = len(gt_labels)
    N = rois.size(1)
    gt_coords, gt_cls = gt_labels[:, :, :2].float(), gt_labels[:, :, 2].long()
    rois = rois.type_as(gt_coords)
    overlaps = twins_overlaps_batch(rois, gt_coords) # [B, N+K, K]
    roi_indexes = torch.arange(N).type(torch.LongTensor)
    padding_mask = ~((rois[:, :, 0] == 0) & (rois[:, :, 1] == 0))
    boundary_mask = (rois[:, :, 0] >= -float('inf')) & (rois[:, :, 1] < float('inf'))
    roi_cls_label, roi_reg_label = -rois.new_ones(batch_size, N).long(), -rois.new_ones(batch_size, N, 2)
    for b in range(batch_size):
        max_overlap, max_gt_id = torch.max(overlaps[b], dim=-1)
        pos_mask, neg_mask = (max_overlap >= high_iou_thr), (max_overlap < low_iou_thr) & (max_overlap >= 0)
        pos_indexes, neg_indexes = roi_indexes[pos_mask & boundary_mask[b] & padding_mask[b]], roi_indexes[neg_mask & boundary_mask[b] & padding_mask[b]]
        num_pos, num_neg = len(pos_indexes), int(len(pos_indexes) * neg_to_pos_ratio) if len(pos_indexes) > 0 else min_neg_num
        if len(neg_indexes) > num_neg:
            npm = torch.randperm(len(neg_indexes))[:num_neg]
            neg_indexes = neg_indexes[npm]
        print('roi, positive: %d, negative: %d' % (len(pos_indexes), len(neg_indexes)))
        pos_rois, neg_rois = rois[b, pos_indexes], rois[b, neg_indexes]
        roi_cls_label[b, pos_indexes] = gt_cls[b, max_gt_id[pos_indexes]]
        roi_cls_label[b, neg_indexes] = 0
        if len(pos_indexes) > 0:
            pos_gts = gt_coords[b, max_gt_id[pos_indexes]]
            delta = twin_transform(pos_rois, pos_gts)
            roi_reg_label[b, pos_indexes] = delta
    return roi_cls_label, roi_reg_label


def log_loss(prob, label):
    # prob: [B, N, C], label: [B, N]
    clamp_min = 1.0e-5
    log_prob = torch.log(prob.clamp(min=clamp_min, max=1-clamp_min))
    B, N, C = prob.size(0), prob.size(1), prob.size(-1)
    log_prob, label = log_prob.view(-1, C), label.view(-1)
    label_onehot = log_prob.new_zeros(B*N, C)
    label_onehot[torch.arange(B*N).type(torch.LongTensor), label.abs()] = 1 # -1 --> 1, error
    mask = (label != -1).type_as(prob)
    loss = (-log_prob * label_onehot).sum(dim=-1)
    loss = (loss * mask).sum() / mask.sum()
    return loss

def smooth_l1_loss(delta, reg_label, cls_label):
    # delta: [B, N, 2], reg_label: [B, N, 2], cls_label: [B, N]
    delta, reg_label, cls_label = delta.view(-1, 2), reg_label.view(-1, 2), cls_label.view(-1)
    diff = delta - reg_label
    mask_p1, mask_p2 = (diff.abs() < 1).type_as(diff), (diff.abs() >= 1).type_as(diff)
    val_p1, val_p2 = 0.5 * diff * diff, diff.abs() - 0.5
    loss = (val_p1 * mask_p1 + val_p2 * mask_p2).sum(dim=-1)
    mask = (cls_label >= 1).type_as(delta)
    loss_ = (loss * mask).sum() / mask.sum().clamp(min=1)
    if torch.isnan(loss_).sum() > 0:
        print(loss, mask)
        raise
    return loss_

def roi_pooling(feat_vol, rois, size):
    """
    feat_vol: (B, C, L, H, W)
    """
    output = []
    num_rois = rois.size(0)
    B, C = feat_vol.size(0), feat_vol.size(1)
    for i in range(num_rois):
        bid, start, end = rois[i, 0], rois[i, 1], rois[i, 2]
        if (start == 0 and end == 0) or start >= feat_vol.size(2):
            fout = feat_vol.new_zeros(1, C, *size)
        else:
            fout = F.adaptive_max_pool3d(feat_vol[bid: bid+1, :, start: end+1, :, :], output_size=size)
        output.append(fout)
    output = torch.cat(output)
    if B > 1:
        raise ValueError('Unflatten if batch_size > 1')
    output = output.view(B, output.size(0)//B, *list(output.size()[1:]))
    return output
