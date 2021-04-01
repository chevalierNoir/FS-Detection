import cv2
import logging
import json
import torch
import numpy as np

CLAMP_MIN = 1.0e-5
CLAMP_MAX = 1 - CLAMP_MIN

def get_scale_range(base_size=8, scales=2**np.arange(3, 6)):
    scale_range = np.array([1, base_size]) - 1
    l = scale_range[1] - scale_range[0] + 1
    x_ctr = scale_range[0] + 0.5 * (l - 1)
    ls = l * scales
    ls = ls[:, np.newaxis]
    scale_range = np.hstack((x_ctr - 0.5 * (ls - 1),
                             x_ctr + 0.5 * (ls - 1))).astype(np.float32)
    scale_range = torch.from_numpy(scale_range)
    return scale_range


def smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, sigma=1.0, dim=[1]):
    sigma_2 = sigma ** 2
    box_diff = bbox_pred - bbox_targets
    in_box_diff = bbox_inside_weights * box_diff
    abs_in_box_diff = torch.abs(in_box_diff)
    smoothL1_sign = (abs_in_box_diff < 1. / sigma_2).detach().float()
    in_loss_box = torch.pow(in_box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign \
                  + (abs_in_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
    out_loss_box = bbox_outside_weights * in_loss_box
    loss_box = out_loss_box
    for i in sorted(dim, reverse=True):
      loss_box = loss_box.sum(i)
    loss_box = loss_box.mean()
    return loss_box

def mask_rpn_losses(masks_pred, masks_label):
    """Mask RPN specific losses."""
    batch_size, feat_len = masks_pred.size()
    weight = (masks_label > -1).float()  # masks_int32 {1, 0, -1}, -1 means ignore
    loss = F.binary_cross_entropy(
        masks_pred.view(batch_size, -1), masks_label, weight, size_average=False)
    loss /= weight.sum()
    return loss

def get_map_size(conv_layer, scp_fn, scale):
    test_video = json.load(open(scp_fn, 'r')).values().__iter__().__next__()[1]
    cap = cv2.VideoCapture(test_video)
    width, height  = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap.release()
    size = int(min(width, height)*scale)
    conv_layer.eval()
    with torch.no_grad():
        img = torch.zeros(1, 3, size, size).to(next(conv_layer.parameters()).device)
        fmap = conv_layer(img)
        map_h, map_w = fmap.size(2), fmap.size(3)
    map_h, map_w = 2*map_h+1, 2*map_w+1
    return map_h, map_w

def get_det_bbox(beta, scale=0.5, win_size=5, W=640, H=360):
    size = scale * min(W, H)
    L, w, h = beta.shape[0], beta.shape[1], beta.shape[2]
    margin_x, stride = (W-H)//2, min(H, W) / w
    max_ids = beta.reshape(L, -1).argmax(axis=-1)
    cxs, cys = max_ids % w, max_ids // w
    cxs, cys = stride * cxs, stride * cys
    bboxes = []
    for j in range(L):
        start, end = max(j-win_size//2, 0), min(j+win_size//2, L)
        cx, cy = cxs[start: end].mean(), cys[start: end].mean()
        x0, y0, x1, y1 = cx - size/2 + margin_x, cy - size/2, cx + size/2 + margin_x, cy + size/2
        bboxes.append([x0, y0, x1, y1])
    bboxes = np.array(bboxes, dtype=np.float32)
    return bboxes
