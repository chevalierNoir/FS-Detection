import os
import pickle
import numpy as np


def get_iou(y_pred, y_true):
    # y_pred, y_true: [[start, end],...]
    # return: ndarray of shape (n_pred, n_true)
    y_pred, y_true = np.array(y_pred), np.array(y_true)
    n_pred, n_true = y_pred.shape[0], y_true.shape[0]
    y_pred = np.repeat(y_pred.reshape(n_pred, 1, 2), n_true, axis=1).reshape(-1, 2)
    y_true = np.repeat(y_true.reshape(1, n_true, 2), n_pred, axis=0).reshape(-1, 2)
    max_start, min_end = np.maximum(y_pred[:, 0], y_true[:, 0]), np.minimum(y_pred[:, 1], y_true[:, 1])
    min_start, max_end = np.minimum(y_pred[:, 0], y_true[:, 0]), np.maximum(y_pred[:, 1], y_true[:, 1])
    intersection = min_end - max_start + 1
    union = max_end - min_start + 1
    iou = (intersection / union).reshape(n_pred, n_true).clip(min=0)
    iou = iou.reshape(n_pred, n_true)
    return iou


def get_precision_recall(y_pred, y_true, thr, ordering=False):
    n_pred, n_true = len(y_pred), len(y_true)
    if n_pred == 0:
        precision_tp, recall_tp = 0, 0
    else:
        iou = get_iou(y_pred, y_true)
        conf = (iou > thr).astype(np.int32)
        # if >1 detections for one segment, only the one with highest IoU is TP
        if ordering:
            n_tp = 0
            for i in range(n_pred):
                max_id = np.argmax(iou[i])
                if conf[i, max_id]:
                    n_tp += 1
                    iou, conf = np.delete(iou, max_id, axis=1), np.delete(conf, max_id, axis=1)
                if iou.shape[1] == 0:
                    break
            precision_tp, recall_tp = n_tp, n_tp
        else:
            try:
                mask = (np.max(iou, axis=0).reshape(1, -1) == iou).astype(np.int32)
            except Exception as err:
                print(iou)
                raise err
            conf = conf * mask
            precision_tp, recall_tp = (conf.sum(axis=1) > 0).sum(), (conf.sum(axis=0) > 0).sum()
    return precision_tp, recall_tp, n_pred, n_true


def get_mAP(pred_pkl, stat_pkl, iou_thrs=[0.1], ordering=False):
    ss = pickle.load(open(pred_pkl, 'rb'))
    y_true = ss['grt']
    num_roi_thrs = range(1, 50)
    aps = {}
    for iou_thr in iou_thrs:
        precisions, recalls = [], []
        for num_roi_thr in num_roi_thrs:
            y_pred = [x[:num_roi_thr, :2] for x in ss['pred']]
            # parallel
            args = list(zip(y_pred, y_true, [iou_thr for _ in range(len(y_pred))], [ordering for _ in range(len(y_pred))]))
            prs = list(map(lambda x: get_precision_recall(*x), args))
            prs = list(zip(*prs))
            precision_tp, recall_tp, n_pred, n_true = sum(prs[0]), sum(prs[1]), sum(prs[2]), sum(prs[3])
            precision, recall = precision_tp / max(n_pred, 1), recall_tp / max(n_true, 1)
            precisions.append(precision)
            recalls.append(recall)
        print(f"IoU={iou_thr}")
        aps[iou_thr] = [precisions, recalls]
    pickle.dump(aps, open(stat_pkl, 'wb'))
    return aps

def compute_mAP_from_stat(stat_pkl, ptype='pascal'):
    ss = pickle.load(open(stat_pkl, 'rb'))
    if ptype == 'pascal':
        target_recall_vals = np.arange(0, 1.05, 0.1)
    elif ptype == 'coco':
        target_recall_vals = np.arange(0, 1.00001, 0.01)
    else:
        raise NotImplementedError
    err_thr = 0.1
    iou_to_mAP = {}
    for iou_thr, stats in ss.items():
        precision, recall = np.array(stats[0]), np.array(stats[1])
        target_precision_vals = []
        for val in target_recall_vals:
            recall_diff = recall - val
            valid_ids = recall_diff >= 0
            if valid_ids.sum() > 0:
                target_precision_val = np.max(precision[valid_ids])
            else:
                target_precision_val = 0
            target_precision_vals.append(target_precision_val)
        mAP = np.array(target_precision_vals).mean()
        iou_to_mAP[iou_thr] = mAP
        print('IoU %.1f, mAP: %.3f' % (iou_thr, mAP))
    return iou_to_mAP

def get_precision_per_sample(pred_pkl, stat_pkl, iou_thr=0.5, recall_thr=0.5):
    ss = pickle.load(open(pred_pkl, 'rb'))
    y_true = ss['grt']
    num_roi_thrs = range(1, 50)
    precision_all = []
    for i in range(len(y_true)):
        precisions, recalls = np.zeros(len(num_roi_thrs)), np.zeros(len(num_roi_thrs))
        for j, num_roi_thr in enumerate(num_roi_thrs):
            y_pred_i = [ss['pred'][i][:num_roi_thr, :2]]
            y_true_i = [ss['grt'][i]]
            # parallel
            args = list(zip(y_pred_i, y_true_i, [iou_thr for _ in range(len(y_pred_i))]))
            prs = list(map(lambda x: get_precision_recall(*x), args))
            prs = list(zip(*prs))
            precision_tp, recall_tp, n_pred, n_true = sum(prs[0]), sum(prs[1]), sum(prs[2]), sum(prs[3])
            precision, recall = precision_tp / max(n_pred, 1), recall_tp / max(n_true, 1)
            precisions[j] = precision
            recalls[j] = recall
        mask = recalls >= recall_thr
        if mask.sum() > 0:
            precision_val = precisions[mask].max()
        else:
            precision_val = 0
        precision_all.append(precision_val)
    pickle.dump(precision_all, open(stat_pkl, 'wb'))
    return 0

def get_ap_iou(pred_pkl, iou_thrs=[0.1, 0.2, 0.3, 0.4, 0.5], ptype='coco'):
    stat_pkl, result_txt = pred_pkl + ".tmp", pred_pkl + '.iou'
    get_mAP(pred_pkl, stat_pkl, iou_thrs=iou_thrs, ordering=True)
    iou_to_mAP = compute_mAP_from_stat(stat_pkl, ptype)
    print(f"Write AP@IoU into {result_txt}")
    with open(result_txt, 'w') as fo:
        for iou, mAP in iou_to_mAP.items():
            fo.write(str(iou)+","+str(mAP)+"\n")
    os.remove(stat_pkl)
    return iou_to_mAP
