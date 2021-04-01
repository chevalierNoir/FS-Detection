import os
import cv2
import sys
import pickle
import logging
import json
import torch
import editdistance
import lev
import numpy as np
from collections import defaultdict
from metrics.rec_model import ConvLstm
from ctc_decoder import Decoder 
import torch.nn.functional as functional
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from metrics.ap_iou import get_iou
from nms import nms_cpu

logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(message)s")

class Evaluator(object):
    def __init__(self, label_fn, scp_fn, model_fn):
        # fixed fields
        n_hidden, n_layers, bid = 512, 1, True
        char_list = "_ '&.@acbedgfihkjmlonqpsrutwvyxz"
        self.imsize, self.immean, self.imstd = 128, torch.FloatTensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1).cuda(), torch.FloatTensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1).cuda()
        self.chunk_size = 300
        self.model = ConvLstm(n_hidden, len(char_list), n_layers, bid=bid).cuda()
        logging.info(f"Loading model {model_fn}")
        self.model.load_state_dict(torch.load(model_fn))
        self.decoder = Decoder(char_list)
        data_info = json.load(open(label_fn, 'r'))
        scp_json = json.load(open(scp_fn, 'r'))
        info = defaultdict(list)
        for line_id, line in enumerate(data_info):
            chunk_id = line['video_id'] + '-' + str(line['frames'][0][0]) + '_' + str(line['frames'][0][1])
            subid, rgb_video, opt_video = scp_json[chunk_id][0], scp_json[chunk_id][1], scp_json[chunk_id][2]
            det_label, fsid, fs_label, fs_mask = np.array([[win[0], win[1]] for win in line['wins']]), line['fs_id'], line['fs_label'], np.array(line['fs_mask'])
            if len(fsid) > 0:
                det_label = det_label[fs_mask]
                idx_sorted = np.argsort(det_label[:, 0]).tolist()
                fsid, fs_label, det_label = [fsid[ix] for ix in idx_sorted], [fs_label[ix] for ix in idx_sorted], [det_label[ix].tolist() for ix in idx_sorted]
                info[rgb_video].append([chunk_id, subid, fs_label, det_label, fsid])
        self.info = [[key] + val for key, vals in info.items() for val in vals]
        logging.info(f"{len(self.info)} items")
        self.open_rgb, self.rgb_buffer = None, None

    def batch_predict(self, id2det_pred):
        id2rec_pred = {}
        self.model.eval()
        self.model.lstm.batch_first = True
        for item in self.info:
            rgb_video, chunk_id, subid, fs_label, det_label, fsid = item

            preds = [(int(x), int(y)) for x, y in id2det_pred[chunk_id][:, :-1].tolist()] # to int, added

            # filter out empty preds
            preds_ = list(filter(lambda x: min(x[1], self.chunk_size) - max(x[0], 0) > 0, preds))
            scores = id2det_pred[chunk_id].tolist()
            scores = list(filter(lambda x: min(x[1], self.chunk_size) - max(x[0], 0) > 0, scores))
            scores = [score[-1] for score in scores]
            assert len(scores) == len(preds_)
            if len(preds_) < len(preds):
                logging.info(f'valid intervals : {len(preds_)}/{len(preds)}')
                preds = preds_

            if rgb_video != self.open_rgb:
                self.open_rgb = rgb_video
                self.rgb_buffer = load_video(rgb_video, size=self.imsize, as_gray=False)
            imgs = self.rgb_buffer[self.chunk_size*subid: self.chunk_size*(subid+1)]
            fs_preds, fs_grts = [], []
            with torch.no_grad():
                img_tensor = torch.from_numpy(imgs.transpose((0, 3, 1, 2))).cuda()
                img_tensor = (img_tensor/256 - self.immean) / self.imstd
                feat = self.model.backbone.model1_0(img_tensor)
                feat = functional.adaptive_avg_pool2d(feat, (1, 1)).view(self.chunk_size, -1)
                frame_sizes = torch.LongTensor([min(self.chunk_size, end)-max(start, 0) for start, end in preds])
                batch_feat = feat.new_zeros(len(preds), frame_sizes.max().item(), feat.size(-1))
                for j, (start, end) in enumerate(preds):
                    batch_feat[j, :frame_sizes[j]] = feat[int(start): int(end)]
                batch_feat = pack_padded_sequence(batch_feat, frame_sizes, batch_first=True, enforce_sorted=False)
                batch_output, _ = self.model.lstm(batch_feat)
                batch_output, _ = pad_packed_sequence(batch_output, batch_first=True)
                batch_logits = self.model.lt(batch_output)
                batch_log_probs = self.model.classifier(batch_logits).cpu().numpy()
                frame_sizes = frame_sizes.numpy()
            for j, (start, end) in enumerate(preds):
                log_probs = batch_log_probs[j, :frame_sizes[j]]
                pred_fs = ''.join(self.decoder.greedy_decode(log_probs, digit=False))
                if len(pred_fs) == 0:
                    pred_fs = 'x'
                fs_preds.append([start, end, pred_fs])
            for i in range(len(det_label)):
                fs_grts.append([det_label[i][0], det_label[i][1], fs_label[i]])
            id2rec_pred[chunk_id] = {'pred': fs_preds, 'grt': fs_grts, 'score': scores}
            # break
        return id2rec_pred


def load_video(video_path, size, as_gray=False):
    logging.info(f'Loading video {video_path} (BGR)')
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if ret:
            if not as_gray:
                frame = frame
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            W, H = frame.shape[1], frame.shape[0]
            if W < H:
                frame = frame[H//2-W//2: H//2+W//2]
            elif W > H:
                frame = frame[:, W//2-H//2: W//2+H//2]

            # dx, dy = int(frame.shape[1] * scale), int(frame.shape[0] * scale)
            frame = cv2.resize(frame, (size, size))
            frames.append(frame)
        else:
            break
    frames = np.stack(frames, axis=0).astype(np.float32)
    return frames

def get_rec_dict(grt_pred_pkl, grt_rec_pkl, model_fn, label_fn, scp_fn):
    logging.info(f'DET pred file {grt_pred_pkl} -> REC pred file {grt_rec_pkl}')
    evaluator = Evaluator(label_fn, scp_fn, model_fn)

    grt_pred = pickle.load(open(grt_pred_pkl, 'rb'))
    preds, grts, chunk_ids = grt_pred['pred'], grt_pred['grt'], grt_pred['id']
    id2det = {}
    for i in range(len(preds)):
        pred, grt, chunk_id = preds[i], grts[i], chunk_ids[i]
        id2det[chunk_id] = pred
    id2rec = evaluator.batch_predict(id2det)
    grt_rec_data = {'pred': [], 'grt': [], 'id': [], 'score': []}
    for chunk_id in id2rec.keys():
        grt_rec_data['id'].append(chunk_id)
        grt_rec_data['pred'].append(id2rec[chunk_id]['pred'])
        grt_rec_data['grt'].append(id2rec[chunk_id]['grt'])
        grt_rec_data['score'].append(id2rec[chunk_id]['score'])
    pickle.dump(grt_rec_data, open(grt_rec_pkl, 'wb'))
    return

def get_seq_ler_vars(rec_pred_pkl, nms_thresh=0.1):
    # det_pred is used to retrieve score
    def pad_special(coord_label, chunk_size=300):
        prev = 0
        labels = []
        for start, end, label in coord_label:
            if start != prev:
                labels.append('X')
            labels.append(label)
            prev = end
        if prev != chunk_size:
            labels.append('X')
        labels = ''.join(labels)
        return labels

    grt_rec = pickle.load(open(rec_pred_pkl, 'rb'))
    preds, grts, chunk_ids, scores = grt_rec['pred'], grt_rec['grt'], grt_rec['id'], grt_rec['score']
    id2rec = {}
    for i in range(len(preds)):
        pred, grt, chunk_id, score = preds[i], grts[i], chunk_ids[i], scores[i]
        coord = np.array([p[:-1]+[s] for p, s in zip(pred, score)])
        ids_nms = nms_cpu(coord, nms_thresh).numpy()
        pred = [pred[j]+[score[j]] for j in ids_nms]
        grt = sorted(grt, key=lambda x: x[0])
        id2rec[chunk_id] = {'pred': pred, 'grt': grt}
    max_acc = []
    for thr in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]:
        chars_pred, chars_grt, coords_pred, coords_grt, ious = [], [], [], [], []
        for chunk_id in id2rec.keys():
            pred = list(filter(lambda x: x[-1]>thr, id2rec[chunk_id]['pred']))
            if len(pred) == 0:
                pred = [id2rec[chunk_id]['pred'][0][:-1]]
            else:
                pred = [p[:-1] for p in pred]
            pred = sorted(pred, key=lambda x: x[0])
            iou_mat = get_iou([x[:2] for x in pred], [x[:2] for x in id2rec[chunk_id]['grt']])
            ious.append(iou_mat.max(axis=-1).mean())
            pred = pad_special(pred)
            grt = pad_special(id2rec[chunk_id]['grt'])
            chars_pred.append(pred)
            chars_grt.append(grt)
        acc = lev.compute_acc(chars_pred, chars_grt, costs=(1, 1, 1))
        iou = sum(ious) / len(ious)
        max_acc.append(acc)
    max_acc = max(max_acc)
    result_txt = rec_pred_pkl+'.msa'
    print(f"Write MSA into {result_txt}")
    open(result_txt, 'w').write(f'MSA {max_acc}\n')
    print(f"MSA: {max_acc}")
    return

def get_precision_recall(y_pred, y_true, ler_thr, iou_thr, ordering=False):
    # ordering: going from highest to lowest, gradually selecting true positive
    n_pred, n_true = len(y_pred), len(y_true)
    if n_pred == 0:
        precision_tp, recall_tp = 0, 0
    else:
        if ler_thr != None:
            args = [(y_pred[i][-1], y_true[j][-1]) for i in range(n_pred) for j in range(n_true)]

            ler = list(map(lambda x: 100*(1 - editdistance.distance(*x)/len(x[1])), args))
            ler = np.array(ler).reshape(n_pred, n_true) / 100
            conf_ler = (ler > ler_thr).astype(np.int32)
        if iou_thr != None:
            coord_pred, coord_true = [y_pred[i][:-1] for i in range(n_pred)], [y_true[i][:-1] for i in range(n_true)]
            iou = get_iou(coord_pred, coord_true)
            conf_iou = (iou > iou_thr).astype(np.int32)
        if ler_thr != None and iou_thr != None:
            conf = np.logical_and(conf_iou, conf_ler).astype(np.int32)
        elif ler_thr != None and iou_thr == None:
            conf = conf_ler
        elif ler_thr == None and iou_thr != None:
            conf = conf_iou
        else:
            raise ValueError(f"LER thr and IoU thr both None")
        # if >1 detections for one segment, only the one with highest IoU is TP
        if ordering:
            n_tp = 0
            for i in range(n_pred):
                max_id = np.argmax(ler[i])
                if conf[i, max_id]:
                    n_tp += 1
                    ler, conf = np.delete(ler, max_id, axis=1), np.delete(conf, max_id, axis=1)
                if ler.shape[1] == 0:
                    break
            precision_tp, recall_tp = n_tp, n_tp
        else:
            try:
                mask = (np.max(ler, axis=0).reshape(1, -1) == ler).astype(np.int32)
            except Exception as err:
                print(ler)
                raise err
            conf = conf * mask
            precision_tp, recall_tp = (conf.sum(axis=1) > 0).sum(), (conf.sum(axis=0) > 0).sum()
    return precision_tp, recall_tp, n_pred, n_true


def get_mAP(pred_pkl, stat_pkl, ler_thrs=[0.0], iou_thrs=[0.1], ordering=False):
    ss = pickle.load(open(pred_pkl, 'rb'))
    y_true = ss['grt']
    num_roi_thrs = range(1, 50)
    aps = {}
    for ler_thr, iou_thr in zip(ler_thrs, iou_thrs):
        precisions, recalls = [], []
        for num_roi_thr in num_roi_thrs:
            y_pred = [x[:num_roi_thr] for x in ss['pred']]
            args = list(zip(y_pred, y_true, [ler_thr for _ in range(len(y_pred))], [iou_thr for _ in range(len(y_pred))], [ordering for _ in range(len(y_pred))]))
            prs = list(map(lambda x: get_precision_recall(*x), args))
            prs = list(zip(*prs))
            precision_tp, recall_tp, n_pred, n_true = sum(prs[0]), sum(prs[1]), sum(prs[2]), sum(prs[3])

            precision, recall = precision_tp / max(n_pred, 1), recall_tp / max(n_true, 1)
            precisions.append(precision)
            recalls.append(recall)
        logging.info(f"{ler_thr}")
        aps[(ler_thr, iou_thr)] = [precisions, recalls]
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
    for (ler_thr, iou_thr), stats in ss.items():
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
        iou_to_mAP[(ler_thr, iou_thr)] = mAP
        logging.info(f'Acc {ler_thr}, IoU {iou_thr}, mAP: {mAP}')
    return iou_to_mAP

def get_ap_ler(rec_pred, ler_thrs=[0.0, 0.1, 0.2, 0.3], iou_thrs=[0.1, 0.2, 0.3, 0.4], ordering=False, ptype='pascal'):
    stat_pkl, result_txt = rec_pred + ".tmp", rec_pred+'.acc'
    get_mAP(rec_pred, stat_pkl, ler_thrs=ler_thrs, iou_thrs=iou_thrs, ordering=ordering)
    ler_to_mAP = compute_mAP_from_stat(stat_pkl, ptype=ptype)
    print(f"Write AP@Acc into {result_txt}")
    with open(result_txt, 'w') as fo:
        for (ler_thr, iou_thr), mAP in ler_to_mAP.items():
            fo.write(str(iou_thr)+','+str(mAP)+'\n')
    os.remove(stat_pkl)
    return

# def nms(dets, thresh):
#     # dets = dets.detach().cpu().numpy()
#     x1 = dets[:, 0]
#     x2 = dets[:, 1]
#     scores = dets[:, 2]

#     length = (x2 - x1 + 1)
#     order = scores.argsort()[::-1]

#     keep = []
#     while order.size > 0:
#         i = order.item(0)
#         keep.append(i)
#         xx1 = np.maximum(x1[i], x1[order[1:]])
#         xx2 = np.minimum(x2[i], x2[order[1:]])
#         inter = np.maximum(0.0, xx2 - xx1 + 1)
#         ovr = inter / (length[i] + length[order[1:]] - inter)
#         inds = np.where(ovr < thresh)[0]
#         order = order[inds+1]
#     return np.array(keep)
