import logging
import math
import time
import torch
import lev
import editdistance
import torch.nn as nn
from torch.nn import functional
from ctc_decoder import Decoder

class CTCLoss(nn.Module):
    def __init__(self, ctc_type):
        super(CTCLoss, self).__init__()
        if ctc_type == 'builtin':
            from torch.nn import CTCLoss as BuiltinCTCLoss
            print("Using built-in CTC")
            self.ctc_loss = BuiltinCTCLoss(zero_infinity=True) # normalize over batch
        elif ctc_type == 'warp':
            from warpctc_pytorch import CTCLoss as WarpCTCLoss
            print("Using warp CTC")
            self.ctc_loss = WarpCTCLoss(size_average=True, reduce=True)
        else:
            raise NotImplementedError

    def forward(self, *args):
        return self.ctc_loss(*args)

class FSR(nn.Module):
    def __init__(self, hidden_size, attn_size, n_layers, char_list, prior_gamma=1.0, ctc_type='builtin', iou_thr=0.5, n_top=50):
        super(FSR, self).__init__()
        self.hidden_size, self.attn_size, self.n_layers = hidden_size, attn_size, n_layers
        self.encoder = AttnEncoder(hidden_size, attn_size, output_size=len(char_list), n_layers=n_layers, prior_gamma=prior_gamma, cell='LSTM')
        self.decoder = Decoder(char_list)
        self.ctc_type = ctc_type
        self.ctc_loss = CTCLoss(ctc_type)
        self.int_to_char, self.char_to_int = dict([(i, c) for i, c in enumerate(char_list)]), dict([(c, i) for i, c in enumerate(char_list)])
        self.iou_thr, self.n_top = iou_thr, n_top
        logging.info(f'IoU threshold {self.iou_thr}')

    def forward(self, Fs, fs_label, fs_mask, det_label, Ms=None, digit=True):
        # Fs: [B, L, F, h, w], det_label: [B, N, 3], fs_label: [[label_1, label_2], [label_1, label_2], ...], Ms: [B, L, H, W]
        # Out: betas: [B, L, h, w]
        # S1. prepare frame data
        fs_indexes, labels, prob_sizes, label_sizes = [], [], [], []
        B, N = det_label.size(0), det_label.size(1)
        feat_dim, h, w = Fs.size(2), Fs.size(3), Fs.size(4)
        L, _, _ = Ms.size(1), Ms.size(2), Ms.size(3)

        for b in range(B):
            if len(fs_label[b]) == 0:
                return Ms.new_zeros(1), [], [], [], []
            val_seqs = det_label[b][fs_mask[b] == 1]
            assert len(val_seqs) == len(fs_label[b]), f"unequal fingerspelling segments: {val_seqs}, {fs_label[b]}"
            for i in range(len(val_seqs)):
                start, end = max(int(val_seqs[i, 0].item()), 0), min(int(val_seqs[i, 1].item()), L)
                fs_indexes.append((b, start, end))
                labels.extend([self.char_to_int[char] for char in fs_label[b][i]])
                prob_sizes.append(end-start)
                label_sizes.append(len(fs_label[b][i]))
        fs_feats, fs_priors = Fs.new_zeros(len(prob_sizes), max(prob_sizes), feat_dim, h, w), Ms.new_zeros(len(prob_sizes), max(prob_sizes), h, w)
        for i, _ in enumerate(fs_indexes):
            bid, start, end = fs_indexes[i][0], fs_indexes[i][1], fs_indexes[i][2]
            feat_len = end - start
            fs_feats[i, :feat_len] = Fs[bid, start: end]
            fs_priors[i, :feat_len] = Ms[bid, start: end]
        labels, prob_sizes, label_sizes = torch.LongTensor(labels), torch.LongTensor(prob_sizes), torch.LongTensor(label_sizes)

        # S2. classifier forward
        h0 = init_lstm_hidden(self.n_layers, fs_feats.size(0), self.hidden_size, device=Fs.device)
        logits, probs, _, betas = self.encoder(fs_feats, h0, fs_priors)
        logits, probs = logits.transpose(0, 1), probs.transpose(0, 1)

        if self.ctc_type == 'warp':
            l = self.ctc_loss(logits.cpu(), labels.type(torch.int).cpu(), prob_sizes.type(torch.int).cpu(), label_sizes.type(torch.int).cpu())
            # l = self.ctc_loss(logits, labels.type(torch.int), prob_sizes.type(torch.int), label_sizes.type(torch.int))

        elif self.ctc_type == 'builtin':
            l = self.ctc_loss(probs.clamp(min=1.0e-5, max=1-1.0e-5).log().cpu(), labels, prob_sizes, label_sizes)
        if abs(l.item()) > 1e5 or l.item() < 0:
            # set instable ctc loss to 0
            l = 0*l
        betas_unflat = [[] for _ in range(B)]
        for j in range(len(prob_sizes)):
            batch_id = fs_indexes[j][0]
            betas_unflat[batch_id].append(betas[j][:prob_sizes[j]])
        # S3. greedy decode
        preds = []
        probs = probs.transpose(1, 0).cpu().detach().numpy()
        probs_unflat = [[] for _ in range(B)]
        for j in range(fs_feats.size(0)):
            batch_id = fs_indexes[j][0]
            probs_unflat[batch_id].append(probs[j][:prob_sizes[j]])
            pred = self.decoder.greedy_decode(probs[j][:prob_sizes[j]], digit=digit)
            if not digit:
                pred = "".join(pred)
            preds.append(pred)

        # S4. retrieve ground-truth
        labels_flat = []
        start = 0
        labels, label_sizes = labels.tolist(), label_sizes.tolist()
        for j in range(len(label_sizes)):
            label_flat = list(labels[start: start+label_sizes[j]])
            if not digit:
                label_flat = "".join([self.int_to_char[x] for x in label_flat])
            labels_flat.append(label_flat)
            start = start + label_sizes[j]
        return l, preds, labels_flat, betas_unflat, probs_unflat

    def forward_reward(self, feats, priors, gt_coords, gt_chars, fs_mask, prop_coords, prop_scores):
        batch_size = feats.size(0)
        gt_coords[fs_mask.unsqueeze(dim=-1).expand(-1, -1, gt_coords.size(-1))!=1] = 0
        gt_coords = gt_coords.clamp(min=0, max=feats.size(1))
        roi_feats, roi_labels, batch_mask, prob_sizes = [], [], [], []
        prop_coords = prop_coords.type_as(gt_coords)
        with torch.no_grad():
            indexes = []
            for b in range(batch_size):
                roi_gt_ints = [[self.char_to_int[char] for char in chars] for chars in gt_chars[b]]
                non_zero_mask = gt_coords[b][:, -1] > 0
                non_zero_gt_coord = gt_coords[b][non_zero_mask][:, :2]
                if len(non_zero_gt_coord) == 0:
                    logging.info("No complete fingerspelling in ground-truth")
                    return feats.new_zeros(1)
                iou = get_iou(prop_coords[b], non_zero_gt_coord) # [N, K]
                iou, max_idx = iou.max(dim=-1)
                mask = (iou > self.iou_thr) & (prop_coords[b, :, 0] >= 0) & (prop_coords[b, :, 1] < feats.size(1))
                mask[self.n_top:] = False
                batch_mask.append(mask)
                batch_index, roi_coord, max_idx = (b+torch.zeros(mask.sum().item())).type_as(prop_coords), prop_coords[b][mask], max_idx[mask]
                indexes.append(torch.cat([batch_index.unsqueeze(dim=-1), roi_coord], dim=-1))
                for n in range(len(batch_index)):
                    prob_sizes.append(int(roi_coord[n, 1]-roi_coord[n, 0]))
                    gt_int = roi_gt_ints[max_idx[n]]
                    roi_labels.append(gt_int)
            if len(prob_sizes) == 0:
                return feats.new_zeros(1)
            mask, indexes = torch.cat(batch_mask), torch.cat(indexes).long().cpu().tolist()
            roi_feats, roi_priors = feats.new_zeros(len(prob_sizes), max(prob_sizes), feats.size(2), feats.size(3), feats.size(4)), priors.new_zeros(len(prob_sizes), max(prob_sizes), feats.size(3), feats.size(4))
            for i, (batch_id, start, end) in enumerate(indexes):
                roi_feats[i, :(end-start)] = feats[batch_id, start: end]
                roi_priors[i, :(end-start)] = priors[batch_id, start: end]
            # running recognition model
            h0 = init_lstm_hidden(self.n_layers, roi_feats.size(0), self.hidden_size, device=feats.device)
            roi_logits, roi_probs, _, _ = self.encoder(roi_feats, h0, roi_priors)
            roi_probs = roi_probs.cpu().numpy()
            roi_preds = []
            for b in range(len(roi_logits)):
                roi_preds.append(self.decoder.greedy_decode(roi_probs[b, :prob_sizes[b]], digit=True))
        accs = list(map(lambda x: 1-editdistance.distance(x[0], x[1])/len(x[1]), zip(roi_preds, roi_labels))) # [N1]

        scores = torch.zeros(len(mask)).type_as(prop_scores)
        scores[mask] = torch.FloatTensor(accs).type_as(prop_scores) # [B*N]

        # reinforce prob
        prop_scores = prop_scores.squeeze(dim=-1)
        scores = scores.view(batch_size, prop_scores.size(1))
        prop_probs = prop_scores / prop_scores.sum(dim=-1, keepdim=True).clamp(min=1e-5)
        scores = scores * torch.log(prop_probs.clamp(min=1e-5))
        acc_loss = -((scores * prop_probs).sum(dim=-1) / prop_probs.sum(dim=-1).clamp(min=1e-5)).mean()
        return acc_loss


class AttnEncoder(nn.Module):
    def __init__(self, hidden_size, attn_size, output_size, n_layers, prior_gamma, cell="LSTM"):
        super(AttnEncoder, self).__init__()
        self.cell_type = cell
        self.encoder_cell = AttnEncoderCell(hidden_size, attn_size, n_layers, prior_gamma, cell)
        self.lt = nn.Linear(hidden_size, output_size)
        self.classifier = nn.Softmax(dim=-1)

    def forward(self, Fs, h0, Ms=None):
        """
        Fs: (B, L, F, h, w), h0: (B, n_layer, n_hidden), Ms: (B, L, h, w)
        output: (B, L, V), (B, L, V), (B, F), (B, L, h, w)
        """
        if self.cell_type == 'LSTM':
            h0 = (h0[0].transpose(0, 1), h0[1].transpose(0, 1))
        else:
            h0 = h0.transpose(0, 1)
        fsz = list(Fs.size())
        Fs = Fs.view(*(fsz[:3]) + [-1]).transpose(1, 0).transpose(2, 3)
        if Ms is not None:
            Ms = Ms.view(*(fsz[:2] + [-1])).transpose(1, 0) # (L, B, h*w)
        ys, betas = [], []
        steps = Fs.size(0)
        h = h0
        for i in range(steps):
            Fi = Fs[i].transpose(0, 1).contiguous()
            if Ms is None:
                output, h, beta = self.encoder_cell(h, Fi)
            else:
                Mi = Ms[i].transpose(0, 1).contiguous()
                output, h, beta = self.encoder_cell(h, Fi, prior_map=Mi)
            ys.append(output)
            betas.append(beta)
        ys, betas = torch.stack(ys, 0), torch.stack(betas, 0)
        logits = self.lt(ys)
        probs = self.classifier(logits)
        betas = betas.transpose(0, 2).transpose(1, 2).contiguous()
        bsz, L_enc, hmap, wmap = fsz[0], fsz[1], fsz[3], fsz[4]
        betas = betas.view(bsz*L_enc, -1).view(bsz*L_enc, hmap, wmap).view(bsz, L_enc, hmap, wmap)
        logits, probs = logits.transpose(0, 1), probs.transpose(0, 1)
        return logits, probs, h, betas



class AttnEncoderCell(nn.Module):
    def __init__(self, hidden_size, attn_size, n_layers, prior_gamma, cell="LSTM"):
        super(AttnEncoderCell, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.attn_size = attn_size
        self.cell_type = cell
        if cell == "GRU":
            self.rnn_cell = nn.GRUCell(attn_size, hidden_size)
        elif cell == "LSTM":
            self.rnn_cell = nn.LSTMCell(attn_size, hidden_size)
        self.tanh = nn.Tanh()
        self.v = nn.Parameter(torch.Tensor(hidden_size, 1))
        self.Wa = nn.Parameter(torch.zeros(attn_size, hidden_size))
        self.Wh = nn.Parameter(torch.zeros(hidden_size, hidden_size))
        self.prior_gamma = prior_gamma
        # Initialization
        self.init_weight(self.v)

    def forward(self, hidden, attn, prior_map=None):
        # In: ([Layers, B, H], [Layers, B, H]), [N, B, A], [N, B, M], [N, B]
        # Out: [B, H], ([Layers, B, H], [Layers, B, H]), [N, B]
        prev_out = hidden[0][-1] if self.cell_type == "LSTM" else hidden[-1]
        N, B, A, H = attn.size()[0], attn.size()[1], attn.size()[2], self.hidden_size
        attn_weights = torch.matmul(attn.view(-1, A), self.Wa).view(N, B, H)+torch.matmul(prev_out, self.Wh)
        attn_weights = torch.matmul(self.tanh(attn_weights).view(-1, H), self.v).view(N, B)
        attn_weights = functional.softmax(attn_weights, dim=0)
        attn_weights = attn_weights*(prior_map.pow(self.prior_gamma)) if prior_map is not None else attn_weights
        s = (attn_weights.view(N, B, 1).repeat(1, 1, A) * attn).sum(dim=0)/attn_weights.sum(dim=0).view(B, 1).clamp(min=1.0e-5)  # [B, A]
        output = s
        hx, cx = [], []
        for i in range(self.n_layers):
            if self.cell_type == "GRU":
                h = self.rnn_cell(output, hidden[i])
                output = h
            else:
                h, c = self.rnn_cell(output, (hidden[0][i], hidden[1][i]))
                output = h
                cx.append(c)
            hx.append(h)
        hx = torch.stack(hx, 0)
        if self.cell_type == "GRU":
            return output, hx, attn_weights
        else:
            cx = torch.stack(cx, 0)
            return output, (hx, cx), attn_weights

    def init_weight(self, *args):
        for w in args:
            hin, hout = w.size()[0], w.size()[1]
            w.data.uniform_(-math.sqrt(6.0/(hin+hout)), math.sqrt(6.0/(hin+hout)))


def init_lstm_hidden(nlayer, batch_size, nhid, dtype=torch.float, device=torch.device('cuda')):
    return (torch.zeros((batch_size, nlayer, nhid), dtype=dtype, device=device),
            torch.zeros((batch_size, nlayer, nhid), dtype=dtype, device=device))

def get_iou(y_pred, y_true):
    # y_pred, y_true: [[start, end],...]
    # return: ndarray of shape (n_pred, n_true)
    n_pred, n_true = y_pred.size(0), y_true.size(0)
    y_pred = y_pred.view(n_pred, 1, 2).expand(-1, n_true, -1).contiguous().view(-1, 2)
    y_true = y_true.view(1, n_true, 2).expand(n_pred, -1, -1).contiguous().view(-1, 2)
    max_start, min_end = torch.max(y_pred[:, 0], y_true[:, 0]), torch.min(y_pred[:, 1], y_true[:, 1])
    min_start, max_end = torch.min(y_pred[:, 0], y_true[:, 0]), torch.max(y_pred[:, 1], y_true[:, 1])
    intersection = min_end - max_start + 1
    union = max_end - min_start + 1
    iou = (intersection / union).view(n_pred, n_true).clamp(min=0)
    iou = iou.view(n_pred, n_true)
    return iou
