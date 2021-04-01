import os
import logging
import pickle
import torch
import time
import yaml
import configargparse
import roiloader
import convnet
import utils
import torch.utils.data as tud
import numpy as np
from collections import defaultdict
from torchvision import transforms
from ctc_decoder import Decoder
from sampler import BucketBatchSampler

def eval_pred(encoder, loader, output_file, top_k, stage):
    if os.path.isfile(output_file):
        print(f'{output_file} exists')
        return
    output_dir = "/".join(output_file.split("/")[:-1])
    os.makedirs(output_dir, exist_ok=True)
    encoder.eval()
    preds, grts, ids = [], [], []
    for i_batch, sample in enumerate(loader):
        with torch.no_grad():
            prob, pred = encoder(sample, training=False, fsr_on=False, pose_on=False, stage=stage)

            prob, pred = prob[:, :, 0].cpu().numpy(), np.floor(pred.cpu().numpy())
            batch_size = prob.shape[0]
            video_length = sample['img_global'].size(1)
            mask = np.logical_and(pred[:, :, 0] <= pred[:, :, 1], pred[:, :, 0] >=0, pred[:, :, 1] < video_length)
            for b in range(batch_size):
                prob_, pred_ = prob[b][mask[b]], pred[b][mask[b]]
                idx = np.argsort(prob_)[-top_k:][::-1]
                prob_, pred_ = prob_[idx], pred_[idx]
                pred_ = np.concatenate((pred_, np.expand_dims(prob_, axis=-1)), axis=-1)
                pred_[:, :-1] = pred_[:, :-1] * loader.dataset.det_sample_rate
                preds.append(pred_)
                label_ = sample['det_label_origin'][b]
                grt = label_[label_[:, -1] != 0][:, :-1].cpu().numpy()
                grts.append(grt)
                ids.append(sample['id'][b])
                print(sample['id'][b])
    pickle.dump({'grt': grts, 'pred': preds, 'id': ids}, open(output_file, 'wb'))
    return

def eval_bbox(encoder, loader, output_fn):
    os.makedirs(os.path.dirname(output_fn), exist_ok=True)
    if os.path.isfile(output_fn):
        bbox_dict = pickle.load(open(output_fn, 'rb'))
    else:
        bbox_dict = {}
    encoder.eval()
    for i_batch, sample in enumerate(loader):
        with torch.no_grad():
            betas = encoder.get_beta(sample)
            batch_size = len(betas)
            for b in range(batch_size):
                bboxes = utils.get_det_bbox(betas[b].cpu().numpy(), W=loader.dataset.raw_width, H=loader.dataset.raw_height)
                bbox_dict[sample['id'][b]] = bboxes
                print(sample['id'][b])
    pickle.dump(bbox_dict, open(output_fn, 'wb'))
    return


def main():
    parser = configargparse.ArgumentParser(
        description="Evaluate Detector",
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', is_config_file=False, help='config file path')
    parser.add_argument("--eval_scp", type=str, help="scp file")
    parser.add_argument("--eval_label", type=str, help="label file")
    parser.add_argument("--eval_pose", type=str, help="pose file")
    parser.add_argument("--output_fn", type=str, help="output file")
    parser.add_argument("--top_k", type=int, default=50, help="top k")
    parser.add_argument("--eval_type", type=str, default="pred", help="eval type")
    args, _ = parser.parse_known_args()
    rem_args = yaml.safe_load(open(args.config, 'r'))
    parser.set_defaults(**rem_args)
    args, _ = parser.parse_known_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s')

    device = torch.device('cuda')
    encoder = convnet.convNet(char_list=args.char_list, rd_iou_thr=args.reward_iou_thr, num_concat=args.stage).to(device)
    toTensor = roiloader.ToTensor(device)
    normalize = roiloader.Normalize(args.immean, args.imstd, device)
    map_size = utils.get_map_size(encoder.features, args.train_scp, args.image_scale)

    eval_data = roiloader.VideoData(args.eval_label, args.eval_scp, pose_file=None, transform=transforms.Compose([toTensor, normalize]), bbox_file=args.bbox_file, image_scale=args.image_scale, det_sample_rate=args.det_sample_rate, pose_sample_rate=args.pose_sample_rate, sigma=1, fmap_wh=map_size)
    eval_loader = tud.DataLoader(eval_data, batch_sampler=BucketBatchSampler(shuffle=False, batch_size=args.batch_size, files=eval_data.rgb_videos, cycle=False), collate_fn=eval_data.collate_fn)
    print('Eval data: %d' % (len(eval_data)))

    print("Load checkpoint: %s" % (args.best_dev_path))
    encoder.load_state_dict(torch.load(args.best_dev_path))

    if args.eval_type == 'pred':
        eval_pred(encoder, eval_loader, args.output_fn, args.top_k, stage=args.stage)
    elif args.eval_type == 'bbox':
        eval_bbox(encoder, eval_loader, args.output_fn)
    return

if __name__ == '__main__':
    main()
