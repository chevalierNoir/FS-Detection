import os
import sys
import glob
import torch
import configargparse
import logging
import utils
import roiloader
import convnet
import yaml
import numpy as np
import torch.utils.data as tud
from torchvision import transforms
from sampler import BucketBatchSampler
from evaluate import eval_pred
from metrics import ap_iou

parser = configargparse.ArgumentParser(
    description="Train Detector",
    config_file_parser_class=configargparse.YAMLConfigFileParser,
    formatter_class=configargparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--train_label', type=str, help='train json file')
parser.add_argument('--dev_label', type=str, help='dev json file')
parser.add_argument('--train_scp', type=str, help='train scp file')
parser.add_argument('--dev_scp', type=str, help='dev scp file')
parser.add_argument('--train_pose', type=str, help='train pose file')
parser.add_argument('--dev_pose', type=str, help='dev pose file')
parser.add_argument('--bbox_file', type=str, help='bbox file')
parser.add_argument('--det_sample_rate', type=int, help='loader sample rate')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--pose_mode', type=str, default=None, help='pose mode')
parser.add_argument('--ctc_type', type=str, default='warp', help='ctc type (builtin|warp)')
parser.add_argument("--char_list", type=str, default="_ '&.@acbedgfihkjmlonqpsrutwvyxz", help="char list")
parser.add_argument('--det_coef', type=float, default=1.0, help='det coef')
parser.add_argument('--fsr_coef', type=float, default=1.0, help='fsr coef')
parser.add_argument('--pose_coef', type=float, default=1.0, help='pose coef')
parser.add_argument('--optim', type=str, default='SGD', help='optimizer (SGD|Adam)')
parser.add_argument('--lr', type=float, default='0.01', help='learning rate')
parser.add_argument('--dropout', type=float, default=0.0, help='dropout')
parser.add_argument('--clip', type=float, default=50, help='clip grad')
parser.add_argument('--epoch', type=int, default=10, help='training epochs')
parser.add_argument('--info_interval', type=int, default=100, help='print intervals')
parser.add_argument('--det_interval', type=int, default=1000, help='saving interval for detector')
parser.add_argument('--output', type=str, help='output dir')
# parser.add_argument('--cont', action='store_true', help='resume training from checkpoint')
parser.add_argument('--image_scale', type=float, default=0.3, help='image scale')
parser.add_argument('--immean', type=float, nargs='+', default=[0.485, 0.456, 0.406], help='image mean')
parser.add_argument('--imstd', type=float, nargs='+', default=[0.229, 0.224, 0.225], help='image std')
parser.add_argument('--pose_sample_rate', type=int, default=5, help='pose sample rate')
parser.add_argument('--path_pose', type=str, default=None, help='path to pre-trained pose model')
parser.add_argument('--reward_coef', type=float, default=1, help='coef on reward loss')
parser.add_argument('--reward_iou_thr', type=float, default=0.5, help='IoU threshold for rewarding')
parser.add_argument("--stage", type=int, default=1, help="stage index")
parser.add_argument("--amp", type=int, default=0, help="amp training")
args = parser.parse_args()

torch.manual_seed(222)
torch.cuda.manual_seed_all(222)
np.random.seed(222)

logging.basicConfig(level=logging.INFO, format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s')

if not os.path.isdir(args.output):
    print("Make dir %s" % (args.output))
    os.makedirs(args.output)

latest_model_path = os.path.join(args.output, 'latest.pth')
det_path = os.path.join(args.output, 'det.pth')
best_dev_path = os.path.join(args.output, 'best-dev.pth')
hyp_path = os.path.join(args.output, 'hyp.pth')
log_file = os.path.join(args.output, 'log')

with open(os.path.join(args.output, 'train_conf.yaml'), 'w') as fo:
    yaml.dump({**vars(args), **{'best_dev_path': best_dev_path}}, fo)

loader_seeds = list(range(100))
device = torch.device('cuda')
toTensor = roiloader.ToTensor(device)
normalize = roiloader.Normalize(args.immean, args.imstd, device)
assert args.stage in {1, 2}, f"Option for stage: 1, 2"
encoder = convnet.convNet(char_list=args.char_list, rd_iou_thr=args.reward_iou_thr, num_concat=args.stage).to(device)
map_size = utils.get_map_size(encoder.features, args.train_scp, args.image_scale)

logging.info(f"{encoder}")

train_data = roiloader.VideoData(args.train_label, args.train_scp, args.train_pose, transform=transforms.Compose([toTensor, normalize]), bbox_file=args.bbox_file, image_scale=args.image_scale, det_sample_rate=args.det_sample_rate, pose_sample_rate=args.pose_sample_rate, sigma=1, fmap_wh=map_size)
train_loader = tud.DataLoader(train_data, batch_sampler=BucketBatchSampler(shuffle=True, batch_size=args.batch_size, files=train_data.rgb_videos, seeds=loader_seeds), collate_fn=train_data.collate_fn)
dev_data = roiloader.VideoData(args.dev_label, args.dev_scp, args.dev_pose, transform=transforms.Compose([toTensor, normalize]), bbox_file=args.bbox_file, image_scale=args.image_scale, det_sample_rate=args.det_sample_rate, pose_sample_rate=args.pose_sample_rate, sigma=1, fmap_wh=map_size)
dev_loader = tud.DataLoader(dev_data, batch_sampler=BucketBatchSampler(shuffle=False, batch_size=args.batch_size, files=dev_data.rgb_videos, cycle=False), collate_fn=dev_data.collate_fn)

logging.info('Train: %d, dev: %d' % (len(train_data), len(dev_data)))

optimizer = getattr(torch.optim, args.optim)(encoder.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, min_lr=1.0e-8, verbose=True)

if args.amp == 1:
    from apex import amp
    logging.info(f"AMP training, opt level: O1")
    encoder, optimizer = amp.initialize(encoder, optimizer, opt_level="O1")

hyp = {'epoch': 0, 'step': 0, 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'sampler': train_loader.batch_sampler.state_dict(), 'best_dev_metric': -float('inf')}

if args.amp == 1:
    hyp['amp'] = amp.state_dict()

if os.path.isfile(latest_model_path) and os.path.isfile(hyp_path):
    logging.info("Load from checkpoint: %s and %s" % (latest_model_path, hyp_path))
    encoder.load_state_dict(torch.load(latest_model_path))
    hyp = torch.load(hyp_path)
    optimizer.load_state_dict(hyp['optimizer'])
    scheduler.load_state_dict(hyp['scheduler'])
    train_loader.batch_sampler.load_state_dict(hyp['sampler'])
    if args.amp == 1:
        amp.load_state_dict(hyp['amp'])

def main():
    global hyp
    for epoch in range(args.epoch):
        if epoch < hyp['epoch']:
            continue
        logging.info(f"Epoch {epoch}")
        encoder.train()
        ls, rpn_cls_ls, rpn_reg_ls, rcnn_cls_ls, rcnn_reg_ls, pose_ls, reward_ls = [], [], [], [], [], [], []
        fsr_preds, fsr_labels = [], []
        for i_batch, sample in enumerate(train_loader):
            optimizer.zero_grad()
            rpn_cls_loss, rpn_reg_loss, rcnn_cls_loss, rcnn_reg_loss, fsr_pred, fsr_label, pose_loss, reward_loss = encoder(sample, training=True, pose_on=(args.pose_coef!=0), fsr_on=(args.fsr_coef!=0), pose_sample_rate=args.pose_sample_rate, reward_on=(args.reward_coef!=0), stage=args.stage)
            l = args.det_coef * (rpn_cls_loss + rpn_reg_loss) + args.fsr_coef * (rcnn_cls_loss + rcnn_reg_loss)  + args.pose_coef * pose_loss + args.reward_coef * reward_loss
            if args.amp == 1:
                with amp.scale_loss(l, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.clip)
            else:
                l.backward()
                torch.nn.utils.clip_grad_norm(encoder.parameters(), args.clip)
            optimizer.step()
            ls.append(l.item())
            rpn_cls_ls.append(rpn_cls_loss.item())
            rpn_reg_ls.append(rpn_reg_loss.item())
            rcnn_cls_ls.append(rcnn_cls_loss.item())
            rcnn_reg_ls.append(rcnn_reg_loss.item())
            pose_ls.append(pose_loss.item())
            reward_ls.append(reward_loss.item())
            hyp['step'] += 1
            # logging.info(f"{l}")
            if hyp['step'] % args.info_interval == 0:
                mean_loss = sum(ls) / len(ls)
                mean_rpn_cls, mean_rpn_reg, mean_rcnn_cls, mean_rcnn_reg, mean_pose_loss, mean_reward_loss = sum(rpn_cls_ls) / len(rpn_cls_ls), sum(rpn_reg_ls) / len(rpn_reg_ls), sum(rcnn_cls_ls) / len(rcnn_cls_ls), sum(rcnn_reg_ls) / len(rcnn_reg_ls), sum(pose_ls) / len(pose_ls), sum(reward_ls)/len(reward_ls)
                pcont = 'Epoch %d, Step %d, train loss: %.3f, rpn-cls: %.3f, rpn-reg: %.3f, fsr-ctc-loss: %.3f, pose loss: %.3f, fsr-rec-loss: %.3f' % (hyp['epoch'], hyp['step'], mean_loss, mean_rpn_cls, mean_rpn_reg, mean_rcnn_cls, mean_pose_loss, mean_reward_loss)
                logging.info(pcont)
                open(log_file, 'a+').write(pcont+'\n')
                torch.save(encoder.state_dict(), open(latest_model_path, 'wb'))
                if args.amp == 1:
                    hyp['amp'] = amp.state_dict()
                torch.save(hyp, open(hyp_path, 'wb'))
                ls, rpn_cls_ls, rpn_reg_ls, rcnn_cls_ls, rcnn_reg_ls, pose_ls = [], [], [], [], [], []
            if hyp['step'] % args.det_interval == 0:
                dev_out = os.path.join(args.output, 'dev-proposal.pkl')
                eval_pred(encoder, dev_loader, dev_out, top_k=50, stage=args.stage)
                iou2mAP = ap_iou.get_ap_iou(dev_out, iou_thrs=[0.5])
                mAP = iou2mAP[0.5]
                if mAP > hyp['best_dev_metric']:
                    torch.save(encoder.state_dict(), open(best_dev_path, 'wb'))
                    hyp['best_dev_metric'] = mAP
                tmp_fns = glob.glob(os.path.join(args.output, 'dev-proposal*'))
                for tmp_fn in tmp_fns:
                    os.remove(tmp_fn)
                scheduler.step(mAP)
                pcont = 'Epoch %d, step %d, dev AP %.3f' % (epoch, hyp['step'], mAP)
                logging.info(pcont)
                open(log_file, 'a+').write(pcont+'\n')
                encoder.train()
        hyp['epoch'] += 1
    return

if __name__ == '__main__':
    main()
