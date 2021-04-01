from metrics import msa_acc
from metrics import ap_iou

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='MSA, AP@Acc', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--pred", '-p', type=str, help="prediction file")
    parser.add_argument("--model", '-m', type=str, help="rec model path")
    parser.add_argument("--label", '-l', type=str, help="label file")
    parser.add_argument("--scp", '-s', type=str, help="scp file")
    parser.add_argument("--type", '-t', type=str, default='iou')
    parser.add_argument("--ptype", '-pt', type=str, default='coco', help="type of recall interval")
    args = parser.parse_args()

    rec_file = args.pred + '.rec'
    if args.type == 'rec':
        msa_acc.get_rec_dict(args.pred, rec_file, args.model, args.label, args.scp)
    elif args.type == 'acc':
        msa_acc.get_ap_ler(rec_file, ler_thrs=[0.0, 0.1, 0.2, 0.3, 0.4], iou_thrs=[0.0, 0.0, 0.0, 0.0, 0.0], ordering=True, ptype=args.ptype)
    elif args.type == 'msa':
        msa_acc.get_seq_ler_vars(rec_file, nms_thresh=0.01)
    elif args.type == 'iou':
        ap_iou.get_ap_iou(args.pred, iou_thrs=[0.1, 0.2, 0.3, 0.4, 0.5], ptype=args.ptype)
