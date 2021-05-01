import argparse,os,glob
from download import *
from process import *
from make_json import *
from make_video_loader import *

def main():
    parser = argparse.ArgumentParser(description='processing pipeline')
    parser.add_argument('--data', type=str, help='data dir')
    parser.add_argument('--set', type=str, help='set name', choices=['ChicagoFSWild', 'ChicagoFSWildPlus'])
    parser.add_argument('--step', type=int, help='step')
    parser.add_argument('--job', type=int, help='job id')
    args = parser.parse_args()
    root_dir = os.path.join(os.getcwd(), args.data)
    csv_origin, csv_ready = os.path.join(root_dir, args.set+'.csv'), os.path.join(root_dir, args.set+'-ready.csv')
    assert os.path.isfile(csv_origin), f"{csv_origin} not found."
    raw_video, raw_pose, resize_video, opt_video, tmp_dir, loader_dir = os.path.join(root_dir, 'raw-video'), os.path.join(root_dir, 'raw-pose'), os.path.join(root_dir, 'rgb-640x360'), os.path.join(root_dir, 'opt-640x360'), os.path.join(root_dir, 'tmp'), os.path.join(root_dir, 'loader')
    job_option = {'start': args.job, 'end': args.job+1} if args.job is not None else {}
    splits = ["dev", "test", "train"]
    label_fns = [os.path.join(loader_dir, sp+'.json') for sp in splits]
    label_option = {'win_size': 300, 'stride': 250}
    os.makedirs(tmp_dir, exist_ok=True)
    if args.step == 1:
        print(f"Download videos {args.set}")
        jid = args.job if args.job is not None else 0
        csv_proc = os.path.join(tmp_dir, args.set+'.csv.'+str(jid))
        download(csv_origin, csv_proc, raw_video, **job_option)
    elif args.step == 2:
        print(f"Create csv file for downloaded videos")
        input_csvs = glob.glob(os.path.join(tmp_dir, args.set+'.csv*'))
        merge_csv(input_csvs, csv_ready)
    elif args.step == 3:
        print(f"Resize video to 640x360")
        resize(raw_video, resize_video, csv_ready, **job_option)
        if args.set == 'ChicagoFSWildPlus':
            print(f"Change fps to 29.97")
            resample(resize_video, csv_ready, **job_option)
    elif args.step == 4:
        print(f"Generate optical flow for resized videos")
        get_opt(resize_video, opt_video, csv_ready, **job_option)
    elif args.step == 5:
        print(f"Generate label files")
        os.makedirs(loader_dir, exist_ok=True)
        for split, label_fn in zip(splits, label_fns):
            generate_roidb(csv_ready, label_fn, video_dir=resize_video, split=split, **label_option)
            make_scp(label_fn, os.path.join(loader_dir, 'video', split))
    elif args.step == 6:
        print(f"Generate video loaders")
        for split, label_fn in zip(splits, label_fns):
            make_video(label_fn, os.path.join(loader_dir, 'video', split), rgb_dir=resize_video, opt_dir=opt_video, **job_option)
    else:
        raise ValueError("Step: 1-6")
    return

if __name__ == '__main__':
    main()
