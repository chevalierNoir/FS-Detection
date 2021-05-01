import os,glob,shutil,tempfile,subprocess,json
import cv2
import numpy as np
import pandas as pd
from collections import defaultdict

def resize(raw_dir, resize_dir, csv_fn, start=0, end=2**31):
    os.makedirs(resize_dir, exist_ok=True)
    df = pd.read_csv(csv_fn)
    yids = sorted(list(set(list(df['yid']))))
    print(f"{len(yids)} videos in total")
    for i_yid, yid in enumerate(yids):
        if not (i_yid >= start and i_yid < end):
            continue
        video_fn = glob.glob(os.path.join(raw_dir, yid+'*'))[0]
        print(f"{video_fn}")
        resize_fn = os.path.join(resize_dir, yid+'.mp4')
        cmd = 'ffmpeg -y -i ' + video_fn + ' -vf "[in]scale=iw*min(640/iw\,360/ih):ih*min(640/iw\,360/ih)[scaled]; [scaled]pad=640:360:(640-iw*min(640/iw\,360/ih))/2:(360-ih*min(640/iw\,360/ih))/2[padded]; [padded]setsar=1:1[out]" -c:v libx264 -crf 20 -c:a copy ' + resize_fn
        subprocess.call(cmd, shell=True)
    return

def get_opt(rgb_dir, opt_dir, csv_fn, scale_ratio=0.5, start=0, end=2**31):
    df = pd.read_csv(csv_fn)
    yids = sorted(list(set(list(df['yid']))))
    os.makedirs(opt_dir, exist_ok=True)
    print(f"{len(yids)} videos in total")
    for i_yid, yid in enumerate(yids):
        if not (i_yid >= start and i_yid < end):
            continue
        tmp_dir = tempfile.mkdtemp()
        rgb_fn = glob.glob(os.path.join(rgb_dir, yid+'*'))[0]
        opt_fn = os.path.join(opt_dir, yid+'.mp4')
        print(f"Optical flow: {opt_fn}")
        prv_gray = None
        # imgs = []
        cap = cv2.VideoCapture(rgb_fn)
        W, H = int(cap.get(3)), int(cap.get(4))
        i_frame = 0
        while True:
            ret, frame = cap.read()
            if ret:
                if scale_ratio != 1:
                    frame = cv2.resize(frame, (int(W*scale_ratio), int(H*scale_ratio)))
                cur_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if prv_gray is not None:
                    flow = cv2.calcOpticalFlowFarneback(prv_gray, cur_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                    mag = (255.0*(mag-mag.min())/max(float(mag.max()-mag.min()), 1)).astype(np.uint8)
                    mag = cv2.resize(mag, (W, H))
                else:
                    mag = np.zeros((H, W), dtype=np.uint8)
                cv2.imwrite(os.path.join(tmp_dir, str(i_frame+1).zfill(9)+'.png'), mag)
                prv_gray = cur_gray
                i_frame += 1
            else:
                break
        cmd = "ffmpeg -y -i " + os.path.join(tmp_dir, '%09d.png') + ' -c:v libx264 -pix_fmt yuv420p ' + opt_fn
        print(cmd)
        subprocess.call(cmd, shell=True)
        shutil.rmtree(tmp_dir)
    return

def resample(raw_dir, csv_fn, start=0, end=2**31):
    df = pd.read_csv(csv_fn)
    yids = sorted(list(set(list(df['yid']))))
    for i_yid, yid in enumerate(yids):
        if not (i_yid >= start and i_yid < end):
            continue
        video_fn = glob.glob(os.path.join(raw_dir, yid+'*'))[0]
        res_dir = tempfile.mkdtemp()
        res_fn = os.path.join(res_dir, yid+'.mp4')
        cmd = "ffmpeg -y -i " + video_fn + " -filter:v fps=29.97 " + res_fn
        print(cmd)
        subprocess.call(cmd, shell=True)
        shutil.copyfile(res_fn, video_fn)
        shutil.rmtree(res_dir)
    return

def merge_csv(input_csvs, output_csv):
    keys = pd.read_csv(input_csvs[0], index_col=0).keys()
    df_all = {key: [] for key in keys}
    for csv_fn in input_csvs:
        df = pd.read_csv(csv_fn, index_col=0)
        for key in df.keys():
            df_all[key].extend(list(df[key]))
    pd.DataFrame(df_all).to_csv(output_csv)
    print(f"Write csv file {output_csv}")
    return
