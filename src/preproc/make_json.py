import os
import glob
import json
from datetime import datetime
import numpy as np
import pandas as pd
from collections import defaultdict
from subprocess import Popen, PIPE

def parse_csv(csv_file, split, video_dir):
    df = pd.read_csv(csv_file)
    segment = defaultdict(list) # video: [[start, end, label]]
    vinfo = {}
    for row_id, row in df.iterrows():
        if row['partition'] == split:
            video_id = row['yid']
            if video_id not in vinfo:
                video_path = glob.glob(os.path.join(video_dir, video_id+'*'))[0]
                cmd = "ffprobe -v error -select_streams v -of default=noprint_wrappers=1:nokey=1 -show_entries stream=r_frame_rate " + video_path
                stdout, stderr = Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True).communicate()
                fps = stdout.decode('utf-8').strip()
                if '/' in fps:
                    fps = float(fps.split('/')[0]) / float(fps.split('/')[1])
                else:
                    fps = float(fps)
                cmd = "ffprobe -v error -select_streams v:0 -show_entries stream=nb_frames -of default=nokey=1:noprint_wrappers=1 " + video_path
                stdout, stderr = Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True).communicate()                
                vlen = int(stdout.decode('utf-8').strip())
                vinfo[video_id] = (fps, vlen)
            obj = datetime.strptime(row['start_time'], '%H:%M:%S.%f')
            start_time = obj.hour*3600 + obj.minute*60+obj.second+obj.microsecond/(10**6)
            start_frame = int(start_time * vinfo[video_id][0])
            end_frame = start_frame + row['number_of_frames']
            fsid, label = row['filename'], row['label_proc']
            segment[video_id].append([start_frame, end_frame, fsid, label])
    return segment, vinfo


def generate_roi(rois, fsids, fslabels, overlaps, start, end, video_id):
    tmp = {}
    tmp['wins'] = rois[:,:2] - start
    tmp['durations'] = tmp['wins'][:,1] - tmp['wins'][:,0]
    tmp['frames'] = np.array([[start, end]])
    tmp['video_id'] = video_id
    mask = overlaps >= 0.99
    tmp['fs_mask'] = mask 
    tmp['fs_id'] = fsids[mask].tolist()
    tmp['fs_label'] = fslabels[mask].tolist()
    assert tmp['wins'][:, -1].max() <= end - start
    return tmp


def generate_roidb(csv_file, out_file, video_dir, split, win_size=300, stride=75, min_length=1, overlap_thr=0.1):
    vsegs, vinfo = parse_csv(csv_file, split=split, video_dir=video_dir)
    roidb = []
    for video_id, segments in vsegs.items():
        fps, vlen = vinfo[video_id][0], vinfo[video_id][1]
        db, lb = [segment[:2] for segment in segments], [segment[2:] for segment in segments]
        db, lb = np.array(db).astype(np.int32), np.array(lb)
        for start in range(0, max(1, vlen-win_size), stride):
            end = min(start + win_size, vlen)
            nonzero_idx = np.logical_not(np.logical_or(db[:, 0] >= end, db[:, 1] <= start))
            rois, labels = db[nonzero_idx], lb[nonzero_idx]
            # remove duration < min_length
            if len(rois) > 0:
                duration = rois[:, 1] - rois[:, 0]
                dur_idx = duration >= min_length
                rois, labels = rois[dur_idx], labels[dur_idx]
            # remove overlap < overlap_thr
            if len(rois) > 0:
                dur_in_win = np.minimum(rois[:, 1], end) - np.maximum(rois[:, 0], start)
                overlap = dur_in_win / (rois[:, 1] - rois[:, 0])
                overlap_idx = overlap >= overlap_thr
                rois, labels, overlaps = rois[overlap_idx], labels[overlap_idx], overlap[overlap_idx]
            if len(rois) > 0:
                rois[:, 0] = np.maximum(rois[:, 0], start)
                rois[:, 1] = np.minimum(rois[:, 1], end)
                fsids, fslabels = labels[:, 0], labels[:, 1]
                roi_label = generate_roi(rois, fsids, fslabels, overlaps, start, end, video_id)
                roidb.append(roi_label)
    for i in range(len(roidb)):
        for key, value in roidb[i].items():
            if isinstance(value, np.ndarray):
                value = value.tolist()
                roidb[i][key] = value
    print(f"{len(roidb)} instances -> {out_file}")
    json.dump(roidb, open(out_file, 'w'))
    return roidb
