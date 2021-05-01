import os
import cv2
import json
import pickle
import time
import tempfile
import shutil
import subprocess
import numpy as np
from collections import OrderedDict

def proc_raw(raw_dir, in_imnames, out_imnames, proc_dir):
    for i, _ in enumerate(in_imnames):
        raw_imname = os.path.join(raw_dir, in_imnames[i])
        proc_imname = os.path.join(proc_dir, out_imnames[i])
        shutil.copyfile(raw_imname, proc_imname)
    return

def extract_frames(filename, outfile):
    command = ["ffmpeg", "-i", filename, outfile]
    print(" ".join(command))
    subprocess.run(command, stdout = subprocess.PIPE, stderr = subprocess.STDOUT)
    return

def group_chunk(chunk_ids, output_rgb_path, output_opt_path, raw_rgb_path, raw_opt_path):
    decimal = 9
    suffix = '.png'
    output_paths = [output_rgb_path, output_opt_path]
    raw_video_paths = [raw_rgb_path, raw_opt_path]
    for i, _ in enumerate(raw_video_paths):
        tmp_dir = tempfile.mkdtemp()
        tmp_raw_dir, tmp_proc_dir = os.path.join(tmp_dir, 'raw'), os.path.join(tmp_dir, 'proc')
        os.makedirs(tmp_raw_dir, exist_ok=True)
        os.makedirs(tmp_proc_dir, exist_ok=True)
        raw_video_path, output_path = raw_video_paths[i], output_paths[i]
        extract_frames(raw_video_path, os.path.join(tmp_raw_dir, '%' + str(decimal) + 'd' + suffix))
        proc_imdirs = []
        for j, chunk_id in enumerate(chunk_ids):
            proc_fullpath = os.path.join(tmp_proc_dir, chunk_id)
            start, end = chunk_id.split('-')[-1].split('_')[0], chunk_id.split('-')[-1].split('_')[-1]
            start, end = int(start), int(end)
            in_imnames, out_imnames = [str(x).zfill(decimal)+suffix for x in range(start+1, end+1)], [str(x).zfill(decimal)+suffix for x in range(1, end-start+1)]
            os.makedirs(proc_fullpath, exist_ok=True)
            proc_raw(tmp_raw_dir, in_imnames, out_imnames, proc_fullpath)
            proc_imdirs.append(proc_fullpath)
        list_fn = os.path.join(tmp_dir, "list")
        with open(list_fn, "w") as fo:
            for proc_imdir in proc_imdirs:
                im_fulldir = os.path.join(proc_imdir, "%0"+str(decimal)+"d"+suffix)
                if "'" in im_fulldir:
                    # ' --> '\'' in filename
                    im_fulldir_ = im_fulldir.replace("'", "'\\''")
                else:
                    im_fulldir_ = im_fulldir
                fo.write("file " + "'" + im_fulldir_ + "'\n")
        output_path = output_paths[i]
        if os.path.isfile(output_path):
            # rm existing file
            os.remove(output_path)
        cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_fn, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", output_path]
        print(" ".join(cmd))
        pipe = subprocess.run(cmd, stdout = subprocess.PIPE, stderr = subprocess.STDOUT)
        shutil.rmtree(tmp_dir)
    return

def make_video(label_file, output_dir, rgb_dir, opt_dir, start=0, end=2**31):
    ss = json.load(open(label_file, 'r'))
    id_to_raw_paths = {}
    for item in ss:
        video_id = item['video_id']
        id_to_raw_paths[video_id] = (os.path.join(rgb_dir, video_id+'.mp4'), os.path.join(opt_dir, video_id+'.mp4'))

    scp_data = json.load(open(os.path.join(output_dir, 'scp'), 'r'))
    output_id_to_chunks = OrderedDict()
    for chunk_id, (_, output_rgb_video, output_opt_video) in scp_data.items():
        assert os.path.basename(output_rgb_video).split('-')[0] == os.path.basename(output_opt_video).split('-')[0]
        output_id = os.path.basename(output_rgb_video).split('-')[0]
        if output_id not in output_id_to_chunks:
            output_id_to_chunks[output_id] = []
        output_id_to_chunks[output_id].append([chunk_id, output_rgb_video, output_opt_video])
    for output_id, chunks in output_id_to_chunks.items():
        if int(output_id) >= start and int(output_id) < end:
            chunk_ids = [chunk[0] for chunk in chunks]
            output_rgb_video, output_opt_video = chunks[0][1], chunks[0][2]
            video_id = '-'.join(chunk_ids[0].split('-')[:-1])
            for chunk_id in chunk_ids:
                assert '-'.join(chunk_id.split('-')[:-1]) == video_id
            raw_rgb_path, raw_opt_path = id_to_raw_paths[video_id]
            group_chunk(chunk_ids, output_rgb_video, output_opt_video, raw_rgb_path, raw_opt_path)
    return

def make_scp(label_file, data_dir, N_per_file=40):
    os.makedirs(data_dir, exist_ok=True)
    scp_fn = os.path.join(data_dir, 'scp')
    id_to_part = OrderedDict()
    video_to_chunk_ids = OrderedDict()
    ss = json.load(open(label_file, 'r'))
    for item in ss:
        video_id = item['video_id']
        chunk_id = video_id + '-' + str(item['frames'][0][0]) + '_' + str(item['frames'][0][1])
        if video_id not in video_to_chunk_ids:
            video_to_chunk_ids[video_id] = []
        video_to_chunk_ids[video_id].append(chunk_id)
    data_id = 0
    for video_id, chunk_ids in video_to_chunk_ids.items():
        for i in range(0, len(chunk_ids), N_per_file):
            data_fn = [os.path.join(data_dir, str(data_id) + '-rgb.mp4'), os.path.join(data_dir, str(data_id) + '-opt.mp4')]
            for j, chunk in enumerate(chunk_ids[i: i+N_per_file]):
                id_to_part[chunk] = [j] + data_fn
            data_id += 1
    json.dump(id_to_part, open(scp_fn, 'w'))
    return
