import os
import json
import cv2
import time
import utils
import torch
import pickle
import os.path as osp
import numpy as np
import torch.utils.data as tud


class VideoData(tud.Dataset):
    def __init__(self, label_file, scp_file, pose_file, transform, max_num_wins=30, chunk_size=300, start_end=(0, 1e6), bbox_file=None, image_scale=1, sigma=1, det_sample_rate=1, pose_sample_rate=1, fmap_wh=(12, 12)):
        self.max_num_wins = max_num_wins
        self.chunk_size = chunk_size
        self.transform = transform
        self._parse(label_file, scp_file, pose_file, start_end, bbox_file)
        self.open_rgb, self.open_opt, self.open_pose = None, None, None
        self.image_scale = image_scale
        self.sigma, self.det_sample_rate, self.pose_sample_rate, self.fmap_wh = sigma, det_sample_rate, pose_sample_rate, fmap_wh
        print(f'Image scale: {image_scale} with center cropping')
        print(f"Det sample rate: {det_sample_rate}")

    def __getitem__(self, idx):
        chunk_id, rgb_video, opt_video, pose_json = self.chunk_ids[idx], self.rgb_videos[idx], self.opt_videos[idx], self.pose_data[idx]
        if pose_json != self.open_pose:
            self.open_pose = pose_json
            pose_data = json.load(open(pose_json, 'r'))
            self.handpose_buffer, self.bodypose_buffer, self.openpose_origin_wh = pose_data["hand_keypoints"], pose_data["body_keypoints"], pose_data["wh"]

        if rgb_video != self.open_rgb:
            self.open_rgb = rgb_video
            self.rgb_buffer = self._load_video(rgb_video, as_gray=False, scale=self.image_scale, bboxes=self.rgb_to_bbox[rgb_video])

        if opt_video != self.open_opt:
            self.open_opt = opt_video
            self.opt_buffer = self._load_video(opt_video, as_gray=True, scale=self.image_scale, bboxes=self.opt_to_bbox[opt_video])

        subid = self.subids[idx]
        bbox = self.bboxes[idx]
        opts_global, opts_local = self.opt_buffer['global'][self.chunk_size*subid: self.chunk_size*(subid+1)][::self.det_sample_rate], self.opt_buffer['local'][self.chunk_size*subid: self.chunk_size*(subid+1)][::self.det_sample_rate]
        imgs_global, imgs_local = self.rgb_buffer['global'][self.chunk_size*subid: self.chunk_size*(subid+1)][::self.det_sample_rate], self.rgb_buffer['local'][self.chunk_size*subid: self.chunk_size*(subid+1)][::self.det_sample_rate]
        imgs_global, imgs_local = imgs_global.transpose((0, 3, 1, 2)), imgs_local.transpose((0, 3, 1, 2))
        det_label = np.zeros((self.max_num_wins, 3), dtype=np.int32)
        det_label_ = np.array(self.det_labels[idx], dtype=np.int32)
        det_label[:len(det_label_), :] = det_label_
        det_label_origin = det_label.copy()
        det_label[:, :-1] = det_label[:, :-1] / self.det_sample_rate
        det_label[:len(det_label_), 1] = np.maximum(det_label[:len(det_label_), 0] + 1, det_label[:len(det_label_), 1])
        fs_label = self.fs_labels[idx]
        fs_mask = -np.ones(self.max_num_wins, dtype=np.float32)
        fs_mask[:len(self.fs_masks[idx])][self.fs_masks[idx]] = 1
        fs_id = self.fs_ids[idx]
        # print(chunk_id, fs_id, subid, self.chunk_size, len(self.bodypose_buffer), len(self.handpose_buffer), self.det_sample_rate)
        if self.open_pose is not None:
            body_kps, hand_kps = self.bodypose_buffer[self.chunk_size*subid: self.chunk_size*(subid+1)][::self.det_sample_rate], self.handpose_buffer[self.chunk_size*subid: self.chunk_size*(subid+1)][::self.det_sample_rate]
            heatmap_global, mask_global = self.get_heatmap(body_kps, hand_kps, w_img=self.openpose_origin_wh[0], h_img=self.openpose_origin_wh[1], imgs=imgs_global)
            if bbox is None:
                heatmap_local, mask_local = heatmap_global, mask_global
            else:
                heatmap_local, mask_local = self.get_heatmap(body_kps, hand_kps, w_img=self.openpose_origin_wh[0], h_img=self.openpose_origin_wh[1], imgs=imgs_local, bbox=bbox[::self.det_sample_rate])
        else:
            heatmap_local, mask_local, heatmap_global, mask_global = None, None, None, None
        sample = {'img_local': imgs_local, 'img_global': imgs_global, 'opt_local': opts_local, 'opt_global': opts_global, 'det_label': det_label, 'fs_label': fs_label, 'fs_mask': fs_mask, 'id': chunk_id, 'fs_id': fs_id, 'bbox': bbox, 'pose_heatmap_local': heatmap_local, 'pose_heatmap_global': heatmap_global, 'pose_mask_local': mask_local, 'pose_mask_global': mask_global, 'det_label_origin': det_label_origin}
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def collate_fn(self, data):
        sample = {}
        for key in ['_local', '_global']:

            sample['img'+key] = torch.stack([datum['img'+key] for datum in data])
            sample['opt'+key] = torch.stack([datum['opt'+key] for datum in data])

        sample['det_label'], sample['det_label_origin'] = torch.stack([datum['det_label'] for datum in data]), torch.stack([datum['det_label_origin'] for datum in data])
        sample['fs_label'] = [datum['fs_label'] for datum in data]
        sample['fs_mask'] = torch.stack([datum['fs_mask'] for datum in data])
        sample['id'] = [datum['id'] for datum in data]
        sample['fs_id'] = [datum['fs_id'] for datum in data]
        sample['bbox'] = [datum['bbox'] for datum in data]
        for key in ['_local', '_global']:
            if data[0]['pose_heatmap' + key] is not None:
                # pad
                B, lens, h, w, N = len(data), [datum['pose_heatmap'+key].shape[0] for datum in data], data[0]['pose_heatmap'+key].shape[1], data[0]['pose_heatmap'+key].shape[2], data[0]['pose_heatmap'+key].shape[3]
                hmap_pad, mask_pad = data[0]['pose_heatmap'+key].new_zeros(B, max(lens), h, w, N), data[0]['pose_mask'+key].new_zeros(B, max(lens), N)
                for b in range(B):
                    hmap_pad[b][:lens[b]] = data[b]['pose_heatmap'+key]
                    mask_pad[b][:lens[b]] = data[b]['pose_mask'+key]
                    sample['pose_heatmap'+key], sample['pose_mask'+key] = hmap_pad, mask_pad
            else:
                sample['pose_heatmap'+key], sample['pose_mask'+key] = None, None
        return sample

    def __len__(self):
        return len(self.det_labels)

    def _load_video(self, video_path, as_gray=False, scale=1, bboxes=None):
        print('Loading video', video_path)
        cap = cv2.VideoCapture(video_path)
        frames_global, frames_local = [], []
        i_frame = 0
        while True:
            ret, frame = cap.read()
            if ret:
                if not as_gray:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                else:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                W, H = frame.shape[1], frame.shape[0]
                size_rescale = int(min(W, H)*scale)
                if W < H:
                    frame_global = frame[H//2-W//2: H//2+W//2]
                elif W > H:
                    frame_global = frame[:, W//2-H//2: W//2+H//2]
                dx, dy = int(frame_global.shape[1] * scale), int(frame_global.shape[0] * scale)
                frame_global = cv2.resize(frame_global, (dx, dy))
                frames_global.append(frame_global)
                if bboxes is not None:
                    frame_local = crop(frame, bboxes[i_frame], hw=(size_rescale, size_rescale))
                else:
                    frame_local = frame_global
                frames_local.append(frame_local)
                i_frame += 1
            else:
                break
        frames_local, frames_global = np.stack(frames_local, axis=0).astype(np.float32), np.stack(frames_global, axis=0).astype(np.float32)
        return {'local': frames_local, 'global': frames_global}

    def _parse(self, label_file, scp_file, pose_dir, start_end, bbox_file):
        self.chunk_ids, self.fs_ids, self.rgb_videos, self.opt_videos, self.subids = [], [], [], [], []
        self.det_labels, self.fs_labels, self.fs_masks, self.pose_data, self.bboxes = [], [], [], [], []
        scp_json = json.load(open(scp_file, 'r'))
        data_info = pickle.load(open(label_file, 'rb')) if label_file.split('.')[-1] == 'pkl' else json.load(open(label_file, 'r'))

        id_to_bbox = pickle.load(open(bbox_file, 'rb')) if bbox_file is not None else None
        for line_id, line in enumerate(data_info):
            if not (line_id >= start_end[0] and line_id < start_end[1]):
                continue
            chunk_id = line['video_id'] + '-' + str(line['frames'][0][0]) + '_' + str(line['frames'][0][1])
            subid, rgb_video, opt_video = scp_json[chunk_id][0], scp_json[chunk_id][1], scp_json[chunk_id][2]
            det_label = [[win[0], win[1], 1] for win in line['wins']]
            fs_id, fs_label, fs_mask = line['fs_id'], line['fs_label'], line['fs_mask']
            self.chunk_ids.append(chunk_id)
            self.subids.append(subid)
            self.rgb_videos.append(rgb_video)
            self.opt_videos.append(opt_video)
            self.det_labels.append(det_label)
            self.fs_ids.append(fs_id)
            self.fs_labels.append(fs_label)
            self.fs_masks.append(fs_mask)
            self.pose_data.append(os.path.join(pose_dir, rgb_video.split('/')[-1].split('.')[0]+'.json') if pose_dir != None else None)
            self.bboxes.append(id_to_bbox[chunk_id] if id_to_bbox is not None else None)
        rgb_to_bbox, opt_to_bbox = {video: [] for video in self.rgb_videos}, {video: [] for video in self.opt_videos}
        for rgb_video, opt_video, bbox in zip(self.rgb_videos, self.opt_videos, self.bboxes):
            rgb_to_bbox[rgb_video].append(bbox)
            opt_to_bbox[opt_video].append(bbox)
        for rgb_video, opt_video in zip(rgb_to_bbox.keys(), opt_to_bbox.keys()):
            rgb_to_bbox[rgb_video] = np.concatenate(rgb_to_bbox[rgb_video]) if rgb_to_bbox[rgb_video][0] is not None else None
            opt_to_bbox[opt_video] = np.concatenate(opt_to_bbox[opt_video]) if opt_to_bbox[opt_video][0] is not None else None
        self.rgb_to_bbox, self.opt_to_bbox = rgb_to_bbox, opt_to_bbox
        cap = cv2.VideoCapture(rgb_video)
        self.raw_width, self.raw_height = cap.get(3), cap.get(4)
        cap.release()
        print('Number of data: %d, video raw width: %d, raw height: %d' % (len(self.det_labels), self.raw_width, self.raw_height))
        return

    def get_heatmap(self, body_kps_total, hand_kps_total, w_img, h_img, imgs, bbox=None):
        num_body_kps, num_hand_kps = 25, 42
        w_map, h_map = self.fmap_wh[0], self.fmap_wh[1]

        def kps2heatmap(kps, num_kps, bbox):
            if len(kps) == 0:
                kps = np.zeros([num_kps, 3], dtype=np.float32)
            elif len(kps) == 1:
                kps = np.array(kps[0], dtype=np.float32).reshape(-1, 3)
            elif len(kps) > 1:
                num_person = len(kps)
                kps = np.array(kps, dtype=np.float32).reshape(num_person, num_kps, 3)
                ix = kps[:, :, -1].sum(axis=-1).argmax()
                kps = kps[ix]
                # print(f"{num_person} person, select {ix}")

            heatmaps = np.zeros((h_map, w_map, num_kps), dtype=np.float32)
            mask = []
            for i in range(len(kps)):
                if bbox is not None:
                    x0, y0, x1, y1 = bbox
                    x, y = int(kps[i][0] - x0) * w_map / (x1-x0), int(kps[i][1] - y0) * h_map / (y1-y0)
                else:
                    if w_img < h_img:
                        up_cut = (h_img - w_img)//2
                        x = int(kps[i][0]) * w_map / w_img
                        y = int(kps[i][1] - up_cut) * h_map / w_img
                    elif w_img >= h_img:
                        left_cut = (w_img - h_img)//2
                        x = int(kps[i][0] - left_cut) * w_map / h_img
                        y = int(kps[i][1]) * h_map / h_img
                if not (x >=0 and x < w_map and y >= 0 and y < h_map):
                    valid = 0
                elif (kps[i, 0] == 0 and kps[i, 1] == 0):
                    valid = 0
                else:
                    valid = 1
                    heatmap = gaussian_kernel(size_h=h_map, size_w=w_map, center_x=x, center_y=y, sigma=self.sigma)
                    heatmap[heatmap > 1] = 1
                    heatmap[heatmap < 0.0099] = 0
                    heatmaps[:, :, i] = heatmap
                mask.append(valid)
            mask = np.array(mask, dtype=np.float32)
            return heatmaps, mask

        heatmaps, masks = [], []
        for i in range(0, len(body_kps_total), self.pose_sample_rate):
            # pre-process kps
            body_kps, hand_kps = body_kps_total[i], hand_kps_total[i]
            body_heatmap, body_mask = kps2heatmap(body_kps, num_body_kps, bbox=bbox[i] if bbox is not None else None)
            hand_heatmap, hand_mask = kps2heatmap(hand_kps, num_hand_kps, bbox[i] if bbox is not None else None)
            heatmaps.append(np.concatenate([body_heatmap, hand_heatmap], axis=-1))
            masks.append(np.concatenate([body_mask, hand_mask], axis=-1))
        heatmaps, masks = np.stack(heatmaps), np.stack(masks)
        return heatmaps, masks


class ToTensor(object):
    def __init__(self, device):
        self.device = device

    def __call__(self, sample):
        for k in sample.keys():
            if isinstance(sample[k], np.ndarray):
                sample[k] = torch.FloatTensor(sample[k]).to(self.device)
        return sample

class Normalize(object):
    def __init__(self, mean, std, device):
        self.mean = torch.FloatTensor(mean).to(device).view(1, -1, 1, 1)
        self.std = torch.FloatTensor(std).to(device).view(1, -1, 1, 1)

    def __call__(self, sample):
        sample['img_global'] = ((sample['img_global'] / 255.0) - self.mean) / self.std
        sample['img_local'] = ((sample['img_local'] / 255.0) - self.mean) / self.std
        return sample

def gaussian_kernel(size_w, size_h, center_x, center_y, sigma):
    gridy, gridx = np.mgrid[0:size_h, 0:size_w]
    D2 = (gridx - center_x) ** 2 + (gridy - center_y) ** 2
    return np.exp(-D2 / 2.0 / sigma / sigma)

def crop(img, bbox, hw):
    x0, y0, x1, y1 = bbox
    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
    left_expand = -x0 if x0 < 0 else 0
    up_expand = -y0 if y0 < 0 else 0
    right_expand = x1-img.shape[1]+1 if x1 > img.shape[1]-1 else 0
    down_expand = y1-img.shape[0]+1 if y1 > img.shape[0]-1 else 0
    expand_img = cv2.copyMakeBorder(img, up_expand, down_expand, left_expand, right_expand, cv2.BORDER_CONSTANT, (0, 0, 0))
    hand_patch = expand_img[y0+up_expand: y1+up_expand, x0+left_expand: x1+left_expand]
    hand_patch = cv2.resize(hand_patch, hw)
    return hand_patch

