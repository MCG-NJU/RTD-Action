import argparse
import copy
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.utils.data


def load_json(file):
    with open(file) as json_file:
        data = json.load(json_file)
        return data


class VideoRecord:
    def __init__(self, vid, num_frames, locations, gt, s_e_scores, fps, args):
        self.id = vid
        self.locations = locations
        self.base = float(locations[0])
        self.window_size = args.window_size
        self.interval = args.interval
        self.locations_norm = [
            (i - self.base) / (self.window_size * self.interval)
            for i in locations
        ]
        self.locations_offset = [
            location - self.base for location in locations
        ]
        self.num_frames = num_frames
        self.absolute_position = args.absolute_position

        self.gt = gt
        self.gt_norm = copy.deepcopy(gt)

        # normalize gt start and end
        for i in self.gt_norm:
            i[0][0] = (i[0][0] - self.base) / (self.window_size *
                                               self.interval)
            i[0][1] = (i[0][1] - self.base) / (self.window_size *
                                               self.interval)

        self.gt_s_e_frames = [i[0] for i in self.gt_norm]
        self.fps = fps
        self.duration = num_frames / fps

        if (args.point_prob_normalize is True):
            range_start = np.max(s_e_scores[:, 0]) - np.min(s_e_scores[:, 0])
            range_end = np.max(s_e_scores[:, 1]) - np.min(s_e_scores[:, 1])
            s_e_scores[:, 0] = (s_e_scores[:, 0] -
                                np.min(s_e_scores[:, 0])) / range_start
            s_e_scores[:, 1] = (s_e_scores[:, 1] -
                                np.min(s_e_scores[:, 1])) / range_end
        self.s_e_scores = s_e_scores


class ThumosDetection(torch.utils.data.Dataset):
    def __init__(self, feature_folder, tem_folder, anno_file, split, args):
        annotations = load_json(anno_file)
        video_list = annotations.keys()
        self.window_size = args.window_size
        self.feature_folder = feature_folder
        self.tem_folder = tem_folder
        self.anno_file = load_json(anno_file)
        self.num_gt = args.gt_size
        if split == 'val':
            self.split = 'test'
        else:
            self.split = 'val'
        self.video_dict = {}
        video_pool = list(self.anno_file.keys())
        video_pool.sort()
        self.video_dict = {video_pool[i]: i for i in range(len(video_pool))}

        self.video_list = []
        for vid in video_list:
            if self.split in vid:
                num_frames = int(self.anno_file[vid]['duration_frame'])
                fps = int(self.anno_file[vid]['fps'])
                annotations = [
                    item['segment_frame']
                    for item in self.anno_file[vid]['annotations']
                ]
                labels = [
                    int(item['label'])
                    for item in self.anno_file[vid]['annotations']
                ]

                s_e_seq = pd.read_csv(
                    os.path.join(self.tem_folder, vid + '.csv'))
                start_scores = np.expand_dims(s_e_seq.start.values, 1)
                end_scores = np.expand_dims(s_e_seq.end.values, 1)
                frames = np.expand_dims(s_e_seq.frame.values, 1)

                seq_len = len(s_e_seq)
                if seq_len <= self.window_size:
                    locations = np.zeros((self.window_size, 1))
                    locations[:seq_len, :] = frames
                    s_e_scores = np.zeros((self.window_size, 2))
                    s_e_scores[:seq_len, 0] = start_scores.squeeze()
                    s_e_scores[:seq_len, 1] = end_scores.squeeze()
                    gt = [(annotations[idx], labels[idx])
                          for idx in range(len(annotations))]
                    self.video_list.append(
                        VideoRecord(vid, num_frames, locations, gt, s_e_scores,
                                    fps, args))
                else:
                    if self.split == 'test':
                        overlap_ratio = 2
                    else:
                        overlap_ratio = 4
                    stride = self.window_size // overlap_ratio
                    ws_starts = [
                        i * stride
                        for i in range((seq_len // self.window_size - 1) *
                                       overlap_ratio + 1)
                    ]
                    ws_starts.append(seq_len - self.window_size)

                    for ws in ws_starts:
                        locations = frames[ws:ws + self.window_size]
                        s_scores = start_scores[ws:ws + self.window_size]
                        e_scores = end_scores[ws:ws + self.window_size]
                        s_e_scores = np.concatenate((s_scores, e_scores),
                                                    axis=1)

                        gt = []
                        for idx in range(len(annotations)):
                            anno = annotations[idx]
                            label = labels[idx]
                            if anno[0] >= locations[0] and anno[
                                    1] <= locations[-1]:
                                gt.append((anno, label))
                        if self.split == 'test':
                            self.video_list.append(
                                VideoRecord(vid, num_frames, locations, gt,
                                            s_e_scores, fps, args))
                        elif len(gt) > 0:
                            self.video_list.append(
                                VideoRecord(vid, num_frames, locations, gt,
                                            s_e_scores, fps, args))
        print(split, len(self.video_list))

    def get_data(self, video: VideoRecord):
        '''
        :param VideoRecord
        :return vid_name,
        locations : [N, 1],
        all_props_feature: [N, ft_dim + 2 + pos_dim],
        (gt_start_frame, gt_end_frame): [num_gt, 2]
        '''

        vid = video.id
        num_frames = video.num_frames
        base = video.base

        og_locations = torch.Tensor([location for location in video.locations])

        vid_feature = torch.load(os.path.join(self.feature_folder, vid))
        ft_idxes = [
            min(i // 8, vid_feature.shape[0] - 1) for i in og_locations
        ]
        snippet_fts = []
        for i in ft_idxes:
            i = int(i)
            snippet_fts.append(vid_feature[i].squeeze())

        snippet_fts = torch.stack(snippet_fts)

        assert snippet_fts.shape == (self.window_size,
                                     2048), print(snippet_fts.shape)

        if video.absolute_position:
            locations = torch.Tensor(
                [location for location in video.locations])
        else:
            locations = torch.Tensor(
                [location for location in video.locations_offset])

        s_e_scores = torch.Tensor(video.s_e_scores)

        gt_s_e_frames = [(s, e, 0) for (s, e) in video.gt_s_e_frames]
        for (s, e, _) in gt_s_e_frames:
            assert s >= 0 and s <= 1 and e >= 0 and e <= 1, '{} {}'.format(
                s, e)

        targets = {
            'labels': [],
            'boxes': [],
            'video_id': torch.Tensor([self.video_dict[vid]])
        }
        for (start, end, label) in gt_s_e_frames:
            targets['labels'].append(int(label))
            targets['boxes'].append((start, end))

        targets['labels'] = torch.LongTensor(targets['labels'])

        targets['boxes'] = torch.Tensor(targets['boxes'])

        # all_props_feature = torch.cat((snippet_fts, s_e_scores), dim=1)

        return vid, locations, snippet_fts, targets, num_frames, base, s_e_scores

    def __getitem__(self, idx):
        return self.get_data(self.video_list[idx])

    def __len__(self):
        return len(self.video_list)


def collate_fn(batch):
    vid_name_list, target_list, num_frames_list, base_list = [[]
                                                              for _ in range(4)
                                                              ]
    batch_size = len(batch)
    ft_dim = batch[0][2].shape[-1]
    max_props_num = batch[0][1].shape[0]
    # props_features = torch.zeros(batch_size, max_props_num, ft_dim)
    snippet_fts = torch.zeros(batch_size, max_props_num, ft_dim)
    locations = torch.zeros(batch_size, max_props_num, 1, dtype=torch.double)
    s_e_scores = torch.zeros(batch_size, max_props_num, 2)

    for i, sample in enumerate(batch):
        vid_name_list.append(sample[0])
        target_list.append(sample[3])
        snippet_fts[i, :max_props_num, :] = sample[2]
        locations[i, :max_props_num, :] = sample[1].reshape((-1, 1))
        num_frames_list.append(sample[4])
        if (sample[5] is not None):
            base_list.append(sample[5])
        s_e_scores[i, :max_props_num, :] = sample[6]

    num_frames = torch.from_numpy(np.array(num_frames_list))
    base = torch.from_numpy(np.array(base_list))

    return vid_name_list, locations, snippet_fts, target_list, num_frames, base, s_e_scores


def build(split, args):
    # split = train/val
    root = Path(args.feature_path)
    assert root.exists(
    ), f'provided thumos14 feature path {root} does not exist'
    feature_folder = root
    tem_folder = Path(args.tem_path)
    anno_file = Path(args.annotation_path)

    dataset = ThumosDetection(feature_folder, tem_folder, anno_file, split,
                              args)
    return dataset


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector',
                                     add_help=False)
    parser.add_argument('--batch_size', default=2, type=int)

    # dataset parameters
    parser.add_argument('--dataset_file', default='thumos14')
    parser.add_argument('--window_size', default=100, type=int)
    parser.add_argument('--gt_size', default=100, type=int)
    parser.add_argument('--feature_path',
                        default='/data1/tj/thumos_2048/',
                        type=str)
    parser.add_argument('--tem_path',
                        default='/data1/tj/BSN_share/output/TEM_results',
                        type=str)
    parser.add_argument('--annotation_path',
                        default='thumos14_anno_action.json',
                        type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--num_workers', default=2, type=int)

    return parser
