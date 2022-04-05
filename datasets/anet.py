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
    def __init__(self, vid, num_seconds, num_frames, locations, gt, s_e_scores, args):
        self.id = vid
        self.num_seconds = num_seconds
        self.num_frames = num_frames
        self.locations = locations
        self.locations_norm = [i / num_frames for i in locations]
        self.gt = gt
        self.gt_norm = copy.deepcopy(gt)

        # normalize gt start and end
        for i in self.gt_norm:
            i[0][0] = min(1.0, i[0][0] / self.num_seconds)
            i[0][1] = min(1.0, i[0][1] / self.num_seconds)

        self.gt_s_e_frames = [i[0] for i in self.gt_norm]

        if args.point_prob_normalize is True:
            range_start = np.max(s_e_scores[:, 0]) - np.min(s_e_scores[:, 0])
            range_end = np.max(s_e_scores[:, 1]) - np.min(s_e_scores[:, 1])
            s_e_scores[:, 0] = (
                s_e_scores[:, 0] - np.min(s_e_scores[:, 0])
            ) / range_start
            s_e_scores[:, 1] = (s_e_scores[:, 1] - np.min(s_e_scores[:, 1])) / range_end
        self.s_e_scores = s_e_scores


class ANetDetection(torch.utils.data.Dataset):
    def __init__(self, feature_folder, tem_folder, anno_file, split, args):
        annotations = load_json(anno_file)
        self.csv_file = pd.read_csv(args.csv_path)
        self.feature_folder = feature_folder
        self.tem_folder = tem_folder
        self.temporal_scale = args.temporal_scale

        video_pool = list(annotations.keys())
        video_pool.sort()
        self.video_dict = {video_pool[i]: i for i in range(len(video_pool))}

        self.video_list = []
        for i in range(len(self.csv_file)):
            video_name = self.csv_file.video.values[i]
            data_path = os.path.join(self.feature_folder, video_name)
            # Features () of several videos are missing.
            if not os.path.exists(data_path):
                continue
            video_info = annotations[video_name]
            video_subset = self.csv_file.subset.values[i]
            if split == video_subset:
                num_seconds = float(video_info["duration_second"])
                num_frames = int(video_info["duration_frame"])
                correct_frames = num_frames // 16 * 16
                correct_seconds = float(correct_frames) / num_frames * num_seconds

                gt = []
                for anno in video_info["annotations"]:
                    gt.append([anno["segment"], anno["label"]])

                s_e_seq = pd.read_csv(
                    os.path.join(self.tem_folder, video_name + ".csv")
                )
                start_scores = np.expand_dims(s_e_seq.start.values, 1)
                end_scores = np.expand_dims(s_e_seq.end.values, 1)
                locations = np.expand_dims(s_e_seq.frame.values, 1)
                s_e_scores = np.concatenate((start_scores, end_scores), axis=1)

                self.video_list.append(
                    VideoRecord(
                        video_name,
                        correct_seconds,
                        correct_frames,
                        locations,
                        gt,
                        s_e_scores,
                        args,
                    )
                )

    def get_data(self, video: VideoRecord):
        """
        :param VideoRecord
        :return vid_name,
        locations : [N, 1],
        all_props_feature: [N, ft_dim + 2 + pos_dim],
        (gt_start_frame, gt_end_frame): [num_gt, 2]
        """

        vid = video.id
        num_frames = video.num_seconds

        vid_feature = torch.load(os.path.join(self.feature_folder, vid))
        locations = torch.Tensor([location for location in video.locations])

        s_e_scores = torch.Tensor(video.s_e_scores)

        gt_s_e_frames = [(s, e, 0) for (s, e) in video.gt_s_e_frames]
        for (s, e, _) in gt_s_e_frames:
            assert s >= 0 and s <= 1 and e >= 0 and e <= 1, "{} {}".format(s, e)

        targets = {
            "labels": [],
            "boxes": [],
            "video_id": torch.Tensor([self.video_dict[vid]]),
        }
        for (start, end, label) in gt_s_e_frames:
            targets["labels"].append(int(label))
            targets["boxes"].append((start, end))

        targets["labels"] = torch.LongTensor(targets["labels"])

        targets["boxes"] = torch.Tensor(targets["boxes"])

        # all_props_feature = torch.cat((snippet_fts, s_e_scores), dim=1)

        return vid, locations, vid_feature, targets, num_frames, s_e_scores

    def __getitem__(self, idx):
        return self.get_data(self.video_list[idx])

    def __len__(self):
        return len(self.video_list)


def collate_fn(batch):
    vid_name_list, target_list, num_frames_list = [[] for _ in range(3)]
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
        s_e_scores[i, :max_props_num, :] = sample[5]

    num_frames = torch.from_numpy(np.array(num_frames_list))

    return vid_name_list, locations, snippet_fts, target_list, num_frames, s_e_scores


def build(split, args):
    # split = train/val
    root = Path(args.feature_path)
    assert root.exists(), f"provided thumos14 feature path {root} does not exist"
    feature_folder = root
    tem_folder = Path(args.tem_path)
    anno_file = Path(args.annotation_path)

    dataset = ANetDetection(feature_folder, tem_folder, anno_file, split, args)
    return dataset


def get_args_parser():
    parser = argparse.ArgumentParser("Set transformer detector", add_help=False)
    parser.add_argument("--batch_size", default=2, type=int)

    # dataset parameters
    parser.add_argument("--dataset_file", default="thumos14")
    parser.add_argument("--window_size", default=100, type=int)
    parser.add_argument("--gt_size", default=100, type=int)
    parser.add_argument("--feature_path", default="/data1/tj/thumos_2048/", type=str)
    parser.add_argument(
        "--tem_path", default="/data1/tj/BSN_share/output/TEM_results", type=str
    )
    parser.add_argument(
        "--annotation_path", default="thumos14_anno_action.json", type=str
    )
    parser.add_argument("--remove_difficult", action="store_true")

    parser.add_argument("--num_workers", default=2, type=int)

    return parser
