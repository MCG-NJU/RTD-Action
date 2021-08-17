import json

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

import util.misc as utils


def segment_tiou(target_segments, test_segments):
    """Compute intersection over union btw segments
    Parameters
    ----------
    target_segments : ndarray
        2-dim array in format [m x 2:=[init, end]]
    test_segments : ndarray
        2-dim array in format [n x 2:=[init, end]]
    Outputs
    -------
    tiou : ndarray
        2-dim array [m x n] with IOU ratio.
    Note: It assumes that target-segments are more scarce that test-segments
    """
    if target_segments.ndim != 2 or test_segments.ndim != 2:
        raise ValueError('Dimension of arguments is incorrect')

    m, n = target_segments.shape[0], test_segments.shape[0]
    tiou = np.empty((m, n))
    for i in range(m):
        tt1 = np.maximum(target_segments[i, 0], test_segments[:, 0])
        tt2 = np.minimum(target_segments[i, 1], test_segments[:, 1])

        # Non-negative overlap score
        intersection = (tt2 - tt1 + 1.0).clip(0)
        union = ((test_segments[:, 1] - test_segments[:, 0] + 1) +
                 (target_segments[i, 1] - target_segments[i, 0] + 1) -
                 intersection)
        # Compute overlap as the ratio of the intersection
        # over union of two segments at the frame level.
        tiou[i, :] = intersection / union
    return tiou


def average_recall_vs_nr_proposals(proposals,
                                   ground_truth,
                                   tiou_thresholds=np.linspace(0.5, 1.0, 11)):
    """Computes the average recall given an average number of proposals per
    video.

    Parameters
    ----------
    proposals : DataFrame
        pandas table with the resulting proposals. It must include
        the following columns: {'video-name': (str) Video identifier,
                                'f-init': (int) Starting index Frame,
                                'f-end': (int) Ending index Frame,
                                'score': (float) Proposal confidence}
    ground_truth : DataFrame
        pandas table with annotations of the dataset. It must include
        the following columns: {'video-name': (str) Video identifier,
                                'f-init': (int) Starting index Frame,
                                'f-end': (int) Ending index Frame}
    tiou_thresholds : 1darray, optional
        array with tiou threholds.

    Outputs
    -------
    average_recall : 1darray
        recall averaged over a list of tiou threshold.
    proposals_per_video : 1darray
        average number of proposals per video.
    """
    # Get list of videos.
    video_lst = proposals['video-name'].unique()

    # For each video, computes tiou scores among the retrieved proposals.
    score_lst = []
    for videoid in video_lst:

        # Get proposals for this video.
        prop_idx = proposals['video-name'] == videoid
        this_video_proposals = proposals[prop_idx][['f-init', 'f-end'
                                                    ]].values.astype(np.float)

        # Sort proposals by score.
        sort_idx = proposals[prop_idx]['score'].argsort()[::-1]
        this_video_proposals = this_video_proposals[sort_idx, :]

        # Get ground-truth instances associated to this video.
        gt_idx = ground_truth['video-name'] == videoid
        this_video_ground_truth = ground_truth[gt_idx][['f-init',
                                                        'f-end']].values

        # Compute tiou scores.
        tiou = segment_tiou(this_video_ground_truth, this_video_proposals)
        score_lst.append(tiou)

    # Given that the length of the videos is really varied, we
    # compute the number of proposals in terms of a ratio of the total
    # proposals retrieved, i.e. average recall at a percentage of proposals
    # retrieved per video.

    # Computes average recall.
    pcn_lst = np.arange(1, 201) / 200.0
    matches = np.empty((video_lst.shape[0], pcn_lst.shape[0]))
    positives = np.empty(video_lst.shape[0])
    recall = np.empty((tiou_thresholds.shape[0], pcn_lst.shape[0]))
    # Iterates over each tiou threshold.
    for ridx, tiou in enumerate(tiou_thresholds):

        # Inspect positives retrieved per video at different
        # number of proposals (percentage of the total retrieved).
        for i, score in enumerate(score_lst):
            # Total positives per video.
            positives[i] = score.shape[0]

            for j, pcn in enumerate(pcn_lst):
                # Get number of proposals as a percentage of total retrieved.
                nr_proposals = int(score.shape[1] * pcn)
                # Find proposals that satisfies minimum tiou threhold.
                matches[i, j] = ((score[:, :nr_proposals] >= tiou).sum(axis=1)
                                 > 0).sum()

        # Computes recall given the set of matches per video.
        recall[ridx, :] = matches.sum(axis=0) / positives.sum()

    # Recall is averaged.
    recall = recall.mean(axis=0)

    # Get the average number of proposals per video.
    proposals_per_video = pcn_lst * (float(proposals.shape[0]) /
                                     video_lst.shape[0])

    return recall, proposals_per_video


def recall_vs_tiou_thresholds(proposals,
                              ground_truth,
                              nr_proposals=1000,
                              tiou_thresholds=np.arange(0.05, 1.05, 0.05)):
    """Computes recall at different tiou thresholds given a fixed average
    number of proposals per video.

    Parameters
    ----------
    proposals : DataFrame
        pandas table with the resulting proposals. It must include
        the following columns: {'video-name': (str) Video identifier,
                                'f-init': (int) Starting index Frame,
                                'f-end': (int) Ending index Frame,
                                'score': (float) Proposal confidence}
    ground_truth : DataFrame
        pandas table with annotations of the dataset. It must include
        the following columns: {'video-name': (str) Video identifier,
                                'f-init': (int) Starting index Frame,
                                'f-end': (int) Ending index Frame}
    nr_proposals : int
        average number of proposals per video.
    tiou_thresholds : 1darray, optional
        array with tiou threholds.

    Outputs
    -------
    average_recall : 1darray
        recall averaged over a list of tiou threshold.
    proposals_per_video : 1darray
        average number of proposals per video.
    """
    # Get list of videos.
    video_lst = proposals['video-name'].unique()

    # For each video, computes tiou scores among the retrieved proposals.
    score_lst = []
    for videoid in video_lst:

        # Get proposals for this video.
        prop_idx = proposals['video-name'] == videoid
        this_video_proposals = proposals[prop_idx][['f-init', 'f-end']].values
        # Sort proposals by score.
        sort_idx = proposals[prop_idx]['score'].argsort()[::-1]
        this_video_proposals = this_video_proposals[sort_idx, :]

        # Get ground-truth instances associated to this video.
        gt_idx = ground_truth['video-name'] == videoid
        this_video_ground_truth = ground_truth[gt_idx][['f-init',
                                                        'f-end']].values

        # Compute tiou scores.
        tiou = segment_tiou(this_video_ground_truth, this_video_proposals)
        score_lst.append(tiou)

    # To obtain the average number of proposals, we need to define a
    # percentage of proposals to get per video.
    pcn = (video_lst.shape[0] * float(nr_proposals)) / proposals.shape[0]

    # Computes recall at different tiou thresholds.
    matches = np.empty((video_lst.shape[0], tiou_thresholds.shape[0]))
    positives = np.empty(video_lst.shape[0])
    recall = np.empty(tiou_thresholds.shape[0])
    # Iterates over each tiou threshold.
    for ridx, tiou in enumerate(tiou_thresholds):

        for i, score in enumerate(score_lst):
            # Total positives per video.
            positives[i] = score.shape[0]

            # Get number of proposals at the fixed percentage of total retrieved.
            nr_proposals = int(score.shape[1] * pcn)
            # Find proposals that satisfies minimum tiou threhold.
            matches[i, ridx] = ((score[:, :nr_proposals] >= tiou).sum(axis=1) >
                                0).sum()

        # Computes recall given the set of matches per video.
        recall[ridx] = matches[:, ridx].sum(axis=0) / positives.sum()

    return recall, tiou_thresholds


def load_json(file):
    with open(file) as json_file:
        data = json.load(json_file)
        return data


def formatting(results):
    results_list = []
    frame_dict = load_json('datasets/thumos_frames_info.json')
    for vid, info in results.items():
        num_frames = frame_dict[vid]
        for preds_dict in info:
            scores = preds_dict['scores']
            boxes = preds_dict['boxes']

            boxes = boxes.detach().cpu().numpy()
            scores = scores.detach().cpu().numpy()

            for sample_idx in range(boxes.shape[0]):
                results_list.append([
                    float(boxes[sample_idx][0]),
                    float(boxes[sample_idx][1]),
                    float(scores[sample_idx]), num_frames, vid
                ])

    results_list = np.stack(results_list)

    results_pd = pd.DataFrame(
        results_list,
        columns=['f-init', 'f-end', 'score', 'video-frames', 'video-name'])
    return results_pd


def eval_props(results):
    results = formatting(results)
    ground_truth = pd.read_csv('datasets/thumos14_test_groundtruth.csv')

    # Computes average recall vs average number of proposals.
    average_recall, average_nr_proposals = average_recall_vs_nr_proposals(
        results, ground_truth)

    f = interp1d(average_nr_proposals,
                 average_recall,
                 axis=0,
                 fill_value='extrapolate')

    return {
        '50': str(f(50)),
        '100': str(f(100)),
        '200': str(f(200)),
        '500': str(f(500))
    }, results


class Thumos14Evaluator(object):
    def __init__(self):
        self.predictions = []

    def update(self, vid, predictions):
        self.predictions += [(vid, predictions)]

    def get_result(self):
        return self.predictions

    def synchronize_between_processes(self):
        all_predictions = utils.all_gather(self.predictions)
        merged_predictions = []
        for p in all_predictions:
            merged_predictions += p
        self.predictions = merged_predictions

    def summarize(self):
        results = {}

        for vid, p in self.predictions:
            try:
                results[vid].append(p)
            except KeyError:
                results[vid] = []
                results[vid].append(p)

        return results
