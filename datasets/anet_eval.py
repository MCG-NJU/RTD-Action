import json

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

import util.misc as utils
from datasets.eval_proposal import ANETproposal


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
        raise ValueError("Dimension of arguments is incorrect")

    m, n = target_segments.shape[0], test_segments.shape[0]
    tiou = np.empty((m, n))
    for i in range(m):
        tt1 = np.maximum(target_segments[i, 0], test_segments[:, 0])
        tt2 = np.minimum(target_segments[i, 1], test_segments[:, 1])

        # Non-negative overlap score
        intersection = (tt2 - tt1 + 1.0).clip(0)
        union = (
            (test_segments[:, 1] - test_segments[:, 0] + 1)
            + (target_segments[i, 1] - target_segments[i, 0] + 1)
            - intersection
        )
        # Compute overlap as the ratio of the intersection
        # over union of two segments at the frame level.
        tiou[i, :] = intersection / union
    return tiou


def load_json(file):
    with open(file) as json_file:
        data = json.load(json_file)
        return data


def formatting(results):
    results_list = []
    frame_dict = load_json("datasets/anet_anno_action.json")
    for vid, info in results.items():
        num_frames = frame_dict[vid]["duration_frame"]
        video_results = []
        for preds_dict in info:
            scores = preds_dict["scores"]
            boxes = preds_dict["boxes"]

            boxes = boxes.detach().cpu().numpy()
            scores = scores.detach().cpu().numpy()

            for sample_idx in range(boxes.shape[0]):
                # results_list.append([
                #     float(boxes[sample_idx][0]),
                #     float(boxes[sample_idx][1]),
                #     float(scores[sample_idx]), num_frames, vid
                # ])
                video_results.append(
                    [
                        float(boxes[sample_idx][0]),
                        float(boxes[sample_idx][1]),
                        float(scores[sample_idx]),
                    ]
                )
        video_results_pd = pd.DataFrame(
            video_results, columns=["t-init", "t-end", "score"]
        )
        video_results_pd = video_results_pd.sort_values(by="score", ascending=False)

        for j in range(min(100, len(video_results_pd))):
            results_list.append(
                [
                    video_results_pd["t-init"].values[j],
                    video_results_pd["t-end"].values[j],
                    video_results_pd["score"].values[j],
                    vid,
                ]
            )

    results_list = np.stack(results_list)

    results_pd = pd.DataFrame(
        results_list, columns=["t-init", "t-end", "score", "video-name"]
    )
    results_pd.to_csv("./outputs/results_eval.csv", index=False)


def run_evaluation(
    ground_truth_filename,
    proposal_filename,
    max_avg_nr_proposals=100,
    tiou_thresholds=np.linspace(0.5, 0.95, 10),
    subset="validation",
):

    anet_proposal = ANETproposal(
        ground_truth_filename,
        proposal_filename,
        tiou_thresholds=tiou_thresholds,
        max_avg_nr_proposals=max_avg_nr_proposals,
        subset=subset,
        verbose=True,
        check_status=False,
    )
    auc = anet_proposal.evaluate()

    recall = anet_proposal.recall
    average_recall = anet_proposal.avg_recall
    average_nr_proposals = anet_proposal.proposals_per_video

    return (average_nr_proposals, average_recall, recall, auc)


def eval_props(results):
    formatting(results)
    groundtruth = pd.read_csv("datasets/anet_val_groundtruth_label.csv")
    prediction = pd.read_csv("./outputs/results_eval.csv")

    # Computes average recall vs average number of proposals.
    (
        uniform_average_nr_proposals_valid,
        uniform_average_recall_valid,
        uniform_recall_valid,
        auc,
    ) = run_evaluation(
        groundtruth,
        prediction,
        max_avg_nr_proposals=100,
        tiou_thresholds=np.linspace(0.5, 0.95, 10),
        subset="validation",
    )

    ar1 = np.mean(uniform_recall_valid[:, 0])
    ar5 = np.mean(uniform_recall_valid[:, 4])
    ar10 = np.mean(uniform_recall_valid[:, 9])
    ar50 = np.mean(uniform_recall_valid[:, 49])
    ar100 = np.mean(uniform_recall_valid[:, 99])

    return {
        "1": ar1,
        "5": ar5,
        "10": ar10,
        "50": ar50,
        "100": ar100,
        "auc": auc * 0.01,
    }, prediction


class ANetEvaluator(object):
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
