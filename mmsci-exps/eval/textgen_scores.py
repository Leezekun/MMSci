"""
Code for CLIPScore (https://arxiv.org/abs/2104.08718)
@inproceedings{hessel2021clipscore,
  title={{CLIPScore:} A Reference-free Evaluation Metric for Image Captioning},
  author={Hessel, Jack and Holtzman, Ari and Forbes, Maxwell and Bras, Ronan Le and Choi, Yejin},
  booktitle={EMNLP},
  year={2021}
}
"""

import argparse
from tqdm import tqdm
import numpy as np
import os
import json
import pprint
import warnings
from collections import defaultdict
import evaluate


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Model generation for evaluation.")

    parser.add_argument(
        "--input_json",
        type=str,
        help="Candidates json mapping from image_id --> candidate.",
    )

    parser.add_argument(
        "--answer_json",
        default="../../mmsci-data/benchmark/test/image_caption_generation_data_w_answer.json",
        type=str,
        help="Candidates json mapping from image_id --> candidate.",
    )

    parser.add_argument(
        "--base_generation_output_dir",
        default="./output/image_caption_generation",
        type=str,
    )

    parser.add_argument("--k", default=1, type=int)

    parser.add_argument(
        "--base_score_dir",
        default="./eval_scores/image_caption_generation/textgen",
        type=str,
    )

    parser.add_argument(
        "--overwrite",
        default=0,
        type=int,
    )

    parser.add_argument(
        "--save_per_instance",
        default=None,
        help="if set, we will save per instance clipscores to this file",
    )

    args = parser.parse_args()

    if isinstance(args.save_per_instance, str) and not args.save_per_instance.endswith(
        ".json"
    ):
        print(
            "if you're saving per-instance, please make sure the filepath ends in json."
        )
        quit()
    return args


def get_all_metrics(references, candidates):
    # Initialize the metrics
    bleu_metric = evaluate.load("bleu")
    rouge_metric = evaluate.load("rouge")
    meteor_metric = evaluate.load("meteor")
    bertscore_metric = evaluate.load("bertscore")

    bleu_scores = [[], [], [], [], []]  # B1, B2, B3, B4, BLEU
    rouge_scores = [[], [], [], []]  # ROUGE1, ROUGE2, ROUGEL, ROUGELSUM
    meteor_scores = []
    bertscore_scores = []

    # Calculate scores for each sample
    for ref, cand in tqdm(zip(references, candidates)):
        if not cand.strip():
            print(cand)
            continue

        # bleu score
        try:
            bleu_score = bleu_metric.compute(predictions=[cand], references=ref)
        except:
            bleu_score = {"bleu": 0.0, "precisions": [0.0, 0.0, 0.0, 0.0]}
        overall_score = bleu_score["bleu"]
        bleu_scores[-1].append(overall_score)
        for idx, b in enumerate(bleu_score["precisions"]):
            # print("bleu", idx, b)
            bleu_scores[idx].append(b)

        # rouge score
        try:
            rouge_score = rouge_metric.compute(predictions=[cand], references=ref)
        except:
            rouge_score = {
                "rouge1": 0.0,
                "rouge2": 0.0,
                "rougeL": 0.0,
                "rougeLsum": 0.0,
            }
        r1, r2, rl, rls = (
            rouge_score["rouge1"],
            rouge_score["rouge2"],
            rouge_score["rougeL"],
            rouge_score["rougeLsum"],
        )
        for idx, b in enumerate([r1, r2, rl, rls]):
            # print("rouge", idx, b)
            rouge_scores[idx].append(b)

        # meteor score
        try:
            meteor_score = meteor_metric.compute(predictions=[cand], references=ref)[
                "meteor"
            ]
        except:
            meteor_score = 0.0
        # print("meteor", meteor_score)
        meteor_scores.append(meteor_score)

        # bertscore
        try:
            bertscore_score = bertscore_metric.compute(
                predictions=[cand], references=ref, lang="en"
            )
        except:
            bertscore_score = {"f1": 0.0}
        bertscore_scores.append(bertscore_score["f1"])

    # Calculate average scores
    avg_bleu = {}
    for idx, scores in enumerate(bleu_scores):
        if idx < 4:
            avg_bleu[f"BLEU-{idx+1}"] = np.mean(scores)
        else:
            avg_bleu[f"BLEU"] = np.mean(scores)

    avg_rouge = {}
    for idx, score in enumerate(["rouge1", "rouge2", "rougeL", "rougeLSum"]):
        avg_rouge[score] = np.mean(rouge_scores[idx])

    avg_meteor = np.mean(meteor_scores)
    avg_bertscore = np.mean(bertscore_scores)

    # Create a dictionary to store the average scores
    metrics = {
        "bleu": avg_bleu,
        "rouge": avg_rouge,
        "meteor": avg_meteor,
        "bertscore": avg_bertscore,
    }

    return metrics


def main(args):
    print(f"Evaluating {args.input_json}")

    score_dir = os.path.join(args.base_score_dir, args.tag)
    os.makedirs(score_dir, exist_ok=True)
    score_file = os.path.join(score_dir, args.input_json.split("/")[-1])
    print(score_file)
    if os.path.exists(score_file) and not args.overwrite:
        print(f"Already evaluated. Will skip...")
        return

    with open(args.answer_json) as f:
        answers = json.load(f)

    with open(args.input_json) as f:
        data = json.load(f)

    # reformat data
    old_candidates = {}
    old_references = {}
    for idx, item in enumerate(data):
        if "prediction" not in item:
            continue
        image_id = item["image"].split(".")[0]
        old_candidates[image_id] = item["prediction"]
        old_references[image_id] = [
            item["caption"] if item["caption"] else answers[idx]["caption"]
        ]

    candidates = []
    references = []
    image_ids = old_references.keys()
    for cid in image_ids:
        if cid in old_candidates:
            candidates.append(old_candidates[cid][args.eval_pred_idx])
            references.append(old_references[cid])

    if isinstance(references[0], str):
        references = [[r] for r in references]

    avg_scores = {}
    metrics = get_all_metrics(references, candidates)
    for k, v in metrics.items():
        if k == "bleu":
            for sn, sv in v.items():
                avg_scores[sn] = sv
        elif k == "rouge":
            for score in ["rouge1", "rouge2", "rougeL", "rougeLSum"]:
                print("{}: {:.4f}".format(score, v[score]))
                avg_scores[score] = v[score]
        else:
            print("{}: {:.4f}".format(k.upper(), v))
            avg_scores[k.upper()] = v

        if args.save_per_instance:
            with open(args.save_per_instance, "w") as f:
                f.write(json.dumps(scores))

    return score_file, avg_scores


if __name__ == "__main__":
    args = parse_args()
    args.overwrite = 1

    base_output_dir = (
        "/mnt/raid0/zekun/MMSci/mmsci-exps/eval/output/image_caption_generation"
    )

    for w_abs in [True]:
        for w_ctx in [False]:
            if w_abs and w_ctx:
                continue
            tag = f"abs{w_abs}_ctx{w_ctx}"
            args.tag = tag
            k = args.k
            file = f"{args.model}.json"
            args.input_json = os.path.join(base_output_dir, tag, f"k_{k}", file)

            all_scores = defaultdict(list)
            for i in range(k):
                args.eval_pred_idx = i
                score_file, scores = main(args)
                print(scores)
                for metric, score in scores.items():
                    all_scores[metric].append(score)
            with open(score_file, "w") as fout:
                json.dump(all_scores, fout, indent=4)
