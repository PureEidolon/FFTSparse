import os
import json
import argparse
import numpy as np

from metrics import (
    qa_f1_score,
    rouge_zh_score,
    qa_f1_zh_score,
    rouge_score,
    classification_score,
    retrieval_score,
    retrieval_zh_score,
    count_score,
    code_sim_score,
)

dataset2metric = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "vcsum": rouge_zh_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
    "longbook_qa_eng_88K": qa_f1_score,
}


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--e", action="store_true", help="Evaluate on LongBench-E")
    parser.add_argument("--results_path", type=str, default=None)
    parser.add_argument("--datasets", type=str, nargs="+", default=None,
                        help="Only evaluate these datasets")
    return parser.parse_args(args)


def scorer_e(dataset, predictions, answers, lengths, all_classes):
    scores = {"0-4k": [], "4-8k": [], "8k+": []}
    for prediction, ground_truths, length in zip(predictions, answers, lengths):
        score = 0.0
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip("\n").split("\n")[0]
        for ground_truth in ground_truths:
            score = max(
                score,
                dataset2metric[dataset](
                    prediction, ground_truth, all_classes=all_classes
                ),
            )
        if length < 4000:
            scores["0-4k"].append(score)
        elif length < 8000:
            scores["4-8k"].append(score)
        else:
            scores["8k+"].append(score)
    for key in scores.keys():
        scores[key] = round(100 * np.mean(scores[key]), 2)
    return scores


def scorer(dataset, predictions, answers, all_classes):
    total_score = 0.0
    for prediction, ground_truths in zip(predictions, answers):
        score = 0.0
        prediction = (
            prediction.split(".assistant")[0]
            .split("\n\nQuestion")[0]
            .split("</s>")[0]
            .split("(Document")[0]
            .split("\n\nQuestion")[0]
            .split("\n\nAnswer")[0]
            .split("(Passage")[0]
            .strip()
        )
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip("\n").split("\n")[0]
        if dataset in ["multifieldqa_zh", "dureader"]:
            prediction = prediction.split("问题：")[0].strip()
        if dataset in ["lsht"]:
            prediction = prediction.split("新闻内容：")[0].strip()
        if dataset in ["passage_retrieval_zh"]:
            prediction = prediction.split("请问")[0].split("提示")[0].strip()
        for ground_truth in ground_truths:
            score = max(
                score,
                dataset2metric[dataset](
                    prediction, ground_truth, all_classes=all_classes
                ),
            )
        total_score += score
    return round(100 * total_score / len(predictions), 2)


if __name__ == "__main__":
    args = parse_args()
    scores = dict()
    if args.results_path:
        path = args.results_path
    else:
        if args.e:
            path = f"pred_e/{args.model}/"
        else:
            path = f"pred/{args.model}/"
    all_files = os.listdir(path)
    all_files.sort()
    print("Evaluating on:", all_files)
    for filename in all_files:
        if not filename.endswith("jsonl"):
            continue
        predictions, answers, lengths = [], [], []
        attn_times, prefill_times, input_lens = [], [], []
        dataset = filename.split("-")[0]
        if dataset == "repobench":
            dataset = "repobench-p"
        if args.datasets and dataset not in args.datasets:
            continue




        print(filename, end=' ')
        with open(f"{path}{filename}", "r", encoding="utf-8") as f:
            content = f.read()

        decoder = json.JSONDecoder()
        pos = 0
        while pos < len(content):
            s = content[pos:].lstrip()
            if not s:
                break
            try:
                data, end = decoder.raw_decode(s)
            except json.JSONDecodeError:
                pos += 1
                continue
            pos += end + (len(content[pos:]) - len(s))

            predictions.append(data["pred"])
            answers.append(data["answers"])
            all_classes = data["all_classes"]
            if "length" in data:
                lengths.append(data["length"])
            if "attn_ms" in data:
                attn_times.append(data["attn_ms"])
            if "prefill_ms" in data:
                prefill_times.append(data["prefill_ms"])
            if "input_len" in data:
                input_lens.append(data["input_len"])



        if len(predictions) == 0:
            continue
        if args.e:
            score = scorer_e(dataset, predictions, answers, lengths, all_classes)
        else:
            score = scorer(dataset, predictions, answers, all_classes)


        filename = f"{filename}({len(predictions)} samples)"
        if attn_times:
            scores[filename] = {
                "score": score,
                "avg_input_len": round(np.mean(input_lens), 0),
                "avg_prefill_ms": round(np.mean(prefill_times), 2),
                "avg_attn_ms": round(np.mean(attn_times), 2),
                "attn_ratio": round(np.mean(attn_times) / np.mean(prefill_times) * 100, 1),
            }
        else:
            scores[filename] = score


        print(f"({len(predictions)} samples): {score}")
        if attn_times:
            print(f"  ⏱️  avg_input_len={np.mean(input_lens):.0f}, "
                  f"avg_prefill={np.mean(prefill_times):.2f}ms, "
                  f"avg_attn={np.mean(attn_times):.2f}ms, "
                  f"attn_ratio={np.mean(attn_times) / np.mean(prefill_times) * 100:.1f}%")

    if args.results_path:
        out_path = os.path.join(args.results_path, "result.json")
    else:
        if args.e:
            out_path = f"pred_e/{args.model}/result.json"
        else:
            out_path = f"pred/{args.model}/result.json"

    with open(out_path, "w") as f:
        json.dump(scores, f, ensure_ascii=False, indent=4)
