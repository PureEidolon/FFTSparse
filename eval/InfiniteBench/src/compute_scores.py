"""
InfiniteBench 评分脚本

用法:
    python compute_scores.py --pred_path pred/llama3-8b-inst/passkey-full.jsonl --task passkey
"""

import os
import json
import argparse
import re
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict

# Rouge 评分（用于摘要和问答任务）
try:
    from rouge import Rouge
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    print("⚠️  rouge 未安装，摘要和问答任务将无法计算 ROUGE 分数")
    print("   安装命令: pip install rouge")


def normalize_answer(s: str) -> str:
    """标准化答案字符串"""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    
    def white_space_fix(text):
        return ' '.join(text.split())
    
    def remove_punc(text):
        return re.sub(r'[^\w\s]', '', text)
    
    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def score_passkey(pred: str, answer: List[str]) -> float:
    """Passkey 任务评分：精确匹配"""
    pred = pred.strip()
    for ans in answer:
        if ans.strip() in pred:
            return 1.0
    return 0.0


def score_number_string(pred: str, answer: List[str]) -> float:
    """Number String 任务评分：精确匹配"""
    pred = pred.strip()
    for ans in answer:
        if ans.strip() in pred:
            return 1.0
    return 0.0


def score_kv_retrieval(pred: str, answer: List[str]) -> float:
    """KV Retrieval 任务评分：精确匹配"""
    pred = pred.strip()
    for ans in answer:
        if ans.strip() in pred:
            return 1.0
    return 0.0


def score_choice(pred: str, answer: List[str]) -> float:
    """选择题评分：匹配选项字母"""
    pred = pred.strip().upper()
    
    # 提取预测的选项
    pred_option = None
    for opt in ['A', 'B', 'C', 'D']:
        if opt in pred:
            pred_option = opt
            break
    
    if pred_option is None:
        return 0.0
    
    # 检查答案
    for ans in answer:
        ans = ans.strip().upper()
        if ans == pred_option or ans.startswith(pred_option):
            return 1.0
    
    return 0.0


def score_qa_rouge(pred: str, answer: List[str]) -> float:
    """问答任务评分：ROUGE-L F1"""
    if not ROUGE_AVAILABLE:
        return 0.0
    
    rouge = Rouge()
    pred = pred.strip()
    
    if not pred:
        return 0.0
    
    max_score = 0.0
    for ans in answer:
        ans = ans.strip()
        if not ans:
            continue
        try:
            scores = rouge.get_scores(pred, ans)
            score = scores[0]['rouge-l']['f']
            max_score = max(max_score, score)
        except:
            continue
    
    return max_score


def score_sum_rouge(pred: str, answer: List[str]) -> float:
    """摘要任务评分：ROUGE-L Sum"""
    if not ROUGE_AVAILABLE:
        return 0.0
    
    rouge = Rouge()
    pred = pred.strip()
    
    if not pred:
        return 0.0
    
    max_score = 0.0
    for ans in answer:
        ans = ans.strip()
        if not ans:
            continue
        try:
            scores = rouge.get_scores(pred, ans)
            score = scores[0]['rouge-l']['f']
            max_score = max(max_score, score)
        except:
            continue
    
    return max_score


def score_dialogue(pred: str, answer: List[str]) -> float:
    """对话角色识别评分：名字匹配"""
    pred = pred.strip().lower()
    
    for ans in answer:
        ans = ans.strip().lower()
        if ans in pred or pred in ans:
            return 1.0
    
    return 0.0


def score_math_find(pred: str, answer: List[str]) -> float:
    """Math Find 任务评分：精确匹配"""
    pred = pred.strip()
    
    # 提取数字
    pred_nums = re.findall(r'-?\d+', pred)
    if not pred_nums:
        return 0.0
    
    pred_num = pred_nums[-1]  # 取最后一个数字
    
    for ans in answer:
        if str(ans).strip() == pred_num:
            return 1.0
    
    return 0.0


def score_math_calc(pred: str, answer: List[str]) -> float:
    """Math Calc 任务评分：序列匹配"""
    # 提取预测的数字序列
    pred_nums = re.findall(r'-?\d+', pred)
    
    for ans in answer:
        # 答案也是数字序列
        if isinstance(ans, str):
            ans_nums = re.findall(r'-?\d+', ans)
        elif isinstance(ans, list):
            ans_nums = [str(x) for x in ans]
        else:
            continue
        
        # 计算匹配数量
        if not ans_nums:
            continue
        
        matches = sum(1 for p, a in zip(pred_nums, ans_nums) if p == a)
        return matches / len(ans_nums)
    
    return 0.0


def score_code_run(pred: str, answer: List[str]) -> float:
    """Code Run 任务评分：精确匹配返回值"""
    pred = pred.strip()
    
    # 提取数字
    pred_nums = re.findall(r'-?\d+\.?\d*', pred)
    if not pred_nums:
        return 0.0
    
    pred_num = pred_nums[-1]  # 取最后一个数字
    
    for ans in answer:
        ans_str = str(ans).strip()
        if ans_str == pred_num:
            return 1.0
        # 尝试浮点数比较
        try:
            if abs(float(pred_num) - float(ans_str)) < 1e-6:
                return 1.0
        except:
            pass
    
    return 0.0


def score_code_debug(pred: str, answer: List[str]) -> float:
    """Code Debug 任务评分：选择题"""
    return score_choice(pred, answer)


# 任务到评分函数的映射
TASK_TO_SCORER = {
    "passkey": score_passkey,
    "number_string": score_number_string,
    "kv_retrieval": score_kv_retrieval,
    "longbook_sum_eng": score_sum_rouge,
    "longbook_qa_eng": score_qa_rouge,
    "longbook_choice_eng": score_choice,
    "longbook_qa_chn": score_qa_rouge,
    "longdialogue_qa_eng": score_dialogue,
    "math_find": score_math_find,
    "math_calc": score_math_calc,
    "code_run": score_code_run,
    "code_debug": score_code_debug,
}


def compute_scores(pred_path: str, task: str) -> Dict[str, Any]:
    """计算预测结果的分数"""
    
    if task not in TASK_TO_SCORER:
        raise ValueError(f"未知任务: {task}，可用任务: {list(TASK_TO_SCORER.keys())}")
    
    scorer = TASK_TO_SCORER[task]
    
    # 加载预测结果
    preds = []
    with open(pred_path, "r", encoding="utf-8") as f:
        for line in f:
            preds.append(json.loads(line))
    
    # 计算分数
    scores = []
    for item in preds:
        pred = item.get("pred", "")
        answer = item.get("answer", item.get("answers", []))
        
        if isinstance(answer, str):
            answer = [answer]
        
        score = scorer(pred, answer)
        scores.append(score)
    
    # 统计
    avg_score = sum(scores) / len(scores) if scores else 0.0
    
    results = {
        "task": task,
        "num_samples": len(scores),
        "avg_score": avg_score,
        "avg_score_percent": f"{avg_score * 100:.2f}%",
        "scores": scores,
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description="InfiniteBench 评分脚本")
    parser.add_argument("--pred_path", type=str, required=True, help="预测结果文件路径")
    parser.add_argument("--task", type=str, required=True, help="任务名称")
    parser.add_argument("--output", type=str, default=None, help="输出文件路径（可选）")
    parser.add_argument("--result_jsonl", type=str, default="result.jsonl", help="汇总结果保存路径")

    args = parser.parse_args()

    # 计算分数


    results = compute_scores(args.pred_path, args.task)

    # 文件名（不含路径）
    file_name = Path(args.pred_path).name

    # 打印结果：文件名+(样本数):分数
    file_stem = Path(args.pred_path).stem  # 去掉 .jsonl 后缀
    print(f"{file_stem:<90} ({results['num_samples']:>4}): {results['avg_score_percent']:>8}")


    # 保存到 result.jsonl（追加模式）
    result_line = {
        "file": file_name,
        "task": results["task"],
        "num_samples": results["num_samples"],
        "score": results["avg_score_percent"],
    }
    with open(args.result_jsonl, "a", encoding="utf-8") as f:
        json.dump(result_line, f, ensure_ascii=False)
        f.write("\n")


    # 保存完整结果（如果指定了输出路径）
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)



if __name__ == "__main__":
    main()
