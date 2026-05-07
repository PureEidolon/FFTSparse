# InfiniteBench 评估框架

本项目用于在 InfiniteBench 基准上评估不同 Attention 方法的效果。

## 📁 目录结构

```
infinitebench_eval/
├── config/                          # 配置文件
│   ├── dataset2prompt.json          # 各任务的 prompt 模板
│   ├── dataset2maxlen.json          # 各任务的最大生成长度
│   ├── dataset2file.json            # 任务名到数据文件的映射
│   ├── model2path.json              # 模型名到路径的映射（需要修改）
│   └── model2maxlen.json            # 模型的最大上下文长度
├── data/                            # InfiniteBench 数据集（需要下载）
│   ├── passkey.jsonl
│   ├── number_string.jsonl
│   ├── kv_retrieval.jsonl
│   ├── longbook_sum_eng.jsonl
│   ├── longbook_qa_eng.jsonl
│   ├── longbook_choice_eng.jsonl
│   ├── longbook_qa_chn.jsonl
│   ├── longdialogue_qa_eng.jsonl
│   ├── math_find.jsonl
│   ├── math_calc.jsonl
│   ├── code_run.jsonl
│   └── code_debug.jsonl
├── pred/                            # 预测结果输出目录
│   └── {model_name}/
│       └── {task}-{method}.jsonl
├── src/                             # 源代码
│   ├── pred_infinitebench.py        # 主评估脚本
│   └── compute_scores.py            # 评分脚本
├── scripts/                         # 运行脚本
│   ├── run_eval.sh                  # 基础评估脚本
│   └── run_compare_methods.sh       # 多方法对比脚本
└── README.md
```

## 🚀 快速开始

### 1. 下载 InfiniteBench 数据集

```bash
# 方法一：使用 Hugging Face
from datasets import load_dataset
dataset = load_dataset("xinrongzhang2022/InfiniteBench")

# 方法二：从 GitHub 下载
git clone https://github.com/OpenBMB/InfiniteBench.git
cd InfiniteBench
bash scripts/download_dataset.sh
```

将数据文件放到 `data/` 目录下。

### 2. 配置模型路径

编辑 `config/model2path.json`，添加你的模型路径：

```json
{
    "llama3-8b-inst": "/path/to/your/llama3-8b-instruct",
    "your-model-name": "/path/to/your/model"
}
```

### 3. 运行评估

```bash
# 单个任务评估
python src/pred_infinitebench.py \
    --model llama3-8b-inst \
    --method full \
    --task passkey \
    --num_samples 10

# 计算分数
python src/compute_scores.py \
    --pred_path pred/llama3-8b-inst/passkey-full.jsonl \
    --task passkey
```

### 4. 对比多种 Attention 方法

```bash
bash scripts/run_compare_methods.sh
```

## 📋 支持的任务

| 任务名 | 说明 | 评估指标 |
|--------|------|----------|
| passkey | 密钥检索 | Accuracy |
| number_string | 数字序列检索 | Accuracy |
| kv_retrieval | KV 键值检索 | Accuracy |
| longbook_sum_eng | 英文书籍摘要 | ROUGE-L |
| longbook_qa_eng | 英文书籍问答 | ROUGE-L |
| longbook_choice_eng | 英文书籍选择题 | Accuracy |
| longbook_qa_chn | 中文书籍问答 | ROUGE-L |
| longdialogue_qa_eng | 对话角色识别 | Accuracy |
| math_find | 数学查找 | Accuracy |
| math_calc | 数学计算 | Accuracy |
| code_run | 代码执行 | Accuracy |
| code_debug | 代码调试 | Accuracy |

## 🔧 支持的 Attention 方法

| 方法 | 参数 | 说明 |
|------|------|------|
| full | - | 标准 Flash Attention (baseline) |
| myattn | sink_ratio, recent_ratio, local_ratio, corr_thres, block_size | 自定义 FFT 相关性注意力 |
| xattn | - | Xattention |
| flex | - | FlexPrefill |
| minference | - | MInference |
| sparge | - | SpargeAttn |

## 🛠️ 添加新的 Attention 方法

1. 在 `pred_infinitebench.py` 顶部导入你的 attention 模块
2. 在 `new_attention_forward` 函数中添加你的方法实现
3. 在 `parse_args` 中添加相关参数

示例：

```python
# 1. 导入模块
from your_attn_module import your_attn_prefill

# 2. 在 new_attention_forward 中添加
elif self.method == "your_method":
    if self.layer_idx == 0:
        print("执行 YourMethod")
    attn_output = your_attn_prefill(
        query_states,
        key_states,
        value_states,
        # your parameters
    )
```

## 📊 结果格式

预测结果保存为 JSONL 格式：

```json
{"id": 0, "pred": "12345", "answer": ["12345"], "options": [], "input_length": 122400}
```

评分结果：

```json
{
    "task": "passkey",
    "num_samples": 590,
    "avg_score": 0.95,
    "avg_score_percent": "95.00%"
}
```

## 📝 注意事项

1. **显存管理**：InfiniteBench 的输入很长（100K+ tokens），建议使用支持长上下文的模型和充足的 GPU 显存
2. **math_calc 任务**：该任务输出很长（~44K tokens），大多数模型表现较差
3. **ROUGE 评分**：需要安装 `rouge` 包：`pip install rouge`

## 🔗 参考

- [InfiniteBench GitHub](https://github.com/OpenBMB/InfiniteBench)
- [InfiniteBench Paper](https://aclanthology.org/2024.acl-long.814)
- [Hugging Face Dataset](https://huggingface.co/datasets/xinrongzhang2022/InfiniteBench)
