# FFTSparse: Efficient Block Sparse Attention

This repository contains the official implementation of **FFTSparse**, a high-performance framework designed for efficient long-context inference and block-sparse attention optimization.

## 🛠️ Installation

### 1. Environment Setup
```
conda create -yn xattn python=3.10
conda activate xattn

conda install -y git
conda install -y nvidia/label/cuda-12.4.0::cuda-toolkit
conda install -y nvidia::cuda-cudart-dev
conda install -y pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia

pip install transformers==4.46 accelerate sentencepiece minference datasets wandb zstandard matplotlib huggingface_hub==0.23.2 torch torchaudio torchvision xformers  vllm==0.6.3.post1 vllm-flash-attn==2.6.1
pip install tensor_parallel==2.0.0

pip install ninja packaging
pip install flash-attn==2.6.3 --no-build-isolation
pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/

# LongBench evaluation
pip install seaborn rouge_score einops pandas


# Install xAttention
git clone https://github.com/mit-han-lab/x-attention.git
pip install -e .

# Install Block Sparse Streaming Attention
git clone https://github.com/mit-han-lab/Block-Sparse-Attention.git
cd Block-Sparse-Attention
python setup.py install
cd ..

export PYTHONPATH="$PYTHONPATH:$(pwd)"

```

##1. LongBench Evaluation
To evaluate the model's accuracy on the LongBench dataset:

```

# Generate model predictions for various long-context tasks
./scripts/run_longbench.sh

# Calculate evaluation metrics (F1 score, Rouge-L, etc.)
./scripts/eval_longbench.sh


```


##2. Efficiency Benchmarking
To measure inference latency, memory usage, and throughput:

```
# Navigate to the efficiency benchmark directory
cd ./eval/efficiency

# Step 0: Run the generation baseline
./0_run_generate.sh

# Step 1: Run multi-layer efficiency analysis
./1_run_eval_multilayer.sh
```

![Uploading Fig4_Speedup_bar.png…]()


























