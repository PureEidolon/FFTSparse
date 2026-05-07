#Install
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
pip install -e .

# Install Block Sparse Streaming Attention
git clone https://github.com/mit-han-lab/Block-Sparse-Attention.git
cd Block-Sparse-Attention
python setup.py install
cd ..

export PYTHONPATH="$PYTHONPATH:$(pwd)"
```


#Run Longbench
```
./scripts/run_longbench.sh

./scripts/eval_longbench.sh
```


#Run efficiency
```
cd ./eval/efficiency

./0_run_generate.sh

1_run_eval_multilayer.sh
```
