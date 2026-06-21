  Model key      HF model            Device   PPL mean   PPL median   Time/doc
  ─────────────────────────────────────────────────────────────────────────────
  llama-8b       Meta-Llama-3.1-8B   xpu          9.11        9.07     1.96s
  qwen3-8b       Qwen/Qwen3-8B       xpu         11.25       10.93     1.29s
  qwen3-4b       Qwen/Qwen3-4B       xpu         14.42       14.16     1.29s
  qwen3-1.7b     Qwen/Qwen3-1.7B     xpu         16.31       16.28     0.58s
  qwen3-0.6b     Qwen/Qwen3-0.6B     xpu         21.48       21.70     0.32s


docker run --gpus all -it --rm \
  -v ~/.cache/huggingface:/hf_cache \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/checkpoints:/workspace/checkpoints \
  -v $(pwd)/logs:/workspace/logs \
  -e HF_TOKEN=$HF_TOKEN \
  mini-llama-finance:latest


python train_cpt.py \
    --train-samples 2000 \
    --eval-samples 200 \
    --epochs 1 \
    --batch-size 4 \
    --grad-accum 8 \
    --chunk-size 512 \
    --lr 2e-5 \
    --device cuda 2>&1 | tee logs/train_cuda.log