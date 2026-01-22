#!/bin/zsh

# 示例脚本：根据需要编辑以下变量，再直接运行本文件即可。
# 默认示例：批量遍历 jewelry/inference/images，执行 base+LoRA 推理并渲染。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# === 根据自己的需求修改 ===
IMAGES_DIR="$SCRIPT_DIR/jewelry/inference/test_qianzong_1229"
MODE="base"               # base / lora / both
LORA_SOURCE="/home/wangxinxing/code/Step1X-3D/outputs/step1x-3d-geometry/jewelry-lora-v1/michelangelo-autoencoder+n32768+AdamWlr0.0001@20251216-150521/converted/pytorch_model.bin"            # 训练 run 目录或已转换的 ckpt，留空表示仅 base
RENDER_FLAG="skip"      # render / skip
CUDA_VISIBLE_DEVICES=0
# =========================

echo "[pipeline.sh] 示例调用 -> images_dir=$IMAGES_DIR mode=$MODE render=$RENDER_FLAG"
python "$SCRIPT_DIR/pipeline.py" "$IMAGES_DIR" "$MODE" "$LORA_SOURCE" "$RENDER_FLAG"
