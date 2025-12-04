
先跑prepare_data.py脚本准备数据，一共九个，

CUDA_VISIBLE_DEVICES=0 python train.py --config configs/train-geometry-diffusion/jewelry-3d-geometry.yaml --train --gpu 0 system.use_lora=True