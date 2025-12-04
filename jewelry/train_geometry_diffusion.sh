# config='./configs/train-geometry-diffusion/step1x-3d-geometry-1300m.yaml'
root_dir='/home/wangxinxing/code/Step1X-3D'
config="${root_dir}/configs/train-geometry-diffusion/jewelry-3d-geometry.yaml"
export CUDA_VISIBLE_DEVICES=0
# python train.py --config $config --train --gpu 0 

python $root_dir/train.py --config $config --train --gpu 0 system.use_lora=True # for lora training

# multi-GPU training
# torchrun train.py \
#     --config $config \
#     --train \
#     --gpu 0,1,2,3,4,5,6,7 \
#     trainer.num_nodes=$num_nodes \
#     system.use_lora=True \
#     --use_ema \