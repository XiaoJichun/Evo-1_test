# ！ 检查设置
### nvcc --version
!!! 在安装之前需要先查看本地环境中的  nvcc --version
如果可以实现nvidia-smi，但是nvcc --version不行： 需要安装cuda-toolkit
```
apt update
apt install -y cuda-toolkit-12-8

如果失败：E: Unmet dependencies. Try 'apt --fix-broken install' with no packages (or specify a solution).

可采用以下方案，临时禁用 dpkg 备份功能：
    nano /etc/dpkg/dpkg.cfg

    在文件末尾添加以下内容：
        path-exclude=/usr/lib/x86_64-linux-gnu/libGLX.so.0.0.0
        path-exclude=/usr/lib/x86_64-linux-gnu/libGL.so.1.7.0
        path-exclude=/usr/lib/x86_64-linux-gnu/libOpenGL.so.0
        path-exclude=/usr/lib/x86_64-linux-gnu/libEGL.so.1.1.0

    保存退出后，重新尝试修复安装：
        sudo apt-get -f install

再次：apt install -y cuda-toolkit-12-8 即可
问题解决后，记得删除刚才添加的配置行，恢复 dpkg 的默认行为。


RoboTwin网址：https://robotwin-platform.github.io/
```

### 1.1 创建Evo-1环境
```
conda create -n Evo1 python=3.10 -y
conda activate Evo1
cd Evo_1
pip install -r requirements.txt

MAX_JOBS=64 pip install -v flash-attn --no-build-isolation
# 如果上述指令出现问题的话，请按照以下安装方式运行(源码安装)：

git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
git checkout v2.8.3   # 切换到对应的tag
MAX_JOBS=20 python setup.py install   # 该步骤一定要打开VPN
```

### 1.2 下载libero数据集
```
hf auth login
# 这个是libero V2.1版本的数据集
hf download IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot \
    --repo-type dataset \
    --local-dir /root/workspace/Evo-1/dataset/Evo1_training_dataset/Evo1_Libero_Dataset/libero_spatial_no_noops_1.0.0_lerobot

```

### 1.3 下载metaworld数据集
```
conda activate metaworld
cd /root/workspace/Evo-1/dataset/Evo1_training_dataset
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/MINT-SJTU/Evo1_MetaWorld_Dataset
cd Evo1_MetaWorld_Dataset/
git lfs pull
```

## 2 训练

### 2.1 第一阶段训练
```
conda activate Evo1
cd Evo1

accelerate config 
/root/workspace/Evo-1/Evo_1/json/file.json

# LIBERO训练
在dataset/config.yaml中替换为 libero_franka的参数

accelerate launch --num_processes 1 --num_machines 1 --deepspeed_config_file ds_config.json scripts/train.py --run_name Evo1_libero_stage1 --action_head flowmatching --use_augmentation --lr 1e-5 --dropout 0.2 --weight_decay 1e-3 --batch_size 16 --image_size 448 --max_steps 5000 --log_interval 10 --ckpt_interval 2500 --warmup_steps 1000 --grad_clip_norm 1.0 --num_layers 8 --horizon 50 --finetune_action_head --disable_wandb --vlm_name OpenGVLab/InternVL3-1B --dataset_config_path dataset/config.yaml --per_action_dim 24 --state_dim 24 --save_dir /root/workspace/Evo-1/checkpoints/libero_stage1


# metaworld训练
在dataset/config.yaml中替换为 metaworld_sawyer的参数

CUDA_VISIBLE_DEVICES=0,1 accelerate launch --gpu_ids 0,1 --num_processes 2 --num_machines 1 --deepspeed_config_file ds_config.json scripts/train.py --run_name Evo1_metaworld_stage1 --action_head flowmatching --use_augmentation --lr 1e-5 --dropout 0.2 --weight_decay 1e-3 --batch_size 16 --image_size 448 --max_steps 5000 --log_interval 10 --ckpt_interval 2500 --warmup_steps 1000 --grad_clip_norm 1.0 --num_layers 8 --horizon 50 --finetune_action_head --disable_wandb --vlm_name OpenGVLab/InternVL3-1B --dataset_config_path dataset/config.yaml --per_action_dim 24 --state_dim 24 --save_dir /root/workspace/Evo-1/checkpoints/metaworld_stage1

# 如果需要续训的话，加上resume：--resume --resume_path /root/workspace/Evo-1/checkpoints/metaworld_stage1/step_2500
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --gpu_ids 0,1 --num_processes 2 --num_machines 1 --deepspeed_config_file ds_config.json scripts/train.py --run_name Evo1_metaworld_stage1 --action_head flowmatching --use_augmentation --lr 1e-5 --dropout 0.2 --weight_decay 1e-3 --batch_size 16 --image_size 448 --max_steps 5000 --log_interval 10 --ckpt_interval 2500 --warmup_steps 1000 --grad_clip_norm 1.0 --num_layers 8 --horizon 50 --finetune_action_head --disable_wandb --vlm_name OpenGVLab/InternVL3-1B --dataset_config_path dataset/config.yaml --per_action_dim 24 --state_dim 24 --save_dir /root/workspace/Evo-1/checkpoints/metaworld_stage1 --resume --resume_path /root/workspace/Evo-1/checkpoints/metaworld_stage1/step_2500


# 自定义数据集 训练
在dataset/config.yaml中替换为 自定义数据集相关的参数

CUDA_VISIBLE_DEVICES=0,1 accelerate launch --gpu_ids 0,1 --num_processes 2 --num_machines 1 --deepspeed_config_file ds_config.json scripts/train.py --run_name Evo1_ur_stage1 --action_head flowmatching --use_augmentation --lr 1e-5 --dropout 0.2 --weight_decay 1e-3 --batch_size 16 --image_size 448 --max_steps 5000 --log_interval 10 --ckpt_interval 2500 --warmup_steps 1000 --grad_clip_norm 1.0 --num_layers 8 --horizon 50 --finetune_action_head --disable_wandb --vlm_name OpenGVLab/InternVL3-1B --dataset_config_path dataset/config.yaml --per_action_dim 24 --state_dim 24 --save_dir /root/workspace/Evo-1/checkpoints/ur_stage1
```

### 2.2 第二阶段训练
```
conda activate Evo1
cd Evo1

# LIBERO训练
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --gpu_ids 0,1 --num_processes 2 --num_machines 1 --deepspeed_config_file ds_config.json scripts/train.py --run_name Evo1_libero_stage2 --action_head flowmatching --use_augmentation --lr 1e-5 --dropout 0.2 --weight_decay 1e-3 --batch_size 16 --image_size 448 --max_steps 80000 --log_interval 10 --ckpt_interval 2500 --warmup_steps 1000 --grad_clip_norm 1.0 --num_layers 8 --horizon 50 --finetune_vlm --finetune_action_head --disable_wandb --vlm_name OpenGVLab/InternVL3-1B --dataset_config_path dataset/config.yaml --per_action_dim 24 --state_dim 24 --save_dir /root/workspace/Evo-1/checkpoints/libero_stage2 --resume --resume_pretrain --resume_path /root/workspace/Evo-1/checkpoints/libero_stage1/step_5000

# metaworld训练
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --gpu_ids 0,1 --num_processes 2 --num_machines 1 --deepspeed_config_file ds_config.json scripts/train.py --run_name Evo1_metaworld_stage2 --action_head flowmatching --use_augmentation --lr 1e-5 --dropout 0.2 --weight_decay 1e-3 --batch_size 16 --image_size 448 --max_steps 80000 --log_interval 10 --ckpt_interval 2500 --warmup_steps 1000 --grad_clip_norm 1.0 --num_layers 8 --horizon 50 --finetune_vlm --finetune_action_head --disable_wandb --vlm_name OpenGVLab/InternVL3-1B --dataset_config_path dataset/config.yaml --per_action_dim 24 --state_dim 24 --save_dir /root/workspace/Evo-1/checkpoints/metaworld_stage2 --resume --resume_pretrain --resume_path /root/workspace/Evo-1/checkpoints/metaworld_stage1/step_5000
```


## 3 验证
### 3.1 libero验证

```
# Terminal 1
conda activate Evo1
cd Evo_1
python scripts/Evo1_server.py

# Terminal 2
conda activate libero
cd LIBERO_evaluation
python libero_client_4tasks.py

```

#### 3.2 metaworld

```
# Terminal 1
conda activate Evo1
cd Evo_1
python scripts/Evo1_server.py


# Terminal 2
conda activate metaworld
cd MetaWorld_evaluation
python mt50_evo1_client_prompt.py
```