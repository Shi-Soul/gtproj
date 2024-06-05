# Game Theory Project | Melting Pot Contest @ AI3617-2024-Spring

## Overview
- jd_comp: an example code for submitting agent to jidi.ai competition
- mp_contest: code for trainning agent
- tess: tess framework, with our modification

## Run
Setup
```bash
cd mp_contest
conda create -n gtp python=3.10 -y
conda activate gtp
SYSTEM_VERSION_COMPAT=0 pip install dmlab2d
pip install -e .
sh ray_patch.sh
pip install jax==0.4.12 jaxlib==0.4.12
pip install gym==0.22.0

# for tess
pip install stable-baselines3 opencv-python shimmy 
pip install "gymnasium<0.30,>=0.28.1" # Here is a Library Version Conflict, We are running on the tip of the knife
pip install tqdm
```

Ray Baseline Training
```bash 
wandb login # I have modified the code slightly, for the wandb to work, you need to login first


# Train (different configs for reference) 
CUDA_VISIBLE_DEVICES=2 python baselines/train/run_ray_train.py --num_workers 6 --num_gpus 1 --wandb 1 --exp pd_matrix

CUDA_VISIBLE_DEVICES=1 python baselines/train/run_ray_train.py --num_workers 8 --num_gpus 1 --wandb 1 --exp pd_matrix --network large

python baselines/train/run_ray_train.py --num_workers 60 --num_gpus 0 --wandb 1 --exp pd_matrix


# Continue Training
CUDA_VISIBLE_DEVICES=0 python baselines/train/run_ray_train.py --num_workers 20 --num_gpus 1 --wandb 1 --exp pd_matrix --continue_training results/torch/pd_matrix/PPO_meltingpot_397b4_00000_0_2024-05-21_13-58-39/checkpoint_007270

# Run Evaluation
RUN_DIR=results/torch/clean_up/PPO_meltingpot_5ecb9_00000_0_2024-05-21_14-28-20
CKP_NAME=checkpoint_000100
CUDA_VISIBLE_DEVICES=-1 python baselines/evaluation/evaluate.py --num_episodes 5 --config_dir $RUN_DIR --policies_dir $RUN_DIR/$CKP_NAME/policies --eval_on_scenario True --scenario clean_up_7 #--create_videos True --video_dir $RUN_DIR/videos

    


RUN_DIR=results/torch/pd_matrix/PPO_meltingpot_e0761_00000_0_2024-05-24_13-10-03/
CKP_NAME=checkpoint_001400
python baselines/train/render_models.py --config_dir $RUN_DIR --policies_dir $RUN_DIR/$CKP_NAME/policies --horizon 500
CUDA_VISIBLE_DEVICES=-1 python baselines/evaluation/evaluate.py --num_episodes 5 --config_dir $RUN_DIR --policies_dir $RUN_DIR/$CKP_NAME/policies  #--create_videos True --video_dir $RUN_DIR/videos

# Create videos may be super slow, depend on the hardware

declare -a scenarios=(
    "prisoners_dilemma_in_the_matrix__repeated_0"
    "prisoners_dilemma_in_the_matrix__repeated_1"
    "prisoners_dilemma_in_the_matrix__repeated_2"
    "prisoners_dilemma_in_the_matrix__repeated_3"
)
for scenario in "${scenarios[@]}"
do
    CUDA_VISIBLE_DEVICES=1 python baselines/evaluation/evaluate.py --num_episodes 5 --config_dir $RUN_DIR --policies_dir $RUN_DIR/$CKP_NAME/policies --eval_on_scenario True --scenario $scenario --param_sharing
done

echo $RUN_DIR $CKP_NAME

# SUPPORTED_SCENARIOS = [
#     'clean_up_7',
#     'prisoners_dilemma_in_the_matrix__repeated_0',
#     'prisoners_dilemma_in_the_matrix__repeated_1',
#     'prisoners_dilemma_in_the_matrix__repeated_2',
#     'prisoners_dilemma_in_the_matrix__repeated_3',
# ]
```

Jidi
```bash

# For jidi submission & compete

cd jd_comp
# Example 1: beat with fixed_scenarios
python run_log_fixed_scenario.py --my_ai "rl_agent"
# Example 2: beat with another agent
python run_log.py --my_ai "rl_agent" --opponent "random"


```


Tess
```bash

# Train Tess
CUDA_VISIBLE_DEVICES=1 python train_multitask.py --substrate prisoners_dilemma_in_the_matrix__repeated

# Visualize
python visualize.py --substrate prisoners_dilemma_in_the_matrix__repeated
python visualize.py --substrate clean_up --agent-file ./Tess/saved_models/clean_up/cu_base.pt

# Evaluate
CUDA_VISIBLE_DEVICES=0 python baselines/evaluation/eval_tess.py --num_episodes 5 --policies_dir /home/wjxie/wjxie/env/gtproj/tess/Tess/saved_models/prisoners_dilemma_in_the_matrix__repeated/pd_betray_29024311.pt

CUDA_VISIBLE_DEVICES=0 python baselines/evaluation/eval_tess.py --num_episodes 2 --eval_on_scenario True --scenario clean_up_7 --policies_dir /home/wjxie/wjxie/env/gtproj/tess/Tess/saved_models/clean_up/cu_base.pt
CUDA_VISIBLE_DEVICES=0 python baselines/evaluation/eval_tess.py --num_episodes 2 --eval_on_scenario True --scenario clean_up --policies_dir /home/wjxie/wjxie/env/gtproj/tess/Tess/saved_models/clean_up/cu_base.pt >> cu_result/cu_base_cu.log



```


## Ref


https://docs.ray.io/en/latest/rllib/rllib-env.html#multi-agent-and-hierarchical

https://docs.ray.io/en/latest/rllib/rllib-algorithms.html#multi-agent

https://wandb.ai/marshin/Meltingpot/reports/Meltingpot-initial-trials--VmlldzoyNTIxMDg5

https://github.com/google-deepmind/meltingpot/blob/main/examples/rllib/self_play_train.py

https://github.com/utkuearas/MeltingPot-Tess-v1

https://github.com/benjamin-swain/meltingpot-2023-solution
