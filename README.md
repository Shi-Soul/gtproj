# Game Theory Project | Melting Pot Contest @ AI3617-2024-Spring

## Overview
- jd_comp: an example code for submitting agent to jidi.ai competition
- mp_contest: code for trainning agent




## Run
```bash
cd mp_contest
conda create -n gtp python=3.10 -y
conda activate gtp
SYSTEM_VERSION_COMPAT=0 pip install dmlab2d
pip install -e .
sh ray_patch.sh
pip install jax==0.4.12 jaxlib==0.4.12
pip install gym==0.22.0

```

```bash 
wandb login # I have modified the code slightly, for the wandb to work, you need to login first


# Train 
CUDA_VISIBLE_DEVICES=2 python baselines/train/run_ray_train.py --num_workers 50 --num_gpus 1 --wandb 1 --exp pd_matrix

python baselines/train/run_ray_train.py --num_workers 60 --num_gpus 0 --wandb 1 --exp pd_matrix

# Continue Training
CUDA_VISIBLE_DEVICES=0 python baselines/train/run_ray_train.py --num_workers 20 --num_gpus 1 --wandb 1 --exp pd_matrix --continue_training results/torch/pd_matrix/PPO_meltingpot_397b4_00000_0_2024-05-21_13-58-39/checkpoint_007270

# Run Evaluation
RUN_DIR=results/torch/clean_up/PPO_meltingpot_5ecb9_00000_0_2024-05-21_14-28-20
CKP_NAME=checkpoint_000100
CUDA_VISIBLE_DEVICES=-1 python baselines/evaluation/evaluate.py --num_episodes 5 --config_dir $RUN_DIR --policies_dir $RUN_DIR/$CKP_NAME/policies --eval_on_scenario True --scenario clean_up_7 #--create_videos True --video_dir $RUN_DIR/videos




RUN_DIR=results/torch/pd_matrix/PPO_meltingpot_397b4_00000_0_2024-05-21_13-58-39
CKP_NAME=checkpoint_007270
python baselines/train/render_models.py --config_dir $RUN_DIR --policies_dir $RUN_DIR/$CKP_NAME/policies --horizon 500
python baselines/evaluation/evaluate.py --num_episodes 5 --config_dir $RUN_DIR --policies_dir $RUN_DIR/$CKP_NAME/policies  #--create_videos True --video_dir $RUN_DIR/videos


declare -a scenarios=(
    "prisoners_dilemma_in_the_matrix__repeated_0"
    "prisoners_dilemma_in_the_matrix__repeated_1"
    "prisoners_dilemma_in_the_matrix__repeated_2"
    "prisoners_dilemma_in_the_matrix__repeated_3"
)
for scenario in "${scenarios[@]}"
do
    python baselines/evaluation/evaluate.py --num_episodes 5 --config_dir $RUN_DIR --policies_dir $RUN_DIR/$CKP_NAME/policies --eval_on_scenario True --scenario $scenario
done
# Create videos may be super slow, depend on the hardware


# For jidi submission & compete

cd jd_comp
# Example 1: beat with fixed_scenarios
python run_log_fixed_scenario.py --my_ai "rl_agent"
# Example 2: beat with another agent
python run_log.py --my_ai "rl_agent" --opponent "random"


```


## Ref


https://docs.ray.io/en/latest/rllib/rllib-env.html#multi-agent-and-hierarchical

https://docs.ray.io/en/latest/rllib/rllib-algorithms.html#multi-agent
