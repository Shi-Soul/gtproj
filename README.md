# Game Theory Project | Melting Pot Contest @ AI3617-2024-Spring

- jd_comp: an example code for submitting agent to jidi.ai competition
- mp_contest: code for trainning agent

## Run
```bash
cd mp_contest
conda create -n gtp python=3.10
conda activate gtp
SYSTEM_VERSION_COMPAT=0 pip install dmlab2d
pip install -e .
sh ray_patch.sh
pip install jax==0.4.12 jaxlib==0.4.12
pip install gym==0.22.0

```

```bash 
# Train 
CUDA_VISIBLES_DEVICES=2 python baselines/train/run_ray_train.py --num_workers 50 --num_gpus 1 --wandb 1

# Run Evaluation
python baselines/evaluation/evaluate.py --num_episodes 5 --config_dir results/torch/pd_matrix/PPO_meltingpot_0afe2_00000_0_2024-05-20_02-59-58 --policies_dir results/torch/pd_matrix/PPO_meltingpot_0afe2_00000_0_2024-05-20_02-59-58/checkpoint_000001/policies --create_videos True --video_dir results/torch/pd_matrix/PPO_meltingpot_0afe2_00000_0_2024-05-20_02-59-58/videos
```
