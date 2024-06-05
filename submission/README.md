 


## PD Matrix



## Clean Up

Since we use a different framework to train the model, we slightly modify the evaluation script to make it work with the new model. Most of the interface and functionality of it is unchanged. Please use `eval_tess.py` instead of `evaluate.py`

The model should be evaluated as following.

```bash
# to evaluate it in the clean up 7 scenario
CUDA_VISIBLE_DEVICES=0 python baselines/evaluation/eval_tess.py --num_episodes 2 --eval_on_scenario True --scenario clean_up_7 --policies_dir rin_model.pt


# to evaluate it by self play
CUDA_VISIBLE_DEVICES=0 python baselines/evaluation/eval_tess.py --num_episodes 2 --eval_on_scenario True --scenario clean_up --policies_dir rin_model.pt
```

The usage of the modified evaluation script is as following.

```
OPTIONS:
  -h, --help            show this help message and exit
  --num_episodes NUM_EPISODES
                        Number of episodes to run evaluation
  --eval_on_scenario EVAL_ON_SCENARIO
                        Must be True if you want to evaluate on a clean up 7 scenario. If it's not set, the evaluation will be done by self play in clean up substrate. 
  --scenario SCENARIO   Name of the scenario. In our setting it should be choosen from ['clean_up_7', 'clean_up']
  --policies_path POLICIES_PATH
                        File path to the model you want to evaluate
  --create_videos CREATE_VIDEOS
                        Whether to create evaluation videos
  --video_dir VIDEO_DIR
                        Directory where you want to store evaluation videos
```
