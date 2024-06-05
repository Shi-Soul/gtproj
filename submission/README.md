# Game Theory Project | Melting Pot Contest @ AI3617-2024-Spring
This submission contains our agent for the project, including one rule-based agent for PD Matrix, and two RL agent for PD Matrix and Clean Up respectively.

The complete code for our project is available at [our public github repo](https://github.com/Shi-Soul/gtproj).

## PD Matrix

`pd_matrix/agents` is our two agents for PD Matrix. `rule_agent` is a rule-based agent, and `rin` is a RL agent. They follow the interface of Jidi competition.

In the fixed scenarios, `rule_agent` scored 90.14 $\pm$ 6.72 points, and `rin` scored 91.85 $\pm$ 10.38 points. ( Calculated from 10 episodes)

By default `rin` is run in CPU.

## Clean Up
### File Structure


Our training code is not included here.
Most of files in the `baselines` directory are not used and we don't make meaningful modifications to them. 

Files related to our model:
- `clean_up/rin_model.pt` : Our model checkpoint
- `clean_up/baselines/customs/impala_v4_new.py` : Model definition

- `clean_up/baselines/customs/policies_tess.py`: Policy wrapper, to make our model runnable in the original framework.

- `clean_up/baselines/evaluation/eval_tess.py` : Evaluation script, slightly modified from `evaluation.py`.

### Evaluation
Since we use a different framework to train the model, we slightly modify the evaluation script to make it work with the new model. Most of the interface and functionality of it is unchanged, except the way it load models and config files. Please use `eval_tess.py` instead of `evaluate.py`

The model should be evaluated as following.
In self-play scenario, the model scored 450.80 $\pm$ 174.14 points, and in clean up 7 scenario, the model scored 1161.38 $\pm$ 377.92 points. ( `focal_per_capita_return`, Calculated from 100 episodes)

```bash
cd clean_up

# to evaluate it in the clean up 7 scenario
CUDA_VISIBLE_DEVICES=0 python baselines/evaluation/eval_tess.py --num_episodes 2 --eval_on_scenario True --scenario clean_up_7 --policies_path rin_model.pt


# to evaluate it by self play
CUDA_VISIBLE_DEVICES=0 python baselines/evaluation/eval_tess.py --num_episodes 2 --eval_on_scenario True --scenario clean_up --policies_path rin_model.pt
```



The usage of the modified evaluation script is as following. The output of it is in the same format as the original evaluation script. By default it should be run in GPU.

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


