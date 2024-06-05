# -*- coding:utf-8  -*-
from typing import Tuple, List, Any
import random
import torch
import numpy as np
import time
import os
import cv2
from pathlib import Path
import sys

base_dir = Path(__file__).resolve().parent
print(base_dir)
sys.path.append(str(base_dir))

AGENT_FILE = str(base_dir)+"/pd_best.pt"
OBS_S = (2, 20, 20, 3)
ACT_S = (2,)
NUM_ACT = 8
DEVICE = "cpu"

def get_agent(agent_file=AGENT_FILE):
    kwargs = {"inv":True}
    from impala_v4_new import Model as model
    agent = model(OBS_S[1:], NUM_ACT, task_count=2, **kwargs)
    agent = agent.to(DEVICE)
    agent.load_state_dict(torch.load(agent_file, map_location=torch.device(DEVICE)))
    return agent

class EvalPolicy():
    """Loads the policies from  Policy checkpoints and removes unrequired observations
    that policies cannot expect to have access to during evaluation.
    """

    def __init__(
        self,
        agent_file=AGENT_FILE,
        *args,
        **kwargs,
    ) -> None:
        self.agent = get_agent(agent_file)

    def initial_state(self):
        """See base class."""
        h_n = torch.zeros((1,256)).to(DEVICE)   
        c_n = torch.zeros((1,256)).to(DEVICE)
        
        self.state = (h_n, c_n)
        return self.state

    def step(self, observation:dict, prev_reward: Any=None):
        # Need: obs(torch.Size([2, 3, 20, 20])), shoot, inv, 
        
        obs_list = [cv2.resize(i["RGB"], (i["RGB"].shape[0] // 2,i["RGB"].shape[1] // 2), cv2.INTER_AREA) for i in [observation]]
        obs = np.array(obs_list)
        self.last_partial_obs = obs_list[0]
        obs = np.array(obs / 255, dtype=np.float32)
        
        
        shoot = [int(i["READY_TO_SHOOT"]) for i in [observation]]
        
        ts_invs = [i["INVENTORY"] for i in [observation]]
        invs = []
        for i in ts_invs:
            summa = sum(i)
            invs.append([i[0]/summa,i[1]/summa])
        
        obs = torch.from_numpy(obs).to(DEVICE).permute((0,3,1,2))
        inv = torch.tensor(invs, dtype=torch.float32,device=DEVICE)
        shoot = torch.tensor(shoot,device=DEVICE).view(-1)
        act, log_prob, value, self.state = self.agent.sample_act_and_value(obs, shoot=shoot, history=self.state, timestep = None, inv=inv, reduce=True)
        return act
    def close(self):
        pass


class Population:
    def __init__(
        self,
        policy_ids: List[str],
        *args,
        **kwargs
    ):
        self._policies = {
            p_id: EvalPolicy() for p_id in policy_ids
        }
        self._policy_ids = policy_ids

        self.selected_ids = None
        self.selected_poilces = []

    def _load_policy(self, pid):
        return EvalPolicy(self.ckpt_paths, pid, self.scale)

    def prepare(self, multi_agent_ids, seed=None):
        self.finish()
        if seed is not None:
            random.seed(seed)
        self.selected_ids = [
            random.choice(self._policy_ids) for _ in range(len(multi_agent_ids))
        ]
        self.selected_poilces = [self._policies[p_id] for p_id in self.selected_ids]
        for p in self.selected_poilces:
            p.initial_state()

    def finish(self):
        for p in self.selected_poilces:
            p.close()
        del self.selected_poilces
        self.selected_ids = None
        self.selected_poilces = []
        # torch.cuda.empty_cache()

    def step(self, observations: List[Any], prev_rewards: List[Any]):
        if observations[0]["STEP_TYPE"] == 0:
            self.prepare([0])
        actions = []
        for pi, obs, prev_r in zip(self.selected_poilces, observations, prev_rewards):
            actions.append(pi.step(obs, prev_r))

        return actions

def init():
    policy_ids = [f"agent_{i}" for i in range(2)]

    mypop = Population(policy_ids)
    return mypop

mypop = init()

def my_controller(observation, action_space_list_each, is_act_continuous=False):
    """
    WARNING: KEEP AN EYE ON THIS FUNNY INPUT PARAMS!
        IF YOU NEED TO WRITE RULES PLEASE MATCH THIS INPUT
    observation: an state which in our project is a DICT
    action_space_list_each: only one action space which is wrapped into a list, :(
    
    PD_Matrix:
    observation.keys()
dict_keys(['COLLECTIVE_REWARD', 'READY_TO_SHOOT', 'RGB', 'INTERACTION_INVENTORIES', 'INVENTORY', 'WORLD.RGB', 'STEP_TYPE', 'REWARD'])
    - COLLECTIVE_REWARD and REWARD: np.float64
    - READY_TO_SHOOT: np.float64
    - observation['RGB'].shape (40, 40, 3)
    - observation['WORLD.RGB'].shape  (120, 184, 3)
    - observation['INTERACTION_INVENTORIES']
        array([[0., 0.],
            [0., 0.]])
    - observation['INVENTORY']      
        array([1., 1.])
    - observation['STEP_TYPE'] <StepType.FIRST: 0>
    action_space = [Discrete(8)]
    is_act_continuous = False
    """
    # start_time = time.time()
    # try:    
    #     actions = mypop.step([observation], [observation["REWARD"]])
    # except Exception as e:
    #     print("BUG>>>>> ", e)
    #     import pdb; pdb.post_mortem()
    actions = mypop.step([observation], [observation["REWARD"]])
        
    
    ret = []
    assert len(actions) == len(action_space_list_each)
    for act, act_space in zip(actions, action_space_list_each):
        if is_act_continuous:
            each = act
        else:
            if act_space.__class__.__name__ == "Discrete":
                # One hot encoding
                each = [0] * act_space.n
                each[act] = 1
            elif act_space.__class__.__name__ == "MultiDiscreteParticle":
                raise not NotImplementedError
            else:
                raise not NotImplementedError
        ret.append(each)
    # end_time = time.time()
    # if end_time - start_time > 0.8:
    #     print(f"Too Slow!! Time: {end_time - start_time}")
    return ret
