from typing import Tuple, List, Any
import random
import torch
import numpy as np

import os
import cv2
from pathlib import Path
import sys
import torch

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
        # agent_file=AGENT_FILE
        self.agent = get_agent(agent_file)

    def initial_state(self):
        """See base class."""
        h_n = torch.zeros((1,256)).to(DEVICE)   
        c_n = torch.zeros((1,256)).to(DEVICE)
        
        self.state = (h_n, c_n)
        return self.state

    def step(self, timestep, prev_state: Any=None):
        # Need: obs(torch.Size([2, 3, 20, 20])), shoot, inv, 
        observation = timestep.observation
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
        return act, None
    def close(self):
        pass