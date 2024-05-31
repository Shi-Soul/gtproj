import Env
import argparse
import cv2
import importlib
import gym
import torch
import sys
import numpy as np
import random

args = argparse.ArgumentParser(description="Visualize Trained Models")
args.add_argument("--agent-file",type=str,default="./Tess/saved_models/prisoners_dilemma_in_the_matrix__repeated/pd_betray_29024311.pt")
args.add_argument("--substrate",type=str,default="prisoners_dilemma_in_the_matrix__repeated")
args.add_argument("--env",type=str,default="TessEnv-v4")
args.add_argument("--model",type=str,default="impala_v4_new")
args.add_argument("--device",type=str,default="cuda")
args.add_argument("--bot",type=bool,default=False)
args.add_argument("--bot-file",type=str)
args.add_argument("--bot-model",type=str)
args.add_argument("--bot_count",type=int)

args = args.parse_args()

#Setting Up Env
env = gym.make(args.env, name=args.substrate)
obs_s = env.observation_space.shape
act_s = env.action_space.shape
num_act = env.num_act
# import pdb;pdb.set_trace()

#Setting Up Model
kwargs = {"inv":"prisoner" in args.substrate}

model = importlib.import_module(f"models.{args.model}").Model
agent = model(obs_s[1:], num_act, task_count=2, **kwargs)
agent = agent.to(args.device)
agent.load_state_dict(torch.load(f"{args.agent_file}"))

if args.bot:
    bot_model = importlib.import_module(f"models.{args.bot_model}").Model
    bot_count = args.bot_count
    bots = bot_model(obs_s[1:], num_act, **kwargs)
    bots = bots.to(args.device)
    bots.load_state_dict(torch.load(f"{args.bot_file}"))
    bot_h = torch.zeros((bot_count,256)).to(args.device)
    bot_c = torch.zeros((bot_count),256).to(args.device)

obs = env.reset()
done = False

if args.bot:
    h_n = torch.zeros((obs_s[0]-bot_count,256)).to(args.device)   
    c_n = torch.zeros((obs_s[0]-bot_count,256)).to(args.device)
else:
    h_n = torch.zeros((obs_s[0],256)).to(args.device)   
    c_n = torch.zeros((obs_s[0],256)).to(args.device)
    # For PD Matirx: 
        # (Pdb) h_n.shape
        # torch.Size([2, 256])
        # (Pdb) c_n.shape
        # torch.Size([2, 256])

if "prisoner" in args.substrate:
    inv = torch.ones((act_s[0],2), device="cuda", dtype=torch.float32) / 2
    # inv.shape: torch.Size([2, 2])
else:
    inv = False


images = []
entropies = []
rewards = []
while not done:
        
    with torch.no_grad():

        img = env.render()
        images.append(img)

        obs = torch.from_numpy(obs).to(args.device).permute((0,3,1,2))
        # obs.shape:  torch.Size([2, 3, 20, 20])
        
            # inv = torch.ones((act_s[0],2), device="cuda", dtype=torch.float32) / 2
            # inv.shape: torch.Size([2, 2])
        if "3" in args.env or "Terr" in args.env:
            time_d = torch.tensor(env.counter, device="cuda").view(1,1).expand(env.num_players,1) / 1000
        else:
            time_d = None
            
        shoot = torch.tensor(env.shoot,device="cuda").view(-1)
        # shape: torch.Size([2])
        
        #shoot = torch.zeros(env.num_players,device="cuda",dtype=torch.float32)
        if args.bot:
            raise NotImplementedError
            bot_act, (bot_h, bot_c), _ = bots.get_action(obs[:bot_count,:,:,:], shoot=shoot[:bot_count], history=(bot_h,bot_c),\
                                    timestep=time_d[:bot_count],inv=inv,reduce=True)
            act, (h_n, c_n), entropy = agent.get_action(obs[bot_count:,:,:,:], shoot=shoot[bot_count:], history=(h_n, c_n), timestep = time_d[bot_count:], inv=inv, reduce=True)
            act = torch.concatenate((bot_act,act),dim=0) 
        else:
            act, log_prob, value, (h_0, c_0) = agent.sample_act_and_value(obs, shoot=shoot, history=(h_n, c_n), timestep = time_d, inv=inv, reduce=True)

        obs, rew, done, info = env.step(act.cpu().numpy())
        if "prisoner" in args.substrate:
            inv = torch.tensor(env.invs, dtype=torch.float32,device="cuda")

images = np.array(images)
width = images[0].shape[1]
height = images[0].shape[0]

video = cv2.VideoWriter("./visualize_model.avi", 0, 6, (width,height))

for img in images:
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    video.write(img)

video.release()

# print(np.mean(entropies))
# print(np.std(entropies))
print(len(images))
print(env.real_rewards)
print(np.mean(env.real_rewards[0]))
print(np.sum(env.real_rewards))