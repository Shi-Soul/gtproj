from meltingpot import substrate
import gym
from gym.envs.registration import register
import numpy as np
import dm_env
import cv2
import queue
import random

NUM_CLEANUP_HIS = 400

class TessEnv(gym.Env):

    def __init__(self, render_mode="rgb_array", **kwargs):
        self.name = kwargs["name"]
        self.default_config = substrate.get_config(self.name)
        self.env = substrate.build(kwargs["name"], roles=self.default_config.default_player_roles)
        self.obs_spec = self.env.observation_spec()
        self.action_spec = self.env.action_spec()
        self.num_players = len(self.default_config.default_player_roles)
        self.rewards = [0] * self.num_players
        self.observation_space = gym.spaces.Box(0,1,(self.num_players,)+(self.obs_spec[0]["RGB"].shape[0]// 2,self.obs_spec[0]["RGB"].shape[1]// 2,self.obs_spec[0]["RGB"].shape[2]))
        self.num_act = self.action_spec[0].num_values
        self.action_space = gym.spaces.MultiDiscrete([self.action_spec[0].num_values for _ in range(self.num_players)])
        self.env.observables().events.subscribe(on_next=self.on_next)
        self.real_rewards = [0] * self.num_players
        self.random_player = -1
        super(TessEnv, self).__init__()
    def reset(self):
        self.done = 0
        self.reset_state = 0
        old_random = self.random_player
        while old_random == self.random_player:
            self.random_player = random.randint(0,self.num_players-1)
        #Clean up specific
        if self.name == "clean_up":
            self.clean_up_histories = [queue.Queue(maxsize=NUM_CLEANUP_HIS) for _ in range(self.num_players)]
            for ind,i in enumerate(self.clean_up_histories):
                for _ in range(NUM_CLEANUP_HIS):
                    i.put(0)
                    
                    
            self.clean_rewards = [0] * self.num_players

        #Harvest Specific
        if "harvest" in self.name:
            self.replant_states = [0] * self.num_players
            self.task_count = 3

        #Territory Specific
        if "territory" in self.name:
            self.stack = np.zeros((self.num_players,))
        
        #Prisoner Specific
        if "prisoner" in self.name:
            self._pd_seeit = [False] * self.num_players

        res = self.env.reset()

        self.shoot = [int(i["READY_TO_SHOOT"]) for i in res.observation]
        self.last_rgb = res.observation[0]["WORLD.RGB"]
        self.full_obs = cv2.resize(np.array(self.last_rgb / 255,dtype=np.float32), (self.last_rgb.shape[0] // 2, self.last_rgb.shape[1] // 2), interpolation=cv2.INTER_AREA)
        obs_list = [cv2.resize(i["RGB"], (i["RGB"].shape[0] // 2,i["RGB"].shape[1] // 2), cv2.INTER_AREA) for i in res.observation]
        obs = np.array(obs_list)
        self.last_partial_obs = obs_list[0]
        obs = np.array(obs / 255, dtype=np.float32)
        self.real_rewards = [0] * self.num_players
        self.died = 0
        self.counter = 1
        self.no_limit = False
        return obs
    
    def convert_oar(self, rewards):

        tanh = np.tanh(rewards)
        max = np.where(tanh < 0, 0, tanh)
        min = np.where(tanh < 0, tanh, 0)
        return 5 * max + .3 * min
    
    def render(self, mode="rgb_array"):
        
        return self.last_rgb

    def step(self, action):

        self.reset_state = 0
        if "clean_up" in self.name:
            self.rewards = [[0]*2 for _ in range(self.num_players)]
        elif "harvest" in self.name:
            self.rewards = [[0]*3 for _ in range(self.num_players)]
        elif "prisoner" in self.name:
            self.rewards = [[0]*2 for _ in range(self.num_players)]
        else:
            self.rewards = [0] * self.num_players
        timestep = self.env.step(action)

        self.shoot = [int(i["READY_TO_SHOOT"]) for i in timestep.observation]
        self.timestep_o = timestep
        self.counter += 1

        #Clean up specific
        if self.name == "clean_up":
            for ind, clean_reward in enumerate(self.clean_rewards):
                self.clean_up_histories[ind].get()
                self.clean_up_histories[ind].put(clean_reward)
                
            self.clean_rewards = [0] * self.num_players


        #Inventory specific
        if "prisoner" in self.name:
            # Find opponent. If we find it, give interaction reward
            for ind,i in enumerate(timestep.observation):
                rgb_obs = i["RGB"]
                red = np.array([200,100,50])
                num_red = (rgb_obs==red).all(axis=2).sum()
                if num_red>0 and self._pd_seeit[ind]==False:
                    # print("DEBUG: Seeit!",ind,num_red)
                    # self.rewards[ind][1]+=1
                    self._pd_seeit[ind]=True
            # Set invs
            invs = [i["INVENTORY"] for i in timestep.observation]
            self.total_inv = invs
            self.invs = []
            for i in invs:
                summa = sum(i)
                self.invs.append([i[0]/summa,i[1]/summa])
            for ind,i in enumerate(timestep.reward):
                self.real_rewards[ind] += i
                #self.rewards[ind] += i

        #Territory specific
        if "territory" in self.name:
            for ind,i in enumerate(timestep.reward):
                """for u in range(self.num_players):
                    self.rewards[u] += i / 9"""
                self.real_rewards[ind] += i
        
        obs_list = [cv2.resize(i["RGB"], (i["RGB"].shape[0] // 2,i["RGB"].shape[1] // 2), cv2.INTER_AREA) for i in timestep.observation]
        obs = np.array(obs_list)
        self.last_partial_obs = obs_list[0]
        obs = np.array(obs / 255, dtype=np.float32)
        done = 1 if int(timestep.step_type) == 2 else 0
        if done:
            print(self.counter)
        if self.counter == 1001:
            done = 1
            self.counter = 1

        info=dict()
        if done:
            info = {"real_rewards":self.real_rewards}
        self.last_rgb = timestep.observation[0]["WORLD.RGB"]
        self.full_obs = cv2.resize(np.array(self.last_rgb / 255,dtype=np.float32), (self.last_rgb.shape[0] // 2, self.last_rgb.shape[1] // 2), interpolation=cv2.INTER_AREA)
        #self.rewards = self.convert_oar(self.rewards)
        #print(self.rewards)
        return obs, self.rewards , done , info
    
    def on_next(self, event):
        
        if "clean_up" in self.name:
            if event[0] == "player_cleaned":
                ind = int(event[1][2])-1
                # The more you clean, the less rewards you get when you clean
                # self.clean_rewards[ind] += .01
                # self.rewards[ind][0] += .1
                
                hist_list = sum(list(self.clean_up_histories[ind].queue)) /0.01 /5
                scaled_rew = (np.tanh(NUM_CLEANUP_HIS-  hist_list)+1)/2
                self.rewards[ind][0] += scaled_rew*0.1
                self.clean_rewards[ind] += scaled_rew*0.01
                
            elif event[0] == "edible_consumed":
                theind = int(event[1][2])-1
                # if you are hardworking, you get a little more rewards when **someone** eat
                for ind,history in enumerate(self.clean_up_histories):
                    hist_list = sum(list(history.queue)) 
                    self.rewards[ind][0] += hist_list*0.01
                self.rewards[theind][1] += .1
                self.real_rewards[theind] += 1

        elif "harvest" in self.name:
            if event[0] == "eating":
                player_id = int(event[1][-3])
                berry_id = int(event[1][-1])
                self.replant_states[player_id-1] = 0
                if berry_id == 1:
                    self.real_rewards[player_id-1] += 2
                    self.rewards[player_id-1][0] += .2
                else:
                    self.rewards[player_id-1][0] += .2
                    self.real_rewards[player_id-1] += 1
            elif event[0] == "replanting":
                player_id = int(event[1][-5])
                berry_id = int(event[1][-1])
                if berry_id == 1:
                    self.rewards[player_id-1][1] += .2
                    self.replant_states[player_id-1] = 1
                else:
                    self.rewards[player_id-1][1] -= .2
                    self.replant_states[player_id-1] = 2
            elif event[0] == "zap":
                source = int(event[1][-3])
                target_player = int(event[1][-1])
                if self.replant_states[target_player-1] == 2:
                    self.rewards[source-1][2] += .2
                else:
                    self.rewards[source-1][2] -= .2
            elif event[0] == "removal_due_to_sanctioning":
                target = int(event[1][-1])
                self.replant_states[target-1] = 0
                self.real_rewards[target-1] -= 10
                
        elif "prisoner" in self.name:

            if event[0] == "collected_resource":
                # event
                        # ('collected_resource', [b'dict', b'player_index', array(2.), b'class', array(1.)])
                # for u in range(self.num_players):
                #     self.rewards[u][0] += 0.1
                # import pdb;pdb.set_trace()
                # Heuristics:
                    # deny is better
                    # don't eat too much, 3 is good
                    
                ind = int(event[1][2])-1
                inv = self.total_inv[ind]
                # rew_timing = np.tanh(-sum(inv)+3)
                rew_timing = 0
                self.rewards[ind][0] += 1*(0.1*(inv[1]+1)/(sum(inv)+1) + rew_timing*0.03)

            if event[0] == "interaction":
                # event
                        # ('interaction', [b'dict', b'row_player_idx', array(2.), b'col_player_idx', array(1.), b'row_reward', array(2.52380952), b'col_reward', array(2.84126984), b'row_inventory', array([7., 2.]), b'col_inventory', array([5., 2.])])
                # Weiji Xie: I can't determine whether row_player is the player who initiated the interaction or the player who is being interacted with
                # But here we assume he is.
                # Our wish:
                    # if they obtain more than 2 resource, go to interact as fast as possible
                    # no matter who initiated the interaction, they all should be rewarded
                    # if they obtain less than 2 resource, go to eat resource first
                # Hence, reward composition:
                    # rew_interact = 0.3 + 0.2*(initialize the interaction)
                    # rew_timing = (f(cur_res)-2)/40, f= min(x**2,8/(sqrt(x))) (max~=5)
                    # rew_game = row_rew 
                    # rew_friendly = 0.05*(row_inv+col_inv) 
                row_ind = int(event[1][2])-1
                col_ind = int(event[1][4])-1
                row_inv = event[1][10]
                col_inv = event[1][12]
                
                row_rew = float(event[1][6])
                col_rew = float(event[1][8])
                # for u in range(self.num_players):
                #     self.rewards[u][1] += row_rew + col_rew + 0.1
                # rew_timing = lambda x: min(x**2,8/(np.sqrt(x)))
                # scaled_rew_timing = lambda x: (rew_timing(x)-2)/20
                scaled_rew_timing = lambda x: 0
                prefer_first = 0 # 0.3
                
                # Pure Betray
                self.rewards[row_ind][1] = (row_inv[1]+1)/(sum(row_inv)+1) + 0.1
                self.rewards[col_ind][1] = (col_inv[1]+1)/(sum(col_inv)+1) + 0.1
                
                # self.rewards[row_ind][1] += (0.2+prefer_first) + scaled_rew_timing(sum(row_inv)) + row_rew + 0.05*(row_rew+col_rew)
                # self.rewards[col_ind][1] += (0.2) + scaled_rew_timing(sum(col_inv)) + col_rew + 0.05*(row_rew+col_rew)
                
                self._pd_seeit = [False] * self.num_players

        elif "territory" in self.name:
            if event[0] == "claimed_unclaimed_resource":
                source_id = int(event[1][-1])-1
                #for u in range(self.num_players):
                self.rewards[source_id] += .75
                #print(1)
            if event[0] == "claimed_claimed_resource":
                source_id = int(event[1][-1])-1
                self.rewards[source_id] -= .05
            if event[0] == "removal_due_to_sanctioning":
                source_id = int(event[1][-3])-1
                target_id = int(event[1][-1])-1
                #self.rewards[source_id] += 1
            if event[0] == "destroyed_resource":
                self.rewards[int(event[1][-1])-1] -= .05
                #print(2)
                #self.rewards[target_id] -= 1
            
            
