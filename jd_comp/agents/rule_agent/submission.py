# --------------------------------------------------------------------------------
# Author: Loping151(Kailing Wang)
# GitHub: https://github.com/Loping151
# Description: The script is implemented from scratch by Loping151.
# --------------------------------------------------------------------------------


import random
import numpy as np
import heapq
from scipy.stats import norm



GLOBAL_MAP = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
])

RED_CARD_COLOR = (128, 33, 53)
BLUE_CARD_COLOR = (33, 128, 109)
EMPTY_SPACE = (0, 0, 0)
WALL = (114, 114, 114)
LASER = (252, 252, 106)
AGENT_VIEW_SIZE = 5
AGENT_SELF = (20, 40, 81)
AGENT_OTHER = (80, 45, 27)
DIRECTIONS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1)
}
NAMED_ITEMS = ['EMPTY', 'WALL', 'RED', 'BLUE', 'SELF', 'OTHER'] # EMPTY 0, WALL 1
NAMED_ITEMS_IDX = {name: i for i, name in enumerate(NAMED_ITEMS)}
ACTIONS = ['NOOP', 'FORWARD', 'BACKWARD', 'STEP_LEFT', 'STEP_RIGHT', 'TURN_LEFT', 'TURN_RIGHT', 'INTERACT']
ACTION_TO_IDX = {action: i for i, action in enumerate(ACTIONS)}


# functions for update global info
def oppo_coor(coor, rw):
    '''
    coor: my coor prob; rw: reward
    return opponent coor prob
    '''
    return (rw+coor-1)/(4-coor)


def fit_gaussian(data):
    '''calculate weight gaussian params, weighted by the time'''
    wdata=[] # weighted data
    for t, d in enumerate(data):
        wdata.extend([d]*np.floor(100/(len(data)-t)).astype(int))
    mean, std_dev = norm.fit(wdata)
    return mean, std_dev


def sample_gaussian(mean, std_dev, num_samples):
    samples = np.random.normal(mean, std_dev, num_samples)
    return samples


def get_policy(coor):
    '''hardcode policy'''
    if coor < 1/7:
        target = (1, 6)
    elif coor < 1/6:
        target = (1, 5)
    elif coor < 1/5:
        target = (1, 4)
    elif coor < 1/4:
        target = (1, 3)
    elif coor < 1/3:
        target = (1, 2)
    elif coor < 2/5:
        target = (2, 3)
    elif coor < 1/2:
        target = (2, 2)
    elif coor < 3/5:
        target = (3, 2)
    elif coor < 2/3:
        target = (2, 1)
    elif coor < 3/4:
        target = (3, 1)
    elif coor < 4/5:
        target = (4, 1)
    elif coor < 5/6:
        target = (5, 1)
    else: # coor < 6/7
        target = (6, 1)
    return np.array(target)


# changable global variable
class Memory:
    def __init__(self):
        self.time_cnt = 0
        self.local_position = None
        self.local_direction = None
        self.last_move = None # last action
        self.position_memo = [] # to remember the last matches for localization
        self.apath = None # cached path for path planning towards centers
        self.collect_target = np.array([5, 1])
        self.collect_priority = None
        self.inventory = None
        self.valid_inventory = None
        self.oppo_info = [] # opponent info, coor prob
        self.memory_map = GLOBAL_MAP
        self.last_observation = None
        self.lasered = False
    
    def update_collect(self):
        diff = self.collect_target - self.inventory if self.inventory is not None else np.array([1, 1])
        if sum(diff) == 0:
            if self.collect_priority is not None:
                self.memory_map = GLOBAL_MAP
            self.collect_priority = None
            return None
        elif self.collect_priority is None:
            memory.memory_map = GLOBAL_MAP
            self.collect_priority = 'RED' if diff[0] >= diff[1] else 'BLUE'
        elif self.collect_priority == 'RED':
            if diff[1] > 0:
                self.collect_priority = 'RED'
            else:
                memory.memory_map = GLOBAL_MAP
                self.collect_priority = 'BLUE'
        elif self.collect_priority == 'BLUE':
            if diff[0] > 0:
                self.collect_priority = 'BLUE'
            else:
                memory.memory_map = GLOBAL_MAP
                self.collect_priority = 'RED'
        return self.collect_priority
    
    def policy_update(self, rw):
        if rw == 0:
            return # waiting for game
        self_coor = self.valid_inventory[0] / sum(self.valid_inventory)
        oppo_coor_prob = oppo_coor(self_coor, rw)
        self.oppo_info.append(oppo_coor_prob)
        if DEBUG:
            print('OPPO INFO', self.oppo_info)
        mean, dev = fit_gaussian(self.oppo_info)
        estm_oppo = sample_gaussian(mean, dev/2, 1) # we do not want much randomness
        best_coor = estm_oppo - dev * 2
        # this is because the dev of random or [0.1, 0.9] is around 0.33-0.38 here. 
        # For a random-like opponent, we want to always betray, that is: mean:0.5-0.35*2 -> 0
        # For a fixed oppenent, the best is to betray. I have no good plan for that, but at least we betray to betrayers.
        # For a one-time cooperator, this policy tries to copy the opponent's behavior
        self.collect_target = get_policy(best_coor)
        
    def reset(self):
        self.local_position = None
        self.local_direction = None
        self.last_move = ''
        self.position_memo = []
        self.apath = None
        self.collect_priority = None
        self.memory_map = GLOBAL_MAP
        self.lasered = False


memory = Memory()
DEBUG = 0
if DEBUG:
    from PIL import Image


def my_controller(observation, action_space, is_act_continuous=False):
    """
    observation: (['COLLECTIVE_REWARD', 'READY_TO_SHOOT', 'INVENTORY', 'RGB', 'STEP_TYPE', 'REWARD'])
    """
    global memory
    
    memory.inventory = observation['INVENTORY']
    if sum(memory.inventory) > 2:
        memory.valid_inventory = memory.inventory
    rgb = observation['RGB']
    rgb_grid = downsample_image(rgb, 8)
    grid_info = convert_grid_to_info(rgb_grid)
    memory.last_observation = grid_info
    memory.time_cnt += 1
    
    if DEBUG:
        Image.fromarray(rgb).save('./img_logs/{:06}.png'.format(memory.time_cnt))
        Image.fromarray(rgb).save('./img_logs/current.png'.format(memory.time_cnt))
        # Image.fromarray(rgb_grid).save(f'./img_logs/{memory.time_cnt}_grid.png')
    
    # detect new game or new scenario
    if observation['REWARD']>0: # the rgb will be all black
        if DEBUG:
            print('REWARD', observation['REWARD'])
            print('NEW GAME! Reset.')
        memory.reset()
        memory.policy_update(observation['REWARD'])
        return action_to_one_hot(ACTION_TO_IDX['NOOP'])
    
    elif np.sum(rgb_grid)==0:
        if DEBUG:
            print('NEW GAME! Reset.')
        memory.reset()
        return action_to_one_hot(ACTION_TO_IDX['NOOP'])
        
    # detect if move falure
    if memory.lasered and check_move_failure(grid_info):
        if DEBUG:
            print('MOVE FAILURE!')
        memory.last_move = ''
        return action_to_one_hot(ACTION_TO_IDX['NOOP'])
            
    if not check_location_correct(grid_info, memory.local_position, memory.local_direction): # a new scenario starts
        if DEBUG:
            print('Crashed!')
        memory.local_position = None
        
    if memory.time_cnt > 500:
        memory = Memory()
        
    # grid info example: [['EMPTY', 'WALL', 'EMPTY', 'BLUE', 'BLUE'], ['EMPTY', 'WALL', 'EMPTY', 'RED', 'RED'], ['EMPTY', 'EMPTY', 'EMPTY', 'RED', 'RED'], ['EMPTY', 'EMPTY', 'SELF', 'EMPTY', 'EMPTY'], ['EMPTY', 'WALL', 'EMPTY', 'EMPTY', 'WALL']]
    
    try:
        # localization phase
        if memory.local_position is None:
            return localization_phase(grid_info)
        
        # path planning phase, go to the nearest area according to betray policy
        else:
            return path_planning_phase(grid_info, observation)
        
    except Exception as e:
        # This usually happens when the agent is shot. We act but not move, thus get wrong localization
        if DEBUG:
            print('ERROR!', e)
        try:
            action = feasible_explore(grid_info, memory.local_position is not None)  
            memory.last_move = action
            return action_to_one_hot(ACTION_TO_IDX[action])
        except:
            return action_to_one_hot(ACTION_TO_IDX['NOOP'])


### Code for basic tools
def action_to_one_hot(action):
    '''Input should be action id'''
    one_hot = np.zeros(8).astype(int)
    one_hot[action] = 1
    return [one_hot.tolist()]


def downsample_image(image, block_size=8):
    """
    Downsample image by taking the mean value of each block of block_size x block_size pixels
    RGB is [40, 40, 3], return [5, 5, 3] int array
    """
    downsampled_shape = (image.shape[0] // block_size, image.shape[1] // block_size, image.shape[2])
    
    downsampled_image = np.zeros(downsampled_shape)
    
    for i in range(0, image.shape[0], block_size):
        for j in range(0, image.shape[1], block_size):
            block = image[i:i + block_size, j:j + block_size]
            block_mean = block.mean(axis=(0, 1))
            downsampled_image[i // block_size, j // block_size] = block_mean
    
    return downsampled_image.astype(np.uint8)


def convert_grid_to_info(grid):
    '''Input is downsampled image, output is grid info: list[list[str]]'''
    global memory
    
    info = []
    row, col, _ = grid.shape
    for i in range(row):
        row_info = []
        for j in range(col):
            if i==3 and j==2:
                row_info.append('SELF')
            elif np.array_equal(grid[i, j], RED_CARD_COLOR):
                row_info.append('RED')
            elif np.array_equal(grid[i, j], BLUE_CARD_COLOR):
                row_info.append('BLUE')
            elif np.array_equal(grid[i, j], WALL):
                row_info.append('WALL')
            elif np.array_equal(grid[i, j], EMPTY_SPACE):
                row_info.append('EMPTY')
            elif np.array_equal(grid[i, j], LASER):
                memory.lasered = True
                row_info.append('EMPTY')
            else:
                row_info.append('OTHER') # Must be another agent
        info.append(row_info)
    return info


def scan_grid_info(grid_info):
    '''Used to check if [ITEM] in localmap'''
    row, col = len(grid_info), len(grid_info[0])
    scan = []
    for i in range(row):
        for j in range(col):
            scan.append(grid_info[i][j])
    return list(set(scan))


def can_hit(grid_info):
    '''For simplification, we only consider the 4*3 area before agent'''
    for i in range(1, 5):
        for j in range(1, 4):
            if grid_info[i][j] == 'OTHER':
                return True
    return False


def grid_info_equal(grid_info1, grid_info2):
    '''Check if two grid info are the same'''
    row, col = len(grid_info1), len(grid_info1[0])
    for i in range(row):
        for j in range(col):
            if grid_info1[i][j] != grid_info2[i][j]:
                return False
    return True


def in_back(grid_info, target):
    '''Only check the back 1 line of the agent, used to turn agent around'''
    return target in grid_info[-1]


def action2movement(action, direction):
    '''input is text, output is (di, dj)'''
    redirect = ['up', 'left', 'down', 'right']
    if direction == 'left':
        redirect = ['left', 'down', 'right', 'up']
    if direction == 'down':
        redirect = ['down', 'right', 'up', 'left']
    if direction == 'right':
        redirect = ['right', 'up', 'left', 'down']
    act_id = {'FORWARD': redirect[0], 'STEP_LEFT': redirect[1], 'BACKWARD': redirect[2], 'STEP_RIGHT': redirect[3]}
    return DIRECTIONS[act_id[action]]
    
    
def movement2action(movement, direction):
    '''Input is (di, dj), output is text'''
    redirect = ['FORWARD', 'STEP_LEFT', 'BACKWARD', 'STEP_RIGHT']
    if direction == 'left':
        redirect = ['STEP_RIGHT', 'FORWARD', 'STEP_LEFT', 'BACKWARD']
    if direction == 'down':
        redirect = ['BACKWARD', 'STEP_RIGHT', 'FORWARD', 'STEP_LEFT']
    if direction == 'right':
        redirect = ['STEP_LEFT', 'BACKWARD', 'STEP_RIGHT', 'FORWARD']
    move_id = {DIRECTIONS['up']: redirect[0], DIRECTIONS['left']: redirect[1], DIRECTIONS['down']: redirect[2], DIRECTIONS['right']: redirect[3]}
    return move_id[movement]


def heuristic(a, b):
    '''Manhattan distance on a square grid'''
    return abs(b[0] - a[0]) + abs(b[1] - a[1])


### Code for phase 1: locolization
def info2mask(info):
    '''Convert grid info to mask, 1 is wall, 0 is empty space'''
    mask = np.zeros((5, 5))
    for i in range(5):
        for j in range(5):
            if info[i][j] == 'WALL':
                mask[i, j] = 1
    return mask


def check_location_correct(grid_info, position=None, direction=None):
    '''
    Check if the local map is correctly located in the global map.
    The game will end if timestamp is over, but the agent do not know when a new scenario starts.
    So if mislocated, we reset the memory.
    '''
    global memory    
    if position is None:
        return True
    
    grid_info_mask = info2mask(grid_info)
    ax, ay = position
    if direction == 'up':
        global_submap = GLOBAL_MAP[ax-3:ax+2, ay-2:ay+3]
    elif direction == 'left':
        global_submap = GLOBAL_MAP[ax-2:ax+3, ay-3:ay+2]
        grid_info_mask = np.rot90(grid_info_mask, k=1)
    elif direction == 'down':
        global_submap = GLOBAL_MAP[ax-1:ax+4, ay-2:ay+3]
        grid_info_mask = np.rot90(grid_info_mask, k=2)
    elif direction == 'right':
        global_submap = GLOBAL_MAP[ax-2:ax+3, ay-1:ay+4]
        grid_info_mask = np.rot90(grid_info_mask, k=3)
    return np.array_equal(global_submap, grid_info_mask)


def check_move_failure(grid_info):
    '''
    True if move failure, False if not. This is used to detect if the agent is shot.
    '''
    global memory
    if memory.local_position is None:
        return False
    
    last_action = memory.last_move
    if last_action == "TURN_LEFT":
        last_dir = {'left': 'up', 'down': 'left', 'right': 'down', 'up': 'right'}
        pos, direction = memory.local_position, last_dir[memory.local_direction] 
    elif last_action in ['FORWARD', 'BACKWARD', 'STEP_LEFT', 'STEP_RIGHT']:
        pos, direction = memory.local_position - np.array(action2movement(last_action, memory.local_direction)), memory.local_direction
    else:
        return False
    failure = check_location_correct(grid_info, pos, direction)
    if failure:
        memory.local_position, memory.local_direction = pos, direction
    return failure
        

def find_matches(global_map, local_map, last_matches=None):
    '''Hentai code for localization. Very naive.'''
    local_map = info2mask(local_map)
    adject = [(3, 2), (2, 3), (1, 2), (2, 1)]
    matches = []
    matches_all = []
    global_map_height, global_map_width = global_map.shape

    for rotation in range(4):
        rotated_map = np.rot90(local_map, k=rotation)
        rotated_map_height, rotated_map_width = rotated_map.shape
        
        match_list = []
        match_list_all = []
        for row in range(global_map_height - rotated_map_height + 1):
            for col in range(global_map_width - rotated_map_width + 1):
                sub_map = global_map[row:row + rotated_map_height, col:col + rotated_map_width]
                if np.array_equal(sub_map, rotated_map):
                    center_row_adjust, center_col_adjust = adject[rotation]
                    match = (row + center_row_adjust, col + center_col_adjust)
                    if min(match) < 3 or match[0] > 15 or match[1] > 23:
                        continue
                    if last_matches is not None:
                        for last_match in last_matches[rotation]:
                            if heuristic(np.array(match), np.array(last_match)) == 1:
                                match_list.append(match)
                                break
                    else:
                        match_list.append(match)
                    match_list_all.append(match)
        matches.append(match_list)
        matches_all.append(match_list_all)
    
    # potential error
    if len(matches[0]+matches[1]+matches[2]+matches[3]) == 0:
        matches = matches_all

    return matches # up left down right


def locolization(matchs):
    '''Return True if localized, False if not, and the position and direction'''
    directions = ['up', 'left', 'down', 'right']
    total = 0
    direction = ''
    position = None
    for i in range(4):
        total +=len(matchs[i])
        if total > 1:
            return False, None, ''
        if len(matchs[i]) == 1:
            position = matchs[i][0]
            direction = directions[i]
    if DEBUG:
        print('Localized at', position, 'at frame', len(memory.position_memo)-1)
    return True, np.array(position) if position is not None else None, direction


def feasible_explore(grid_info, is_localized=True):
    '''Explore phase, tend to walk from walls for quicker localization.'''
    global memory
    
    # while not is_localized, walk into empty space only
    feasible_empty = [grid_info[2][2]=='EMPTY', grid_info[3][1]=='EMPTY', grid_info[4][2]=='EMPTY', grid_info[3][3]=='EMPTY'] # up left down right
    id2action = ['FORWARD', 'STEP_LEFT', 'BACKWARD', 'STEP_RIGHT']
    feasible_list = []
    for idx, feasible in enumerate(feasible_empty):
        if feasible:
            feasible_list.append(id2action[idx])
            if not feasible_empty[(idx+2)%4]: # opposite direction is also wall
                for _ in range(3):
                    feasible_list.append(id2action[idx]) # more willing to keep from the wall
    if memory.last_move is not None and memory.last_move in feasible_list:
        for _ in range(3):
            feasible_list.append(memory.last_move)
        
    action = random.choice(feasible_list)
    if is_localized:
        memory.local_position += np.array(action2movement(action, memory.local_direction))
    return action


def localization_phase(grid_info):
    '''Localization phase, tend to walk from walls for quicker localization.'''
    global memory
    
    matches = find_matches(GLOBAL_MAP, grid_info, None if len(memory.position_memo) == 0 else memory.position_memo[-1])
    memory.position_memo.append(matches)
    is_localized, local_position, direction = locolization(matches)
    if is_localized:
        memory.local_position, memory.local_direction = local_position, direction
    
    action = feasible_explore(grid_info, is_localized)
    memory.last_move = action
    return action_to_one_hot(ACTION_TO_IDX[action])
    
        
### Code for phase 2: path planning
def nearest_center(card_centers):
    '''Find nearest center of cards.'''
    min_dist = 1000
    nearest_center = None
    for center in card_centers:
        dist = heuristic(memory.local_position, center)
        if dist < min_dist:
            min_dist = dist
            nearest_center = center
    return nearest_center


def nearest_card(grid_info, color):
    '''Find nearest cards in sight.'''
    card_pos = []
    arow, acol = memory.local_position
    row, col = len(grid_info), len(grid_info[0])
    for i in range(row):
        for j in range(col):
            if grid_info[i][j] == color:
                card_pos.append((i, j))

    near_card = None
    min_dist = 1000
    for card in card_pos:
        dist = heuristic((3, 2), card) # 3, 2 is the agent relative position
        if dist < min_dist:
            min_dist = dist
            near_card = card
    r_card = np.array(near_card) - np.array((3, 2))

    if memory.local_direction == 'left':
        r_card = np.array([-r_card[1], r_card[0]])
    elif memory.local_direction == 'down':
        r_card = np.array([-r_card[0], -r_card[1]])
    elif memory.local_direction == 'right':
        r_card = np.array([r_card[1], -r_card[0]])
    
    arow, acol = memory.local_position
    return (arow + r_card[0], acol + r_card[1]) # global position of the card
    

def add_card_as_obstacle(grid_info, color):
    '''Add cards as obstacles in the memory map. This map should be cached in memory, or would cause agent stuck!'''
    if type(color) == str:
        color = [color]
    card_pos = []
    color_map = memory.memory_map.copy()
    arow, acol = memory.local_position
    row, col = len(grid_info), len(grid_info[0])
    for i in range(row):
        for j in range(col):
            if grid_info[i][j] in color or grid_info[i][j] == 'OTHER': # in case agents block each other
                card_pos.append((i, j))
    for pos in card_pos:
        rpos = np.array(pos) - np.array((3, 2))
        if memory.local_direction == 'left':
            rpos = np.array([-rpos[1], rpos[0]])
        elif memory.local_direction == 'down':
            rpos = np.array([-rpos[0], -rpos[1]])
        elif memory.local_direction == 'right':
            rpos = np.array([rpos[1], -rpos[0]])
        
        color_map[arow + rpos[0], acol + rpos[1]] = 1
    return color_map
    

def path_planning_phase(grid_info, observation):
    '''Main logic for path planning.'''
    global memory
    
    target_priority = memory.update_collect()
    
        # collection phase
    if target_priority is not None:
        
        card_centers = [(6, 10), (6, 16), (12, 10), (12, 16)]
        if target_priority == 'BLUE':
            card_centers = [card_centers[0], card_centers[3]] # more blue here
        else:
            card_centers = [card_centers[1], card_centers[2]] # more red here
            
        desired_color = target_priority
        another_color = 'BLUE' if target_priority=='RED' else 'RED'
        memory.memory_map = add_card_as_obstacle(grid_info, another_color) # ensure not collect blue card
        if desired_color in scan_grid_info(grid_info): # if at RED center, we can still collect BLUE card if available
            if DEBUG:
                print('Collecting', desired_color)
            memory.apath = a_star_search(memory.local_position, nearest_card(grid_info, desired_color), grid=memory.memory_map)
        else:
            memory.apath = a_star_search(memory.local_position, nearest_center(card_centers), grid=memory.memory_map)
    else: # go to center and wait
        memory.memory_map = add_card_as_obstacle(grid_info, ['RED', 'BLUE']) # ensure not collect card
        if can_hit(grid_info) and observation['READY_TO_SHOOT']:
            memory.last_move = ''
            return action_to_one_hot(ACTION_TO_IDX['INTERACT'])
        elif 'OTHER' in scan_grid_info(grid_info): # go to
            if in_back(grid_info, 'OTHER'):
                next_direction = {'up': 'left', 'left': 'down', 'down': 'right', 'right': 'up'}
                memory.local_direction = next_direction[memory.local_direction]
                memory.last_move = 'TURN_LEFT'
                return action_to_one_hot(ACTION_TO_IDX['TURN_LEFT'])
            memory.apath = a_star_search(memory.local_position, nearest_card(grid_info, 'OTHER'), grid=memory.memory_map)
        else:
            global_center = (9, 13)
            memory.apath = a_star_search(memory.local_position, global_center, memory.memory_map)
        # The policy will fail if the opponent do not come to center
        
    if not memory.apath or len(memory.apath) == 0:
        action = feasible_explore(grid_info)
        memory.last_move = action
    else:
        next_position = memory.apath.pop(0)
        agent_action = movement2action(tuple(np.array(next_position) - np.array(memory.local_position)), memory.local_direction)
        
        memory.local_position += np.array(action2movement(agent_action, memory.local_direction))
        memory.last_move = agent_action
        
    return action_to_one_hot(ACTION_TO_IDX[agent_action])


def a_star_search(start, goal, grid=GLOBAL_MAP):
    '''This function is generated with GPT and slightly modified. It main contain bugs but I haven't found any.'''
    start = tuple(start)
    goal = tuple(goal)
    grid[goal] = 0
    
    neighbors = [(0,1), (1,0), (0,-1), (-1,0)]  # Move right, down, left, up
    close_set = set()
    came_from = {}
    gscore = {start: 0}
    fscore = {start: heuristic(start, goal)}
    oheap = []

    heapq.heappush(oheap, (fscore[start], start))
    
    while oheap:
        current = heapq.heappop(oheap)[1]

        if current == goal:
            data = []
            while current in came_from:
                data.append(current)
                current = came_from[current]
            return data[::-1]

        close_set.add(current)
        for i, j in neighbors:
            neighbor = current[0] + i, current[1] + j            
            tentative_g_score = gscore[current] + heuristic(current, neighbor)
            if 0 <= neighbor[0] < grid.shape[0]:
                if 0 <= neighbor[1] < grid.shape[1]:                
                    if grid[neighbor[0]][neighbor[1]] == 1:
                        continue
                else:
                    # grid bounds exceeded
                    continue
            else:
                # grid bounds exceeded
                continue

            if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                continue

            if  tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1]for i in oheap]:
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(oheap, (fscore[neighbor], neighbor))
                
    return False # this should not happen
