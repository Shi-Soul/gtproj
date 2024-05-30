import random
import numpy as np
from PIL import Image
import heapq

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

# changable global variable
class Memory:
    def __init__(self):
        self.global_img = 0
        self.map_memo = []
        self.local_position = None
        self.local_direction = None
        self.explore_step = None
        self.position_memo = [] # to remember the last matches for localization
        self.apath = None # cached path for path planning towards centers
        self.betray = True
        self.betray_info = [] # 0 for not betray, 1 for betray
    
    def betray_update(self): # How to do this?
        self.betray_info = ...
        self.betray = True
        
    def reset(self):
        self.map_memo = []
        self.local_position = None
        self.local_direction = None
        self.explore_step = None
        self.position_memo = []
        self.apath = None
        self.betray_update()

memory = Memory()

DEBUG = True
def my_controller(observation, action_space, is_act_continuous=False):
    """
    observation: (['COLLECTIVE_REWARD', 'READY_TO_SHOOT', 'INVENTORY', 'RGB', 'STEP_TYPE', 'REWARD'])
    """
    global memory
    
    rgb = observation['RGB']
    rgb_grid = downsample_image(rgb, 8)
    
    if DEBUG:
        print('REWARD', observation['REWARD'])
        Image.fromarray(rgb).save('./img_logs/{:06}.png'.format(memory.global_img))
        # Image.fromarray(rgb_grid).save(f'./img_logs/{memory.global_img}_grid.png')
        memory.global_img += 1
    
    if np.sum(rgb_grid)==0: # the rgb will be all black
        if DEBUG:
            print('NEW GAME! Waiting.')
        memory.reset()
    grid_info = convert_grid_to_info(rgb_grid)
    # grid info example: [['EMPTY', 'WALL', 'EMPTY', 'BLUE', 'BLUE'], ['EMPTY', 'WALL', 'EMPTY', 'RED', 'RED'], ['EMPTY', 'EMPTY', 'EMPTY', 'RED', 'RED'], ['EMPTY', 'EMPTY', 'SELF', 'EMPTY', 'EMPTY'], ['EMPTY', 'WALL', 'EMPTY', 'EMPTY', 'WALL']]
    
    # localization phase
    if memory.local_position is None:
        return localization_phase(grid_info)
    
    # path planning phase, go to the nearest area according to betray policy
    else:
        return policy_planner(grid_info, observation)
        # agent_action = determine_action(grid_info, observation['INVENTORY'])


### Code for basic tools
def action_to_one_hot(action):
    one_hot = np.zeros(8).astype(int)
    one_hot[action] = 1
    return [one_hot.tolist()]


def downsample_image(image, block_size=8):
    """
    Downsample a 40x40x3 image by averaging over 8x8 blocks.
    
    Args:
        image: numpy array of shape (40, 40, 3)
        block_size: size of the block to average over (e.g., 8 for 8x8 blocks)
    
    Returns:
        downsampled_image: numpy array of shape (5, 5, 3)
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
            else:
                row_info.append('OTHER') # Must be another agent
        info.append(row_info)
    return info


def scan_grid_info(grid_info):
    row, col = len(grid_info), len(grid_info[0])
    scan = []
    for i in range(row):
        for j in range(col):
            scan.append(grid_info[i][j])
    return list(set(scan))


def action2movement(action, direction):
    # input is text, output is (di, dj)
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
    # input is (di, dj), output is text
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
    # Manhattan distance on a square grid
    return abs(b[0] - a[0]) + abs(b[1] - a[1])


### Code for phase 1: locolization
def info2mask(info):
    mask = np.zeros((5, 5))
    for i in range(5):
        for j in range(5):
            if info[i][j] == 'WALL':
                mask[i, j] = 1
    return mask


def find_matches(global_map, local_map, last_matches=None):
    local_map = info2mask(local_map)
    adject = [(3, 2), (2, 3), (1, 2), (2, 1)]
    matches = []
    global_map_height, global_map_width = global_map.shape

    for rotation in range(4):
        rotated_map = np.rot90(local_map, k=rotation)
        rotated_map_height, rotated_map_width = rotated_map.shape
        
        match_list = []
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
        matches.append(match_list)

    return matches # up left down right


def locolization(matchs):
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
    return True, np.array(position), direction


def localization_phase(grid_info):
    global memory
    matches = find_matches(GLOBAL_MAP, grid_info, None if len(memory.position_memo) == 0 else memory.position_memo[-1])
    memory.position_memo.append(matches)
    is_localized, local_position, direction = locolization(matches)
    if is_localized:
        memory.local_position, memory.local_direction = local_position, direction
    
    # while not is_localized, walk into empty space only
    feasible_empty = [grid_info[2][2]=='EMPTY', grid_info[3][1]=='EMPTY', grid_info[4][2]=='EMPTY', grid_info[3][3]=='EMPTY'] # up left down right
    id2action = ['FORWARD', 'STEP_LEFT', 'BACKWARD', 'STEP_RIGHT']
    feasible_list = []
    for idx, feasible in enumerate(feasible_empty):
        if feasible:
            feasible_list.append(id2action[idx])
            if not feasible_empty[3-idx]: # opposite direction is also wall
                feasible_list.append(id2action[idx]) # more willing to keep from the wall
    if memory.explore_step is not None and memory.explore_step in feasible_list:
        action = memory.explore_step # keep the last step
    else:
        action = random.choice(feasible_list)
        memory.explore_step = action
    if is_localized:
        memory.local_position += np.array(action2movement(action, memory.local_direction))
    return action_to_one_hot(ACTION_TO_IDX[action])
    

        
### Code for phase 2: path planning
def nearest_center(card_centers):
    min_dist = 1000
    nearest_center = None
    for center in card_centers:
        dist = heuristic(memory.local_position, center)
        if dist < min_dist:
            min_dist = dist
            nearest_center = center
    return nearest_center


def nearest_card(grid_info, color):
    card_pos = []
    arow, acol = memory.local_position
    row, col = len(grid_info), len(grid_info[0])
    for i in range(row):
        for j in range(col):
            if grid_info[i][j] == color:
                card_pos.append(memory(i, j))

    near_card = None
    min_dist = 1000
    for card in card_pos:
        dist = heuristic((3, 2), card) # 3, 2 is the agent relative position
        if dist < min_dist:
            min_dist = dist
            near_card = card
    r_card = np.array(3, 2) - np.array(near_card)

    if memory.local_direction == 'left':
        r_card = np.array([-r_card[1], r_card[0]])
    elif memory.local_direction == 'down':
        r_card = np.array([-r_card[0], -r_card[1]])
    elif memory.local_direction == 'right':
        r_card = np.array([r_card[1], -r_card[0]])
    
    arow, acol = memory.local_position
    return (arow + r_card[0], acol + r_card[1]) # global position of the card
    

def add_card_as_obstacle(grid_info, color): # PROBLEM: checkit
    card_pos = []
    color_map = GLOBAL_MAP.copy()
    arow, acol = memory.local_position
    row, col = len(grid_info), len(grid_info[0])
    for i in range(row):
        for j in range(col):
            if grid_info[i][j] == color:
                card_pos.append(memory(i, j))
    for pos in card_pos:
        rpos = np.array(3, 2) - np.array(pos)
        if memory.local_direction == 'left':
            rpos = np.array([-rpos[1], rpos[0]])
        elif memory.local_direction == 'down':
            rpos = np.array([-rpos[0], -rpos[1]])
        elif memory.local_direction == 'right':
            rpos = np.array([rpos[1], -rpos[0]])
        
        color_map[arow + rpos[0], acol + rpos[1]] = 1
    return color_map
    

def policy_planner(grid_info, observation):
    global memory
    
    card_centers = [(6, 10), (6, 16), (12, 10), (12, 16)]
    if not memory.betray:
        card_centers = [card_centers[0], card_centers[3]]
    else:
        card_centers = [card_centers[1], card_centers[2]]
    
    global_center = (9, 13)
    
    inventory = observation['INVENTORY']
    
    # collection phase
    if sum(inventory) == 0: # naive. update later
        desired_color = 'RED' if memory.betray else 'BLUE'
        another_color = 'BLUE' if memory.betray else 'RED'
        color_map = add_card_as_obstacle(grid_info, another_color) # ensure not collect blue card
        if desired_color in scan_grid_info(grid_info):
            memory.apath = a_star_search(memory.local_position, nearest_card(grid_info, desired_color), grid=color_map)
        else:
            memory.apath = a_star_search(memory.local_position, nearest_center(card_centers), grid=color_map)
    else: # go to center and wait
        if 'OTHER' in scan_grid_info(grid_info) and observation['READY_TO_SHOOT']:
            return action_to_one_hot(ACTION_TO_IDX['INTERACT'])
        memory.apath = a_star_search(memory.local_position, global_center)
        if len(memory.apath) == 0: # already at the center
            return action_to_one_hot(ACTION_TO_IDX['INTERACT']) if observation['READY_TO_SHOOT'] else action_to_one_hot(ACTION_TO_IDX['NOOP'])
        # The policy will fail if the opponent do not come to center
        
    next_position = memory.apath.pop(0)
    agent_action = movement2action(tuple(np.array(next_position) - np.array(memory.local_position)), memory.local_direction)
    
    memory.local_position += np.array(action2movement(agent_action, memory.local_direction))
    return action_to_one_hot(ACTION_TO_IDX[agent_action])


def a_star_search(start, goal, grid=GLOBAL_MAP):
    start = tuple(start)
    goal = tuple(goal)
    
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
