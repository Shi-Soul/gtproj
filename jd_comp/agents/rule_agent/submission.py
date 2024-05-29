import random
import numpy as np
from PIL import Image

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
        self.position_memo = []
    
    def reset(self):
        self.global_img = 0
        self.map_memo = []
        self.local_position = None
        self.local_direction = None
        self.position_memo = []

memory = Memory()


def my_controller(observation, action_space, is_act_continuous=False):
    """
    observation: (['COLLECTIVE_REWARD', 'READY_TO_SHOOT', 'INVENTORY', 'RGB', 'STEP_TYPE', 'REWARD'])
    """
    global memory
    rgb = observation['RGB']
    Image.fromarray(rgb).save(f'./img_logs/{memory.global_img}.png')
    rgb_grid = downsample_image(rgb, 8)
    # Image.fromarray(rgb_grid).save(f'./img_logs/{memory.global_img}_grid.png')
    memory.global_img += 1
    
    if observation['STEP_TYPE'].FIRST == observation['STEP_TYPE']:
        memory.reset()
    grid_info = convert_grid_to_info(rgb_grid)
    # grid info example: [['EMPTY', 'WALL', 'EMPTY', 'BLUE', 'BLUE'], ['EMPTY', 'WALL', 'EMPTY', 'RED', 'RED'], ['EMPTY', 'EMPTY', 'EMPTY', 'RED', 'RED'], ['EMPTY', 'EMPTY', 'SELF', 'EMPTY', 'EMPTY'], ['EMPTY', 'WALL', 'EMPTY', 'EMPTY', 'WALL']]
    
    # localization phase
    if memory.local_position is None:
        return localization_phase(grid_info)
    
    # path planning phase, go to the nearest area according to betray policy
    
    agent_action = determine_action(grid_info, observation['INVENTORY'])
    
    return action_to_one_hot(agent_action)


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
                    if last_matches is not None:
                        for last_match in last_matches[rotation]:
                            if np.sum(np.abs(np.array(match) - np.array(last_match))) == 1:
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
    return True, position, direction


def localization_phase(grid_info):
    global memory
    matches = find_matches(GLOBAL_MAP, grid_info, None if len(memory.position_memo) == 0 else memory.position_memo[-1])
    memory.position_memo.append(matches)
    is_localized, local_position, direction = locolization(matches)
    if is_localized:
        memory.local_position, memory.local_direction = local_position, direction
    
    # while not is_localized, walk into empty space only
    feasible_empty = [grid_info[2][2]=='EMPTY', grid_info[3][1]=='EMPTY', grid_info[4][2]=='EMPTY', grid_info[3][3]=='EMPTY'] # up left down right
    id2action = {0: 'FORWARD', 1: 'STEP_LEFT', 2: 'BACKWARD', 3: 'STEP_RIGHT'}
    feasible_list = []
    for idx, feasible in enumerate(feasible_empty):
        if feasible:
            feasible_list.append(id2action[idx])
            if not feasible_empty[3-idx]: # opposite direction is also wall
                feasible_list.append(id2action[idx]) # more willing to keep from the wall
    action_id = random.choice(feasible_list)
    return action_to_one_hot(ACTION_TO_IDX[action_id])
    

        
### Code for phase 2: path planning

def get_neighbours(cell):
    i, j = cell
    return [(i + di, j + dj) for di, dj in DIRECTIONS.values() if 0 <= i + di < AGENT_VIEW_SIZE and 0 <= j + dj < AGENT_VIEW_SIZE]

def shortest_path_to_goal(start, goal, walls, bad_cells):
    visited = set()
    queue = [(start, [])]
    while queue:
        (i, j), path = queue.pop(0)
        if (i, j) == goal:
            return path
        if (i, j) in visited:
            continue
        visited.add((i, j))
        for neighbour in get_neighbours((i, j)):
            if neighbour not in walls and neighbour not in bad_cells and neighbour not in visited:
                queue.append((neighbour, path + [neighbour]))
    return []

def get_direction_to_goal(goal, walls, bad_cells):
    start = (3, 2)
    path = shortest_path_to_goal(start, goal, walls, bad_cells)
    if not path:
        return None
    next_cell = path[0]
    for action, (di, dj) in DIRECTIONS.items():
        if (start[0] + di, start[1] + dj) == next_cell:
            return action
    return None

def direction_to_number(direction):
    mapping = {
        "up": 1,
        "down": 2,
        "left": 3,
        "right": 4,
        "same_location": 0
    }
    return mapping.get(direction, 0)

def get_nearest(pixel_positions):
    pixel_positions = [pixel for pixel in pixel_positions if pixel != (3, 2)]
    if len(pixel_positions) == 0:
        return None
    pixels = pixel_positions
    rows = np.array([pixel[0] for pixel in pixels])
    cols = np.array([pixel[1] for pixel in pixels])
    distances = np.sqrt((rows - 3)**2 + (cols - 2)**2)
    min_index = np.argmin(distances)
    nearest_pixel = pixels[min_index]
    return nearest_pixel

def interactable_player_in_zap_range(interactable_players, walls, players, goal_cells):
    for player in interactable_players:
        if player == (2, 2):
            return True
        if player == (3, 1):
            return True
        if player == (3, 3):
            return True
        if player == (1, 1) and not (set(walls + players + goal_cells) & set([(3, 1), (2, 1)])):
            return True
        if player == (2, 1) and not (set(walls + players + goal_cells) & set([(3, 1)])):
            return True
        if player == (1, 3) and not (set(walls + players + goal_cells) & set([(2, 3), (3, 3)])):
            return True
        if player == (2, 3) and not (set(walls + players + goal_cells) & set([(3, 3)])):
            return True
        if player == (0, 2) and not (set(walls + players + goal_cells) & set([(1, 2), (2, 2)])):
            return True
        if player == (1, 2) and not (set(walls + players + goal_cells) & set([(2, 2)])):
            return True

def determine_action(grid_info, inventory):
    red_card_cells = []
    blue_card_cells = []
    walls = []
    empty_space = []
    players = []
    interactable_players = []
    noninteractable_players = []
    for i in range(AGENT_VIEW_SIZE):
        for j in range(AGENT_VIEW_SIZE):
            if grid_info[i][j] == 'RED':
                red_card_cells.append((i, j))
            elif grid_info[i][j] == 'BLUE':
                blue_card_cells.append((i, j))
            elif grid_info[i][j] == 'WALL':
                walls.append((i, j))
            elif grid_info[i][j] == 'EMPTY':
                empty_space.append((i, j))
            elif grid_info[i][j] == 'SELF':
                continue
            else:
                players.append((i, j))
                if grid_info[i][j] == 'OTHER':
                    interactable_players.append((i, j))
                else:
                    noninteractable_players.append((i, j))
    
    if (inventory[0] > 2 or inventory[1] > 2) and interactable_players:
        nearest_player = get_nearest(interactable_players)
        if interactable_player_in_zap_range(interactable_players, walls, players, blue_card_cells + red_card_cells):
            return 7
        if nearest_player[1] < 2 and nearest_player[0] > 1:
            return 5
        if nearest_player[1] > 3 and nearest_player[0] > 1:
            return 6
        nearest_player = (nearest_player[0] + 1, nearest_player[1])
        direction = get_direction_to_goal(nearest_player, walls + noninteractable_players, [])
        if direction:
            return direction_to_number(direction)

    if red_card_cells:
        nearest_goal = get_nearest(red_card_cells)
        direction = get_direction_to_goal(nearest_goal, walls + players, [])
        if direction:
            return direction_to_number(direction)

    if blue_card_cells:
        nearest_goal = get_nearest(blue_card_cells)
        direction = get_direction_to_goal(nearest_goal, walls + players, [])
        if direction:
            return direction_to_number(direction)

    action = random.choice([1, 1, 1, 5, 6])
    if action == 1 and grid_info[3][2] == 'WALL':
        action = random.choice([5, 6])
    return action