import os
import random
import math
import numpy as np
import torch
import matplotlib.pyplot as plt

from tqdm import tqdm 
from collections import deque

from game import Game 
from agents.ppo_agent import PPOAgent 
from gridworld_generator import HotelGenerator 

# --- STATE REPRESENTATION FOR PPO ---
NUM_STATE_HISTORY_FRAMES = 3 # T, T-1, T-2
# Player History: Channels 0 (T), 1 (T-1), 2 (T-2)
# Owner History: Channels 3 (T), 4 (T-1), 5 (T-2)
PLAYER_HISTORY_CHANNELS_START = 0
OWNER_HISTORY_CHANNELS_START = PLAYER_HISTORY_CHANNELS_START + NUM_STATE_HISTORY_FRAMES
# Static Map:
# Channel 6: Walls
# Channel 7: Doors
# Channel 8: Hallways
# Channel 9: Room Floors
# Channel 10: Hiding Spots
# Channel 11: Exit Position
STATIC_CHANNELS_START = OWNER_HISTORY_CHANNELS_START + NUM_STATE_HISTORY_FRAMES
WALL_CHANNEL_OFFSET = 0
DOOR_CHANNEL_OFFSET = 1
HALLWAY_CHANNEL_OFFSET = 2
ROOM_FLOOR_CHANNEL_OFFSET = 3
HIDING_SPOT_CHANNEL_OFFSET = 4
EXIT_CHANNEL_OFFSET = 5
NUM_STATIC_MAP_FEATURES = 6 # Number of unique static features
# Sounds:
# Channel 12: Sounds
SOUND_CHANNEL_OFFSET = STATIC_CHANNELS_START + NUM_STATIC_MAP_FEATURES

NUM_PPO_CHANNELS = SOUND_CHANNEL_OFFSET + 1 # Total channels: 3+3 for history + 6 static + 1 sound = 13

SOUND_PERCEPTION_RADIUS = 8
PLAYER_VISION_RADIUS = 3
OWNER_VISION_RADIUS = 7

def precompute_static_map_representation(game_instance, grid_height, grid_width):
    """
    Creates a multi-channel numpy array representation of the static game map elements.
    Populates channels for Walls, Doors, Hallways, Room Floors, Hiding Spots, Exit.
    Output shape: (NUM_PPO_CHANNELS, grid_height, grid_width)
    """
    static_state_tensor_np = np.zeros((NUM_PPO_CHANNELS, grid_height, grid_width), dtype=np.float32)

    for r_idx in range(grid_height):
        for c_idx in range(grid_width):
            tile = game_instance.grid[r_idx][c_idx]
            if tile == HotelGenerator.WALL:
                static_state_tensor_np[STATIC_CHANNELS_START + WALL_CHANNEL_OFFSET, r_idx, c_idx] = 1.0
            elif tile == HotelGenerator.DOOR:
                static_state_tensor_np[STATIC_CHANNELS_START + DOOR_CHANNEL_OFFSET, r_idx, c_idx] = 1.0
            elif tile == HotelGenerator.HALLWAY:
                static_state_tensor_np[STATIC_CHANNELS_START + HALLWAY_CHANNEL_OFFSET, r_idx, c_idx] = 1.0
            elif tile == HotelGenerator.ROOM_FLOOR:
                static_state_tensor_np[STATIC_CHANNELS_START + ROOM_FLOOR_CHANNEL_OFFSET, r_idx, c_idx] = 1.0
            elif tile == HotelGenerator.HIDING_SPOT:
                static_state_tensor_np[STATIC_CHANNELS_START + HIDING_SPOT_CHANNEL_OFFSET, r_idx, c_idx] = 1.0

    # Channel for Exit Position
    if game_instance.exit_pos:
        r_e, c_e = game_instance.exit_pos
        if 0 <= r_e < grid_height and 0 <= c_e < grid_width:
            static_state_tensor_np[STATIC_CHANNELS_START + EXIT_CHANNEL_OFFSET, r_e, c_e] = 1.0
    return static_state_tensor_np

def get_dqn_state_representation(game_instance, grid_height, grid_width, static_map_representation_np, owner_pos_history, player_pos_history):
    """
    Creates a multi-channel tensor representation of the game state for the PPO agent,
    using a precomputed static map representation and adding dynamic elements including history.
    Output shape: (NUM_PPO_CHANNELS, grid_height, grid_width)
    """
    # Start with a copy of the precomputed static map features
    state_tensor_np = static_map_representation_np.copy()

    # Clear dynamic channels before populating them
    for i in range(NUM_STATE_HISTORY_FRAMES):
        state_tensor_np[PLAYER_HISTORY_CHANNELS_START + i, :, :] = 0.0  # Player history channels
        state_tensor_np[OWNER_HISTORY_CHANNELS_START + i, :, :] = 0.0  # Owner history channels
    state_tensor_np[SOUND_CHANNEL_OFFSET, :, :] = 0.0  # Sounds channel

    # Player Position History (Channels 0, 1, 2 for T, T-1, T-2)
    current_player_pos = game_instance.player_pos
    player_history_to_fill = [current_player_pos] + list(player_pos_history) # player_pos_history is [T-1, T-2]

    for i in range(NUM_STATE_HISTORY_FRAMES):
        pos = player_history_to_fill[i]
        if pos:
            r_p, c_p = pos
            # For player history, we use actual position, not just if visible to owner.
            if 0 <= r_p < grid_height and 0 <= c_p < grid_width:
                state_tensor_np[PLAYER_HISTORY_CHANNELS_START + i, r_p, c_p] = 1.0
    
    # Owner Position History (Channels 3, 4, 5 for T, T-1, T-2)
    current_owner_pos = game_instance.owner_pos
    owner_history_to_fill = [current_owner_pos] + list(owner_pos_history) # owner_pos_history is [T-1, T-2]

    for i in range(NUM_STATE_HISTORY_FRAMES):
        pos = owner_history_to_fill[i]
        if pos:
            r_o, c_o = pos
            if 0 <= r_o < grid_height and 0 <= c_o < grid_width:
                state_tensor_np[OWNER_HISTORY_CHANNELS_START + i, r_o, c_o] = 1.0
    
    # Sounds Channel (Channel 12)
    if game_instance.owner_pos: # Sound perception is relative to owner
        orow, ocol = game_instance.owner_pos
        for sound_event in game_instance.sound_alerts:
            if 'DOOR' in sound_event['type'].upper(): 
                srow, scol = sound_event['pos']
                if (srow - orow)**2 + (scol - ocol)**2 <= SOUND_PERCEPTION_RADIUS ** 2 and game_instance._is_valid_pos(srow, scol):
                    state_tensor_np[SOUND_CHANNEL_OFFSET, srow, scol] = 1.0
                 
    return torch.from_numpy(state_tensor_np)

# --- END STATE REPRESENTATION ---

def bfs_hallway_pathfinding(game_instance, start_pos, target_pos):
    """
    Finds the shortest path from start_pos to target_pos using only hallway tiles
    or the target_pos itself.
    The player is assumed to be starting on a hallway or door tile.
    Returns a list of action constants (e.g., [Game.MOVE_NORTH, ...]) or None if no path.
    """
    queue = deque([(start_pos, [])])  # Stores (current_position, path_actions_to_here)
    visited = {start_pos}
    
    grid_height = game_instance.generator.height
    grid_width = game_instance.generator.width

    while queue:
        (current_r, current_c), path = queue.popleft()

        # Check if the current tile is the target (should not happen if called when start != target, but good check)
        # The main check for target is done when considering next_pos

        for action, (dr, dc) in Game.ACTION_DELTAS.items():
            if action == Game.WAIT: # Pathfinding doesn't involve waiting as a step
                continue

            next_r, next_c = current_r + dr, current_c + dc

            # Check if next position is the target
            if (next_r, next_c) == target_pos:
                # Ensure the target is walkable (e.g., an EXIT tile)
                if game_instance._is_walkable(next_r, next_c, "PLAYER"):
                    return path + [action] # Path found
                else:
                    continue # Target is not walkable (e.g. a wall, though target_pos should be an exit)

            if not (0 <= next_r < grid_height and 0 <= next_c < grid_width):
                continue  # Out of bounds

            if (next_r, next_c) in visited:
                continue

            # Player can only move through hallways or directly onto the exit from a valid tile
            if game_instance._is_walkable(next_r, next_c, "PLAYER"):
                tile_type_next = game_instance._get_tile(next_r, next_c)
                
                # Valid path tiles are hallways. The exit is handled above.
                if tile_type_next == HotelGenerator.HALLWAY:
                    visited.add((next_r, next_c))
                    new_path = path + [action]
                    queue.append(((next_r, next_c), new_path))
    
    return None # No path found

def train_agent():
    # --- DIRECTORY SETUP ---
    MODEL_DIR = "models"
    PLOTS_DIR = "plots"
    LOGS_DIR = "logs"
    LOGGING_FREQUENCY = 500
    PLOTTING_FREQUENCY = 500
    MA_WINDOW = 100
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    # --- END DIRECTORY SETUP ---

    # Game parameters
    GRID_HEIGHT = 15
    GRID_WIDTH = 15
    PLAYER_MOVES_PER_TURN = 1

    game_params_phase1 = {
        'straightness_hallways': 0.9,
        'hall_loops': 2,
        'max_hallway_perc': 0.04,
        'max_rooms': 0,
        'room_min_size': 2,
        'room_max_size': 2,
        'max_hiding_spots_per_room': 0,
    }

    game_params_phase3 = {
        'straightness_hallways': 0.9,
        'hall_loops': 2,
        'max_hallway_perc': 0.05,
        'max_rooms': 1,
        'room_min_size': 2,
        'room_max_size': 3,
        'max_hiding_spots_per_room': 0,
    }

    game_params_phase4 = {
        'straightness_hallways': 0.9,
        'hall_loops': 2,
        'max_hallway_perc': 0.09,
        'max_rooms': 2,
        'room_min_size': 2,
        'room_max_size': 3,
        'max_hiding_spots_per_room': 0,
    }

    # Training parameters for PPO
    NUM_EPISODES = 5000
    MAX_TURNS_PER_EPISODE = 100
    LEARNING_RATE = 0.0005
    GAMMA = 0.99 # Discount factor for PPO
    GAE_LAMBDA = 0.95
    PPO_CLIP_EPSILON = 0.2
    PPO_EPOCHS = 10
    MINI_BATCH_SIZE = 64 
    ENTROPY_COEFFICIENT = 0.05
    VALUE_LOSS_COEFFICIENT = 1
    UPDATE_TIMESTEPS = 2048 # Number of owner steps to collect before PPO update
    PHASE_2_START_EPISODE = 1000
    PHASE_3_START_EPISODE = 2000
    PHASE_4_START_EPISODE = 4000

    # Learning rate decay (can be used with PPO optimizer)
    LEARNING_RATE_DECAY_FACTOR = 0.95
    LEARNING_RATE_DECAY_FREQUENCY = 500 
    MIN_LEARNING_RATE = 0.00001
    current_learning_rate = LEARNING_RATE

    # Rewards
    REWARD_CATCH_PLAYER = 50
    REWARD_PLAYER_ESCAPES = -50
    REWARD_BUMP_WALL = -5
    REWARD_IN_ROOM = -2
    REWARD_PROXIMITY = 30
    PROXIMITY_THRESHOLD = 6
    REWARD_PLAYER_VISIBLE = 0 
    REWARD_BASE = -0.5 
    REWARD_PER_STEP = 5
    REWARD_FIRST_DOOR = 30

    game_actions_list = [Game.MOVE_NORTH, Game.MOVE_SOUTH, Game.MOVE_EAST, Game.MOVE_WEST, Game.WAIT]

    owner_agent = PPOAgent(
        game_actions_list=game_actions_list,
        input_channels=NUM_PPO_CHANNELS, 
        grid_height=GRID_HEIGHT,
        grid_width=GRID_WIDTH,
        learning_rate=current_learning_rate,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        ppo_clip_epsilon=PPO_CLIP_EPSILON,
        ppo_epochs=PPO_EPOCHS,
        mini_batch_size=MINI_BATCH_SIZE,
        entropy_coefficient=ENTROPY_COEFFICIENT,
        value_loss_coefficient=VALUE_LOSS_COEFFICIENT
    )

    model_load_path = os.path.join(MODEL_DIR, "ppo_owner_agent.pth")
    owner_agent.load_model(model_load_path)

    total_rewards_per_episode = []
    moving_avg_rewards_per_episode = []
    
    avg_policy_loss_log = []
    avg_value_loss_log = []
    avg_entropy_loss_log = []

    total_env_steps = 0 # Tracks owner steps for PPO update

    print("Starting PPO training...")

    with tqdm(range(NUM_EPISODES), unit="episode") as episode_pbar:
        for episode in episode_pbar:
            is_phase_1 = episode < PHASE_2_START_EPISODE
            is_phase_3 = episode >= PHASE_3_START_EPISODE and episode < PHASE_4_START_EPISODE
            is_phase_4 = episode >= PHASE_4_START_EPISODE
            current_game_params = game_params_phase3 if is_phase_3 else game_params_phase1
            game = Game(
                width=GRID_WIDTH, 
                height=GRID_HEIGHT, 
                player_moves_per_turn=PLAYER_MOVES_PER_TURN, 
                player_vision_radius=0 if not is_phase_3 else PLAYER_VISION_RADIUS,
                owner_vision_radius=OWNER_VISION_RADIUS,
                generator_params=current_game_params)
            owner_pos_history = deque([game.owner_pos] * NUM_STATE_HISTORY_FRAMES, maxlen=NUM_STATE_HISTORY_FRAMES) 
            player_pos_history = deque([game.player_pos] * NUM_STATE_HISTORY_FRAMES, maxlen=NUM_STATE_HISTORY_FRAMES)
            player_action_queue = deque() 
            player_knows_exit_pos = None 
            last_player_direction = None  
            log_file_handle = None
            current_log_filename = None
            first_door_reward_given = False
            
            if (episode + 1) % LOGGING_FREQUENCY == 0:
                current_log_filename = os.path.join(LOGS_DIR, f"ppo_episode_trace_ep{episode+1}.txt") 
                log_file_handle = open(current_log_filename, "w")

            def do_log(message):
                if log_file_handle:
                    log_file_handle.write(message)

            do_log(f"--- Episode {episode + 1} Initial State ---\n")
            do_log(game.render_grid_to_string(player_pov=False) + "\n\n")

            precomputed_static_map_np = precompute_static_map_representation(game, GRID_HEIGHT, GRID_WIDTH)
            current_owner_state_tensor = get_dqn_state_representation(game, GRID_HEIGHT, GRID_WIDTH, precomputed_static_map_np, owner_pos_history, player_pos_history) # Shape (C, H, W)
            
            episode_reward = 0
            game_turn_count = 0 # Owner turns in current episode
            episode_wall_bumps = 0

            effective_reward_catch_player = 0 if is_phase_1 else REWARD_CATCH_PLAYER
            effective_reward_player_escapes = 0 if is_phase_1 else REWARD_PLAYER_ESCAPES
            effective_reward_proximity = 0 if is_phase_1 else REWARD_PROXIMITY
            effective_reward_base = 0 if is_phase_1 else REWARD_BASE
            effective_reward_per_step = REWARD_PER_STEP if is_phase_1 else 0
            
            # --- GAME LOOP ---
            # Loop for a maximum number of total game steps or owner turns
            for _ in range(MAX_TURNS_PER_EPISODE * (PLAYER_MOVES_PER_TURN + 1)): 
                if game.game_over or game_turn_count >= MAX_TURNS_PER_EPISODE:
                    break

                current_entity = game.get_current_turn_entity()
                player_action_count_in_turn = game.player_moves_taken_this_turn + 1

                if current_entity == "PLAYER":
                    chosen_player_action = Game.WAIT 

                    player_r, player_c = game.player_pos
                    current_player_tile_type = game._get_tile(player_r, player_c)
                    
                    player_vision, _ = game.get_player_vision_data()
                    owner_seen_at = None
                    for loc, entity_types_in_vision in player_vision.items(): 
                        if "OWNER" in entity_types_in_vision:
                            owner_seen_at = loc
                            break
                    
                    if game.exit_pos and game.exit_pos in player_vision and HotelGenerator.EXIT in player_vision[game.exit_pos]:
                         player_knows_exit_pos = game.exit_pos

                    # --- FLEEING LOGIC ---
                    if owner_seen_at:
                        player_action_queue.clear() 
                        flee_actions_with_dist = []
                        for action_key, (dr, dc) in Game.ACTION_DELTAS.items():
                            if action_key == Game.WAIT and len(flee_actions_with_dist) > 0 : continue 
                            next_r, next_c = player_r + dr, player_c + dc
                            if game._is_walkable(next_r, next_c, "PLAYER"):
                                dist_to_owner = math.sqrt((next_r - owner_seen_at[0])**2 + (next_c - owner_seen_at[1])**2)
                                flee_actions_with_dist.append({'action': action_key, 'dist': dist_to_owner})
                        if flee_actions_with_dist:
                            flee_actions_with_dist.sort(key=lambda x: x['dist'], reverse=True) 
                            if flee_actions_with_dist: chosen_player_action = flee_actions_with_dist[0]['action']
                    # --- ACTION QUEUE LOGIC ---
                    elif player_action_queue:
                        action_from_queue = player_action_queue.popleft()
                        q_dr, q_dc = Game.ACTION_DELTAS[action_from_queue]
                        q_next_r, q_next_c = player_r + q_dr, player_c + q_dc

                        if game._is_walkable(q_next_r, q_next_c, "PLAYER"):
                            q_next_tile_type = game._get_tile(q_next_r, q_next_c)
                            is_consistent_hallway_move = not player_knows_exit_pos or \
                                                        (q_next_tile_type == HotelGenerator.HALLWAY or \
                                                         q_next_tile_type == HotelGenerator.DOOR or \
                                                         (q_next_r, q_next_c) == player_knows_exit_pos) 
                            
                            if is_consistent_hallway_move:
                                chosen_player_action = action_from_queue
                                do_log(f"Player using action '{chosen_player_action}' from queue. {len(player_action_queue)} actions remaining.\n")
                            else:
                                do_log(f"Queued action '{action_from_queue}' leads to inconsistent tile '{q_next_tile_type}' for known exit path. Clearing queue.\n")
                                player_action_queue.clear() 
                        else:
                            do_log(f"Queued action '{action_from_queue}' is no longer walkable. Clearing queue.\n")
                            player_action_queue.clear() 

                    # --- EXIT LOGIC ---
                    elif player_knows_exit_pos is not None and game.exit_pos is not None: 
                        global_target_exit_pos = player_knows_exit_pos 
                        current_room_obj = game.get_room_player_is_in(player_r, player_c)
                        
                        if current_room_obj and current_player_tile_type == HotelGenerator.ROOM_FLOOR:
                            room_door = game.get_door_for_room(current_room_obj)
                            if room_door:
                                target_door_pos_on_grid = room_door['door_pos']
                                actions_to_door = []
                                for action_key, (dr, dc) in Game.ACTION_DELTAS.items():
                                    next_r, next_c = player_r + dr, player_c + dc
                                    if game.is_pos_in_room_or_door(next_r, next_c, current_room_obj, target_door_pos_on_grid) and \
                                       game._is_walkable(next_r, next_c, "PLAYER"): 
                                        dist_to_door = math.sqrt((next_r - target_door_pos_on_grid[0])**2 + (next_c - target_door_pos_on_grid[1])**2)
                                        actions_to_door.append({'action': action_key, 'dist': dist_to_door, 'next_pos':(next_r,next_c)})
                                if actions_to_door: 
                                    actions_to_door.sort(key=lambda x: x['dist'])
                                    min_d = actions_to_door[0]['dist']
                                    best_options = [opt for opt in actions_to_door if opt['dist'] == min_d]
                                    non_wait_best_options = [opt for opt in best_options if opt['action'] != Game.WAIT]
                                    if non_wait_best_options:
                                        chosen_player_action = random.choice(non_wait_best_options)['action']
                                    elif best_options: 
                                        chosen_player_action = best_options[0]['action'] 
                            else: 
                                valid_room_moves = [] 
                                for p_act_key, (dr_p, dc_p) in Game.ACTION_DELTAS.items():
                                    next_r_p, next_c_p = player_r + dr_p, player_c + dc_p
                                    if game.is_pos_in_room_or_door(next_r_p, next_c_p, current_room_obj, None) and \
                                       game._is_walkable(next_r_p, next_c_p, "PLAYER"): 
                                        valid_room_moves.append(p_act_key)
                                if Game.WAIT not in valid_room_moves: valid_room_moves.append(Game.WAIT) 
                                if valid_room_moves: chosen_player_action = random.choice(valid_room_moves)

                        elif current_player_tile_type == HotelGenerator.HALLWAY or \
                             current_player_tile_type == HotelGenerator.DOOR or \
                             (player_r, player_c) == global_target_exit_pos: 

                            if (player_r, player_c) == global_target_exit_pos: 
                                chosen_player_action = Game.WAIT 
                            else:                                
                                path_to_exit = bfs_hallway_pathfinding(game, (player_r, player_c), global_target_exit_pos)
                                if path_to_exit and len(path_to_exit) > 0:
                                    player_action_queue.extend(path_to_exit)
                                    do_log(f"BFS success. Path: {path_to_exit}. Queue populated with {len(player_action_queue)} actions.\n")
                                    if player_action_queue: 
                                        chosen_player_action = player_action_queue.popleft()
                                        do_log(f"Player taking first action '{chosen_player_action}' from new BFS path. {len(player_action_queue)} actions remaining.\n")
                                else: 
                                    do_log(f"BFS FAILED or empty path to {global_target_exit_pos}. Fallback hallway movement.\n")
                                    fallback_options = []
                                    for action_key, (dr_fb, dc_fb) in Game.ACTION_DELTAS.items():
                                        next_r_fb, next_c_fb = player_r + dr_fb, player_c + dc_fb
                                        if game._is_walkable(next_r_fb, next_c_fb, "PLAYER"):
                                            tile_type_fallback_target = game._get_tile(next_r_fb, next_c_fb)
                                            if tile_type_fallback_target == HotelGenerator.HALLWAY or \
                                               tile_type_fallback_target == HotelGenerator.DOOR or \
                                               (next_r_fb, next_c_fb) == global_target_exit_pos: 
                                                fallback_options.append(action_key)
                                    if fallback_options:
                                        non_wait_options = [a for a in fallback_options if a != Game.WAIT]
                                        if non_wait_options: chosen_player_action = random.choice(non_wait_options)
                                        elif Game.WAIT in fallback_options: chosen_player_action = Game.WAIT 
                                        else: chosen_player_action = Game.WAIT 
                                    else: chosen_player_action = Game.WAIT 
                        else: 
                            actions_to_target = []
                            for action_key, (dr, dc) in Game.ACTION_DELTAS.items():
                                next_r, next_c = player_r + dr, player_c + dc
                                if game._is_walkable(next_r, next_c, "PLAYER"):
                                    dist_to_target = math.sqrt((next_r - global_target_exit_pos[0])**2 + (next_c - global_target_exit_pos[1])**2)
                                    actions_to_target.append({'action': action_key, 'dist': dist_to_target})
                            if actions_to_target:
                                actions_to_target.sort(key=lambda x: x['dist'])
                                chosen_player_action = actions_to_target[0]['action']
                            else: chosen_player_action = Game.WAIT 
                    # --- EXPLORATION LOGIC ---
                    else: 
                        current_room_obj = game.get_room_player_is_in(player_r, player_c)
                        if current_player_tile_type == HotelGenerator.ROOM_FLOOR and current_room_obj:
                            room_door = game.get_door_for_room(current_room_obj)
                            if room_door:
                                target_door_pos_on_grid = room_door['door_pos']
                                actions_to_door = []
                                for action_key, (dr, dc) in Game.ACTION_DELTAS.items():
                                    next_r, next_c = player_r + dr, player_c + dc
                                    if game.is_pos_in_room_or_door(next_r, next_c, current_room_obj, target_door_pos_on_grid) and \
                                       game._is_walkable(next_r, next_c, "PLAYER"):
                                        dist_to_door = math.sqrt((next_r - target_door_pos_on_grid[0])**2 + (next_c - target_door_pos_on_grid[1])**2)
                                        actions_to_door.append({'action': action_key, 'dist': dist_to_door})
                                if actions_to_door: 
                                    actions_to_door.sort(key=lambda x: x['dist'])
                                    min_d = actions_to_door[0]['dist']
                                    best_options = [opt for opt in actions_to_door if opt['dist'] == min_d]
                                    non_wait_best_options = [opt for opt in best_options if opt['action'] != Game.WAIT]
                                    if non_wait_best_options:
                                        chosen_player_action = random.choice(non_wait_best_options)['action']
                                    elif best_options: 
                                        chosen_player_action = best_options[0]['action'] 
                        elif current_player_tile_type == HotelGenerator.HALLWAY or \
                             current_player_tile_type == HotelGenerator.DOOR:
                            action_chosen_for_hallway = False
                            if last_player_direction and last_player_direction != Game.WAIT:
                                dr_last, dc_last = Game.ACTION_DELTAS[last_player_direction]
                                next_r_last, next_c_last = player_r + dr_last, player_c + dc_last
                                if 0 <= next_r_last < GRID_HEIGHT and 0 <= next_c_last < GRID_WIDTH and \
                                   game._is_walkable(next_r_last, next_c_last, "PLAYER"):
                                    next_tile_type_last = game._get_tile(next_r_last, next_c_last)
                                    if next_tile_type_last == HotelGenerator.HALLWAY or next_tile_type_last == HotelGenerator.DOOR:
                                        chosen_player_action = last_player_direction
                                        action_chosen_for_hallway = True
                            
                            if not action_chosen_for_hallway: 
                                possible_new_directions = []
                                action_options = [Game.MOVE_NORTH, Game.MOVE_SOUTH, Game.MOVE_EAST, Game.MOVE_WEST]
                                random.shuffle(action_options) 
                                for action_key_hallway in action_options:
                                    dr_new, dc_new = Game.ACTION_DELTAS[action_key_hallway]
                                    next_r_new, next_c_new = player_r + dr_new, player_c + dc_new
                                    if 0 <= next_r_new < GRID_HEIGHT and 0 <= next_c_new < GRID_WIDTH and \
                                       game._is_walkable(next_r_new, next_c_new, "PLAYER"):
                                        next_tile_type_new = game._get_tile(next_r_new, next_c_new)
                                        if next_tile_type_new == HotelGenerator.HALLWAY or next_tile_type_new == HotelGenerator.DOOR:
                                            possible_new_directions.append(action_key_hallway)
                                if possible_new_directions:
                                    chosen_player_action = random.choice(possible_new_directions)
                        else: 
                            valid_fallback_moves = []
                            action_options_fallback = [Game.MOVE_NORTH, Game.MOVE_SOUTH, Game.MOVE_EAST, Game.MOVE_WEST]
                            random.shuffle(action_options_fallback)
                            for p_act_key_fallback in action_options_fallback:
                                next_r_fb, next_c_fb = player_r + Game.ACTION_DELTAS[p_act_key_fallback][0], player_c + Game.ACTION_DELTAS[p_act_key_fallback][1]
                                if game._is_walkable(next_r_fb, next_c_fb, "PLAYER"):
                                    valid_fallback_moves.append(p_act_key_fallback)
                            if valid_fallback_moves:
                                chosen_player_action = random.choice(valid_fallback_moves)
                    
                    do_log(f"--- Owner Turn {game_turn_count + 1}, Player Action {player_action_count_in_turn}/{game.player_moves_per_turn} ---\n")
                    do_log(f"Player at ({player_r},{player_c}), Tile: {current_player_tile_type}, Knows Exit: {player_knows_exit_pos}, Owner Seen: {owner_seen_at}\n")
                    do_log(f"Player intends to: {chosen_player_action}\n")
                    
                    player_pos_before_action = game.player_pos 
                    game.handle_player_turn(chosen_player_action) 

                    if game.player_pos is not None and player_pos_before_action is not None:
                        player_pos_history.appendleft(player_pos_before_action)
                        if game.player_pos != player_pos_before_action and chosen_player_action != Game.WAIT:
                            last_player_direction = chosen_player_action
                        elif game.player_pos == player_pos_before_action and chosen_player_action != Game.WAIT: 
                            last_player_direction = None 
                    elif game.player_pos is None: 
                        last_player_direction = None
                    
                    do_log(game.render_grid_to_string(player_pov=False) + "\n")
                    log_player_pos_after_move = game.player_pos
                    log_player_tile_after_move = 'N/A'
                    if log_player_pos_after_move: 
                        log_player_tile_after_move = game._get_tile(log_player_pos_after_move[0], log_player_pos_after_move[1])
                    do_log(f"Player actual position after move: {log_player_pos_after_move}, Tile: {log_player_tile_after_move}\n\n")

                    current_owner_state_tensor = get_dqn_state_representation(game, GRID_HEIGHT, GRID_WIDTH, precomputed_static_map_np, owner_pos_history, player_pos_history)

                elif current_entity == "OWNER":
                    do_log(f"--- Owner Turn {game_turn_count + 1} (Owner's Move) ---\n")
                    
                    # PPO choose_action expects state (1, C, H, W). current_owner_state_tensor is (C,H,W)
                    chosen_owner_action, action_log_prob, state_value = owner_agent.choose_action(current_owner_state_tensor.unsqueeze(0))
                    prev_owner_state_tensor_for_ppo = current_owner_state_tensor 
                    owner_pos_before_action = game.owner_pos
                    
                    moved_successfully = game.handle_owner_turn(chosen_owner_action)
                    
                    owner_pos_history.appendleft(owner_pos_before_action)
                    total_env_steps += 1 
                    
                    do_log(game.render_grid_to_string(player_pov=False) + "\n")
                    log_owner_pos_after_move = game.owner_pos
                    log_owner_tile_after_move = 'N/A'
                    if log_owner_pos_after_move: 
                        log_owner_tile_after_move = game._get_tile(log_owner_pos_after_move[0], log_owner_pos_after_move[1])
                    do_log(f"Owner intends to: {chosen_owner_action}\n")
                    do_log(f"Owner actual position after move: {log_owner_pos_after_move}, Tile: {log_owner_tile_after_move}\n\n")
                    
                    next_owner_state_tensor = get_dqn_state_representation(game, GRID_HEIGHT, GRID_WIDTH, precomputed_static_map_np, owner_pos_history, player_pos_history)
                    
                    step_reward = effective_reward_base

                    if game.owner_pos:
                        owner_r_after_move, owner_c_after_move = game.owner_pos
                        owner_tile_type_after_move = game._get_tile(owner_r_after_move, owner_c_after_move)
                        if owner_tile_type_after_move == HotelGenerator.ROOM_FLOOR:
                            step_reward += REWARD_IN_ROOM

                        if is_phase_4 and not first_door_reward_given and owner_tile_type_after_move == HotelGenerator.DOOR:
                            step_reward += REWARD_FIRST_DOOR
                            first_door_reward_given = True 
                            do_log(f"Owner stepped on a door tile. Adding REWARD_FIRST_DOOR: {REWARD_FIRST_DOOR}\n")
                    
                    if not moved_successfully and chosen_owner_action != Game.WAIT: 
                        step_reward += REWARD_BUMP_WALL
                        episode_wall_bumps += 1
                    else:
                        step_reward += effective_reward_per_step
                    
                    if game.player_pos and game.owner_pos:
                        owner_vision_data = game.get_owner_vision_data()
                        if game.player_pos in owner_vision_data and "PLAYER" in owner_vision_data[game.player_pos]:
                            step_reward += REWARD_PLAYER_VISIBLE
                            do_log(f"Player visible to owner. Adding REWARD_PLAYER_VISIBLE: {REWARD_PLAYER_VISIBLE}\n")
                        
                        dist_to_player = math.sqrt((game.player_pos[0] - game.owner_pos[0])**2 + (game.player_pos[1] - game.owner_pos[1])**2)
                        if dist_to_player <= PROXIMITY_THRESHOLD:
                            normalized_distance = dist_to_player / PROXIMITY_THRESHOLD
                            proximity_reward_value = effective_reward_proximity * (1 - normalized_distance)
                            step_reward += proximity_reward_value
                            if effective_reward_proximity != 0:
                                do_log(f"Owner close to player. Adding proximity reward: {proximity_reward_value:.2f}\n")
                    
                    is_done_for_ppo = game.game_over 

                    if game.game_over: 
                        if game.winner == "OWNER":
                            step_reward += effective_reward_catch_player
                            do_log(f"OWNER WINS! Adding effective_reward_catch_player: {effective_reward_catch_player}\n")
                        elif game.winner == "PLAYER": 
                            step_reward += effective_reward_player_escapes
                            do_log(f"PLAYER ESCAPES (during owner's turn processing)! Adding effective_reward_player_escapes: {effective_reward_player_escapes}\n")

                    elif (game_turn_count + 1) >= MAX_TURNS_PER_EPISODE: 
                        is_done_for_ppo = True 
                        step_reward += effective_reward_player_escapes 
                        game.game_over = True 
                        game.winner = "PLAYER" 
                        do_log(f"Max turns reached for owner. Assigning effective_reward_player_escapes: {effective_reward_player_escapes}.\n")
                    
                    do_log(f"Owner received reward for this turn: {step_reward}\n")
                    episode_reward += step_reward
                    
                    actual_next_state_for_ppo = next_owner_state_tensor if not is_done_for_ppo else None
                    owner_agent.store_transition(prev_owner_state_tensor_for_ppo, 
                                                 chosen_owner_action, 
                                                 action_log_prob,
                                                 state_value.squeeze(), # Ensure state_value is scalar or (1,)
                                                 step_reward, 
                                                 actual_next_state_for_ppo,
                                                 is_done_for_ppo) 
                    
                    current_owner_state_tensor = next_owner_state_tensor 
                    game_turn_count += 1 

                    # PPO Learning Step
                    if total_env_steps % UPDATE_TIMESTEPS == 0 and total_env_steps > 0:
                        if len(owner_agent.memory["states"]) >= owner_agent.mini_batch_size: # Ensure enough samples for at least one batch
                            do_log(f"--- PPO Learning Update at step {total_env_steps} with {len(owner_agent.memory['states'])} transitions ---\n")
                            p_loss, v_loss, e_loss = owner_agent.learn() # learn() clears memory
                            avg_policy_loss_log.append(p_loss)
                            avg_value_loss_log.append(v_loss)
                            avg_entropy_loss_log.append(e_loss)
                            do_log(f"PPO Update Complete. Policy Loss: {p_loss:.4f}, Value Loss: {v_loss:.4f}, Entropy: {e_loss:.4f}\n")
                        else:
                            do_log(f"Skipping PPO Learning Update at step {total_env_steps}: not enough samples ({len(owner_agent.memory['states'])} < {owner_agent.mini_batch_size})\n")
                
                if game.game_over: # Check if game ended after player or owner turn
                    do_log(f"--- GAME OVER (Episode {episode + 1}) ---\n")
                    do_log(f"Winner: {game.winner}\n")
                    do_log(game.render_grid_to_string(player_pov=False) + "\n\n")
                    # This break is for the inner for-loop managing turns
                    break 
            
            # End of episode handling (after for loop or game_over break)
            if not game.game_over and game_turn_count >= MAX_TURNS_PER_EPISODE:
                do_log(f"--- MAX TURNS ({MAX_TURNS_PER_EPISODE}) REACHED (Episode {episode + 1}) ---\n")
                do_log(f"Game outcome treated as Player Escape for reward purposes if not already game_over.\n")
                # If game didn't end by catch, and max turns hit, it's like player escaped from owner's perspective
                # This reward is already handled if is_done_for_ppo was set due to max turns.
                # This block is mostly for logging consistency.
                if game.winner is None: game.winner = "PLAYER" # Ensure winner is set

            # Final reward adjustments or logging for the episode
            # Note: REWARD_PLAYER_ESCAPES is added to step_reward when game ends by timeout or player escape.
            # No need for the specific DQN-style adjustment:
            # if game.game_over and game.winner == "PLAYER" and game_turn_count < MAX_TURNS_PER_EPISODE:
            #     episode_reward += effective_reward_player_escapes

            do_log(f"Episode {episode + 1} Reward: {episode_reward}\n")
            do_log(f"Episode {episode + 1} Wall Bumps: {episode_wall_bumps}\n")
            if log_file_handle: 
                log_file_handle.close()
                log_file_handle = None 

            # PPO does not use epsilon decay in this manner
            # owner_agent.decay_exploration() 
            total_rewards_per_episode.append(episode_reward)

            if (episode + 1) % LEARNING_RATE_DECAY_FREQUENCY == 0 and current_learning_rate > MIN_LEARNING_RATE:
                current_learning_rate *= LEARNING_RATE_DECAY_FACTOR
                current_learning_rate = max(current_learning_rate, MIN_LEARNING_RATE) 
                for param_group in owner_agent.optimizer.param_groups:
                    param_group['lr'] = current_learning_rate
                do_log(f"Decayed learning rate to {current_learning_rate} at episode {episode + 1}\n")


            if len(total_rewards_per_episode) >= MA_WINDOW:
                current_ma = np.mean(total_rewards_per_episode[-MA_WINDOW:])
                moving_avg_rewards_per_episode.append(current_ma)
            else:
                moving_avg_rewards_per_episode.append(np.nan)

            postfix_stats = {
                "Phase": "1" if is_phase_1 else "2" if not is_phase_3 else "3",
                "LR": f"{current_learning_rate:.6f}",
                "Steps": total_env_steps,
            }
            if avg_policy_loss_log: # Add recent losses if available
                postfix_stats["P_Loss"] = f"{avg_policy_loss_log[-1]:.3f}"
                postfix_stats["V_Loss"] = f"{avg_value_loss_log[-1]:.3f}"

            if len(total_rewards_per_episode) >= MA_WINDOW:
                avg_reward = moving_avg_rewards_per_episode[-1] # Use the already calculated MA
                if not np.isnan(avg_reward):
                    postfix_stats[f"Avg Rwd ({MA_WINDOW}-Ep)"] = f"{avg_reward:.2f}"
            elif total_rewards_per_episode:
                avg_reward = np.mean(total_rewards_per_episode)
                postfix_stats[f"Avg Rwd (All)"] = f"{avg_reward:.2f}"
            episode_pbar.set_postfix(postfix_stats)

            if (episode + 1) % PLOTTING_FREQUENCY == 0:
                periodic_model_save_path = os.path.join(MODEL_DIR, f"ppo_owner_agent_ep{episode+1}.pth")
                owner_agent.save_model(periodic_model_save_path)
                
                fig, ax1 = plt.subplots(figsize=(12, 7))
                
                # Plot Rewards
                color = 'tab:blue'
                ax1.set_xlabel("Episode")
                ax1.set_ylabel("Total Reward for Owner", color=color)
                ax1.plot(total_rewards_per_episode, label='Episode Reward', color=color, alpha=0.6)
                ax1.plot(moving_avg_rewards_per_episode, label=f'{MA_WINDOW}-Ep Moving Average', linestyle='-', color='darkblue', linewidth=2)
                ax1.tick_params(axis='y', labelcolor=color)
                ax1.grid(True, axis='y', linestyle=':', alpha=0.7)

                if 1 < PHASE_2_START_EPISODE <= (episode + 1):
                    ax1.axvline(x=PHASE_2_START_EPISODE - 1, color='g', linestyle='--', label=f'Phase 2 Start (Ep {PHASE_2_START_EPISODE})')
                if 1 < PHASE_3_START_EPISODE <= (episode + 1):
                    ax1.axvline(x=PHASE_3_START_EPISODE - 1, color='orange', linestyle='--', label=f'Phase 3 Start (Ep {PHASE_3_START_EPISODE})')
                
                # Plot PPO Losses on a secondary y-axis
                if avg_policy_loss_log: # Check if there's loss data to plot
                    ax2 = ax1.twinx() # instantiate a second axes that shares the same x-axis
                    color_p = 'tab:red'
                    color_v = 'tab:green'
                    # We need to align loss plots with episodes where learning happened.
                    # Learning happens every UPDATE_TIMESTEPS, not every episode.
                    # For simplicity, we plot the sequence of losses.
                    # A more accurate plot would map losses to the episode number *after* which the learning step occurred.
                    # However, for a general trend, plotting the list of losses is indicative.
                    loss_indices = np.linspace(0, episode + 1, len(avg_policy_loss_log)) # Approximate mapping to episodes

                    ax2.set_ylabel('Avg Losses (PPO)', color=color_p) 
                    ax2.plot(loss_indices, avg_policy_loss_log, label='Avg Policy Loss', color=color_p, linestyle=':', alpha=0.7)
                    ax2.plot(loss_indices, avg_value_loss_log, label='Avg Value Loss', color=color_v, linestyle=':', alpha=0.7)
                    # ax2.plot(loss_indices, avg_entropy_loss_log, label='Avg Entropy', color='tab:purple', linestyle=':', alpha=0.7) # Optional
                    ax2.tick_params(axis='y', labelcolor=color_p)
                
                fig.tight_layout() # otherwise the right y-label is slightly clipped
                plt.title(f"PPO Owner's Rewards & Losses up to Episode {episode+1}")
                
                # Combine legends
                lines, labels = ax1.get_legend_handles_labels()
                if avg_policy_loss_log:
                    lines2, labels2 = ax2.get_legend_handles_labels()
                    ax2.legend(lines + lines2, labels + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)
                else:
                    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)

                periodic_plot_filename = os.path.join(PLOTS_DIR, f"ppo_rewards_losses_ep{episode+1}.png")
                plt.savefig(periodic_plot_filename, bbox_inches='tight')
                plt.close() 

    print("\n--- PPO TRAINING COMPLETE ---")
    final_model_save_path = os.path.join(MODEL_DIR, "ppo_owner_agent_final.pth")
    owner_agent.save_model(final_model_save_path)
    print(f"Final PPO model saved as {final_model_save_path}")

    # Final Plot
    fig, ax1 = plt.subplots(figsize=(12, 7))
    color = 'tab:blue'
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Total Reward for Owner", color=color)
    ax1.plot(total_rewards_per_episode, label='Episode Reward', color=color, alpha=0.6)
    ax1.plot(moving_avg_rewards_per_episode, label=f'{MA_WINDOW}-Ep Moving Average', linestyle='-', color='darkblue', linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, axis='y', linestyle=':', alpha=0.7)
    if 1 < PHASE_2_START_EPISODE <= NUM_EPISODES:
        ax1.axvline(x=PHASE_2_START_EPISODE - 1, color='g', linestyle='--', label=f'Phase 2 Start (Ep {PHASE_2_START_EPISODE})')
    if 1 < PHASE_3_START_EPISODE <= NUM_EPISODES:
        ax1.axvline(x=PHASE_3_START_EPISODE - 1, color='orange', linestyle='--', label=f'Phase 3 Start (Ep {PHASE_3_START_EPISODE})')

    if avg_policy_loss_log:
        ax2 = ax1.twinx()
        color_p = 'tab:red'
        color_v = 'tab:green'
        loss_indices = np.linspace(0, NUM_EPISODES, len(avg_policy_loss_log))
        ax2.set_ylabel('Avg Losses (PPO)', color=color_p)
        ax2.plot(loss_indices, avg_policy_loss_log, label='Avg Policy Loss', color=color_p, linestyle=':', alpha=0.7)
        ax2.plot(loss_indices, avg_value_loss_log, label='Avg Value Loss', color=color_v, linestyle=':', alpha=0.7)
        ax2.tick_params(axis='y', labelcolor=color_p)
    
    fig.tight_layout()
    plt.title("PPO Owner's Rewards & Losses per Episode (Full Training)")
    lines, labels = ax1.get_legend_handles_labels()
    if avg_policy_loss_log:
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)
    else:
        ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)
    
    final_plot_filename = os.path.join(PLOTS_DIR, "ppo_final_rewards_losses_plot.png")
    plt.savefig(final_plot_filename, bbox_inches='tight')
    print(f"Final PPO rewards & losses plot saved as {final_plot_filename}")
    plt.close()
    print(f"Periodic episode traces logged to '{LOGS_DIR}' directory.")

if __name__ == '__main__':
    train_agent()
