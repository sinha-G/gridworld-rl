import os
import random
import math
import numpy as np
import torch
import matplotlib.pyplot as plt

from tqdm import tqdm 
from collections import deque

from game import Game 
from agents.dqn_agent import DQNAgent 
from gridworld_generator import HotelGenerator 

# --- STATE REPRESENTATION FOR DQN ---
NUM_DQN_CHANNELS = 9 
SOUND_PERCEPTION_RADIUS = 8

def get_dqn_state_representation(game_instance, grid_height, grid_width):
    """
    Creates a multi-channel tensor representation of the game state for the DQN.
    Output shape: (NUM_DQN_CHANNELS, grid_height, grid_width)
    """
    state_tensor_np = np.zeros((NUM_DQN_CHANNELS, grid_height, grid_width), dtype=np.float32)

    # Channel 0: Owner's Position
    if game_instance.owner_pos:
        orow, ocol = game_instance.owner_pos
        if 0 <= orow < grid_height and 0 <= ocol < grid_width:
            state_tensor_np[0, orow, ocol] = 1.0

    # Channel 1: Player's Position (Visible to Owner)
    owner_vision = game_instance.get_owner_vision_data() 
    if game_instance.player_pos:
        prow, pcol = game_instance.player_pos
        if (prow, pcol) in owner_vision and owner_vision[(prow, pcol)] == "PLAYER":
             if 0 <= prow < grid_height and 0 <= pcol < grid_width:
                state_tensor_np[1, prow, pcol] = 1.0
    
    # Iterate through the grid for static elements
    for r_idx in range(grid_height): # Renamed r to r_idx to avoid conflict
        for c_idx in range(grid_width): # Renamed c to c_idx
            tile = game_instance.grid[r_idx][c_idx]
            # Channel 2: Walls
            if tile == HotelGenerator.WALL:
                state_tensor_np[2, r_idx, c_idx] = 1.0
            # Channel 3: Doors
            elif tile == HotelGenerator.DOOR:
                state_tensor_np[3, r_idx, c_idx] = 1.0
            # Channel 4: Hallways
            elif tile == HotelGenerator.HALLWAY:
                state_tensor_np[4, r_idx, c_idx] = 1.0
            # Channel 5: Room Floors
            elif tile == HotelGenerator.ROOM_FLOOR:
                state_tensor_np[5, r_idx, c_idx] = 1.0
            # Channel 6: Hiding Spots
            elif tile == HotelGenerator.HIDING_SPOT:
                state_tensor_np[6, r_idx, c_idx] = 1.0
            # Channel 7: Exit Position
            if game_instance.exit_pos and r_idx == game_instance.exit_pos[0] and c_idx == game_instance.exit_pos[1]:
                 state_tensor_np[7, r_idx, c_idx] = 1.0

    # Channel 8: Sounds (from game_instance.sound_alerts)
    for sound_event in game_instance.sound_alerts:
        if 'DOOR' in sound_event['type'].upper(): 
            srow, scol = sound_event['pos']
            orow, ocol = game_instance.owner_pos
            if (srow - orow)**2 + (scol - ocol)**2 <= SOUND_PERCEPTION_RADIUS ** 2 and 0 <= srow < grid_height and 0 <= scol < grid_width:
                state_tensor_np[8, srow, scol] = 1.0
                 
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
    LOGGING_FREQUENCY = 20
    PLOTTING_FREQUENCY = 500
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    # --- END DIRECTORY SETUP ---

    # Game parameters
    GRID_HEIGHT = 25
    GRID_WIDTH = 30
    PLAYER_MOVES_PER_TURN = 2

    game_params = {
        'straightness_hallways': 0.9,
        'hall_loops': 5,
        'max_hallway_perc': 0.25,
        'max_rooms': 25,
        'room_min_size': 2,
        'room_max_size': 4,
        'max_hiding_spots_per_room': 1,
    }

    # DQN Agent parameters
    NUM_EPISODES = 10000
    MAX_TURNS_PER_EPISODE = 200
    LEARNING_RATE = 0.0005
    DISCOUNT_FACTOR = 0.99
    EXPLORATION_RATE_INITIAL = 1.0
    EXPLORATION_DECAY_RATE = 0.9999
    MIN_EXPLORATION_RATE = 0.05
    REPLAY_BUFFER_SIZE = 50000
    BATCH_SIZE = 256
    TARGET_UPDATE_FREQUENCY = 1000
    LEARN_START_STEPS = 2500
    LEARN_EVERY_N_STEPS = 4
    PROXIMITY_THRESHOLD = 7

    # Rewards
    REWARD_CATCH_PLAYER = 100
    REWARD_PLAYER_ESCAPES = -100
    REWARD_BUMP_WALL = -10
    REWARD_IN_ROOM = -5
    REWARD_PROXIMITY_MAX = 50 
    # REWARD_PROXIMITY_STEALTH_BONUS = 10

    game_actions_list = [Game.MOVE_NORTH, Game.MOVE_SOUTH, Game.MOVE_EAST, Game.MOVE_WEST, Game.WAIT]

    owner_agent = DQNAgent(
        game_actions_list=game_actions_list,
        input_channels=NUM_DQN_CHANNELS,
        grid_height=GRID_HEIGHT,
        grid_width=GRID_WIDTH,
        learning_rate=LEARNING_RATE,
        discount_factor=DISCOUNT_FACTOR,
        exploration_rate_initial=EXPLORATION_RATE_INITIAL,
        exploration_decay_rate=EXPLORATION_DECAY_RATE,
        min_exploration_rate=MIN_EXPLORATION_RATE,
        replay_buffer_size=REPLAY_BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        target_update_frequency=TARGET_UPDATE_FREQUENCY
    )

    model_load_path = os.path.join(MODEL_DIR, "dqn_owner_agent.pth")
    # owner_agent.load_model(model_load_path) # Comment out if starting fresh or if model is causing issues

    total_rewards_per_episode = []
    total_steps_taken = 0
    print("Starting DQN training...")

    with tqdm(range(NUM_EPISODES), unit="episode") as episode_pbar:
        for episode in episode_pbar:
            game = Game(width=GRID_WIDTH, height=GRID_HEIGHT, player_moves_per_turn=PLAYER_MOVES_PER_TURN, generator_params=game_params)
            player_action_queue = deque() # Queue for player actions
            log_file_handle = None
            current_log_filename = None # To store the filename if logging occurs
            
            if (episode + 1) % LOGGING_FREQUENCY == 0:
                current_log_filename = os.path.join(LOGS_DIR, f"episode_trace_ep{episode+1}.txt")
                log_file_handle = open(current_log_filename, "w")

            def do_log(message):
                if log_file_handle:
                    log_file_handle.write(message)

            do_log(f"--- Episode {episode + 1} Initial State ---\n")
            do_log(game.render_grid_to_string(player_pov=False) + "\n\n")
            # game.print_grid_with_entities(player_pov=False) # Original console print

            player_knows_exit_pos = None
            current_owner_state_tensor = get_dqn_state_representation(game, GRID_HEIGHT, GRID_WIDTH)
            episode_reward = 0
            game_turn_count = 0 
            player_recent_hallway_pos = [] 

            while not game.game_over and game_turn_count < MAX_TURNS_PER_EPISODE:
                current_entity = game.get_current_turn_entity()
                player_action_count_in_turn = game.player_moves_taken_this_turn + 1

                if current_entity == "PLAYER":
                    chosen_player_action = Game.WAIT 

                    player_r, player_c = game.player_pos
                    current_player_tile_type = game._get_tile(player_r, player_c)
                    
                    player_vision, _ = game.get_player_vision_data()
                    owner_seen_at = None
                    for loc, entity_type_in_vision in player_vision.items():
                        if entity_type_in_vision == "OWNER":
                            owner_seen_at = loc
                            break
                    
                    if game.exit_pos and game.exit_pos in player_vision and player_vision[game.exit_pos] == HotelGenerator.EXIT:
                        player_knows_exit_pos = game.exit_pos 

                    # --- FLEEING LOGIC ---
                    if owner_seen_at:
                        # if episode < 5 and game.player_moves_taken_this_turn == 0: print(f"Ep {episode}: Player at ({player_r},{player_c}) FLEEING owner at {owner_seen_at}")
                        player_action_queue.clear() # Clear any planned path
                        flee_actions_with_dist = []
                        for action_key, (dr, dc) in Game.ACTION_DELTAS.items():
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
                        # Validate the queued action
                        q_dr, q_dc = Game.ACTION_DELTAS[action_from_queue]
                        q_next_r, q_next_c = player_r + q_dr, player_c + q_dc

                        if game._is_walkable(q_next_r, q_next_c, "PLAYER"):
                            # Further check: if pathing to known exit, ensure move is consistent
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
                                player_action_queue.clear() # chosen_player_action remains Game.WAIT
                        else:
                            do_log(f"Queued action '{action_from_queue}' is no longer walkable. Clearing queue.\n")
                            player_action_queue.clear() # chosen_player_action remains Game.WAIT

                    # --- EXIT LOGIC ---
                    elif player_knows_exit_pos is not None and game.exit_pos is not None:
                        global_target_exit_pos = player_knows_exit_pos
                        current_room_obj = game.get_room_player_is_in(player_r, player_c)
                        
                        # 1. If in a room, try to move towards the door to exit to a hallway.
                        if current_room_obj and current_player_tile_type == HotelGenerator.ROOM_FLOOR:
                            room_door = game.get_door_for_room(current_room_obj)
                            
                            if room_door:
                                target_door_pos_on_grid = room_door['door_pos']
                                # if episode < 5 and game.player_moves_taken_this_turn == 0: print(f"Ep {episode}: Best door is {target_door_pos_on_grid}, connected_hallway {best_door_to_use['connected_hallway']}")
                                
                                actions_to_door = []
                                for action_key, (dr, dc) in Game.ACTION_DELTAS.items():
                                    next_r, next_c = player_r + dr, player_c + dc
                                    # Check if the move is within the current room or to the target door, and is walkable
                                    if game.is_pos_in_room_or_door(next_r, next_c, current_room_obj, target_door_pos_on_grid) and \
                                       game._is_walkable(next_r, next_c, "PLAYER"):
                                        dist_to_door = math.sqrt((next_r - target_door_pos_on_grid[0])**2 + (next_c - target_door_pos_on_grid[1])**2)
                                        actions_to_door.append({'action': action_key, 'dist': dist_to_door, 'next_pos':(next_r,next_c)})
                                
                                actions_to_door.sort(key=lambda x: x['dist'])
                                min_d = actions_to_door[0]['dist']
                                # Prefer moves that actually change position if multiple options have same min distance
                                best_options = [opt for opt in actions_to_door if opt['dist'] == min_d]
                                # To avoid getting stuck, prefer non-WAIT if possible
                                non_wait_best_options = [opt for opt in best_options if opt['action'] != Game.WAIT]
                                if non_wait_best_options:
                                    chosen_player_action = random.choice(non_wait_best_options)['action']
                                elif best_options: # Only WAIT or no non-WAIT options at min_dist
                                    chosen_player_action = best_options[0]['action'] # Could be WAIT
                            # No door found, fallback to exploring the room
                            else:
                                # if episode < 5 and game.player_moves_taken_this_turn == 0: print(f"Ep {episode}: Could not find a best door. Exploring in room.")
                                valid_room_moves = [] 
                                for p_act_key, (dr_p, dc_p) in Game.ACTION_DELTAS.items():
                                    next_r_p, next_c_p = player_r + dr_p, player_c + dc_p
                                    if game.is_pos_in_room_or_door(next_r_p, next_c_p, current_room_obj, None) and \
                                       game._is_walkable(next_r_p, next_c_p, "PLAYER"): # Check if move stays in room and is walkable
                                        valid_room_moves.append(p_act_key)
                                if Game.WAIT not in valid_room_moves: valid_room_moves.append(Game.WAIT) # Ensure WAIT is an option
                                if valid_room_moves: chosen_player_action = random.choice(valid_room_moves)

                        # 2. If in a hallway or at a door, explore the hallway towards the exit.
                        elif current_player_tile_type == HotelGenerator.HALLWAY or \
                             current_player_tile_type == HotelGenerator.DOOR or \
                             (player_r, player_c) == global_target_exit_pos:
                            
                            if (player_r, player_c) == global_target_exit_pos:
                                # if episode < 5 and game.player_moves_taken_this_turn == 0: print(f"Ep {episode}: Player AT EXIT ({player_r},{player_c}). Waiting for game to end.")
                                chosen_player_action = Game.WAIT 
                            else:                                
                                path_to_exit = bfs_hallway_pathfinding(game, (player_r, player_c), global_target_exit_pos)
                                if path_to_exit and len(path_to_exit) > 0:
                                    player_action_queue.extend(path_to_exit)
                                    do_log(f"BFS success. Path: {path_to_exit}. Queue populated with {len(player_action_queue)} actions.\n")
                                    if player_action_queue: # Should be true
                                        chosen_player_action = player_action_queue.popleft()
                                        do_log(f"Player taking first action '{chosen_player_action}' from new BFS path. {len(player_action_queue)} actions remaining.\n")
                                else: # BFS FAILED or empty path
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
                        
                        # 3. If not in a room or hallway, explore the grid using any valid moves.
                        else: 
                            # if episode < 5 and game.player_moves_taken_this_turn == 0: print(f"Ep {episode}: Player at ({player_r},{player_c}) knows exit {global_target_exit_pos}, but not in Room/Hallway/Door (e.g. Hiding Spot). General approach.")
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
                        # 1. If in a room, try to move towards a random door to exit to a hallway.
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
                        
                        # 2. If in a hallway or at a door, explore the hallway using persistent direction.
                        elif current_player_tile_type == HotelGenerator.HALLWAY or \
                             current_player_tile_type == HotelGenerator.DOOR:
                            action_chosen_for_hallway = False
                            # Try to continue in the last_player_direction
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
                                # If cannot continue, or no last_player_direction, pick a new random valid direction (not WAIT)
                                possible_new_directions = []
                                action_options = [Game.MOVE_NORTH, Game.MOVE_SOUTH, Game.MOVE_EAST, Game.MOVE_WEST]
                                random.shuffle(action_options) # Shuffle to pick a random new direction

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
                        
                        # 3. Fallback: If exploring but not in a room or hallway (e.g., hiding spot, other unexpected tile)
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
                    
                    # --- End of Player Action Decision ---
                    
                    do_log(f"--- Owner Turn {game_turn_count + 1}, Player Action {player_action_count_in_turn}/{game.player_moves_per_turn} ---\n")
                    do_log(f"Player at ({player_r},{player_c}), Tile: {current_player_tile_type}, Knows Exit: {player_knows_exit_pos}, Owner Seen: {owner_seen_at}\n")
                    do_log(f"Player intends to: {chosen_player_action}\n")
                    
                    # Store player's position before the action to check if they actually moved
                    player_pos_before_action = game.player_pos 
                    
                    game.handle_player_turn(chosen_player_action) # Player takes action
                    
                    # Update last_player_direction based on the outcome of the action
                    if game.player_pos is not None and player_pos_before_action is not None:
                        if game.player_pos != player_pos_before_action and chosen_player_action != Game.WAIT:
                            # Player successfully moved to a new tile
                            last_player_direction = chosen_player_action
                        elif game.player_pos == player_pos_before_action and chosen_player_action != Game.WAIT:
                            # Player intended to move but didn't (e.g., bumped a wall, invalid move)
                            # Reset last_player_direction so it tries a new one next time if exploring hallways.
                            last_player_direction = None
                        # If chosen_player_action was Game.WAIT, last_player_direction remains unchanged.
                    elif game.player_pos is None: # Should not happen if game is ongoing
                        last_player_direction = None
                    
                    do_log(game.render_grid_to_string(player_pov=False) + "\n")
                    log_player_pos_after_move = game.player_pos
                    log_player_tile_after_move = 'N/A'
                    if log_player_pos_after_move: # Check if player_pos is not None
                        log_player_tile_after_move = game._get_tile(log_player_pos_after_move[0], log_player_pos_after_move[1])
                    do_log(f"Player actual position after move: {log_player_pos_after_move}, Tile: {log_player_tile_after_move}\n\n")

                    current_player_tile_after_move = game._get_tile(game.player_pos[0], game.player_pos[1]) if game.player_pos else None
                    current_owner_state_tensor = get_dqn_state_representation(game, GRID_HEIGHT, GRID_WIDTH) 

                elif current_entity == "OWNER":
                    do_log(f"--- Owner Turn {game_turn_count + 1} (Owner's Move) ---\n")
                    valid_owner_actions = []
                    for owner_act_key_iter in game_actions_list: 
                        if owner_act_key_iter == Game.WAIT:
                            valid_owner_actions.append(owner_act_key_iter)
                            continue
                        dr_o, dc_o = Game.ACTION_DELTAS[owner_act_key_iter]
                        if game.owner_pos: 
                            next_r_o, next_c_o = game.owner_pos[0] + dr_o, game.owner_pos[1] + dc_o
                            if game._is_walkable(next_r_o, next_c_o, "OWNER"):
                                valid_owner_actions.append(owner_act_key_iter)
                    if not valid_owner_actions and game.owner_pos : valid_owner_actions.append(Game.WAIT)
                    elif not game.owner_pos: 
                        valid_owner_actions.append(Game.WAIT)

                    chosen_owner_action = owner_agent.choose_action(current_owner_state_tensor.unsqueeze(0), game, valid_owner_actions)
                    do_log(f"Owner at {game.owner_pos} intends to: {chosen_owner_action}\n")

                    prev_owner_state_tensor_for_replay = current_owner_state_tensor 
                    
                    moved_successfully = game.handle_owner_turn(chosen_owner_action)
                    
                    do_log(game.render_grid_to_string(player_pov=False) + "\n")
                    log_owner_pos_after_move = game.owner_pos
                    log_owner_tile_after_move = 'N/A'
                    if log_owner_pos_after_move: # Check if owner_pos is not None
                        log_owner_tile_after_move = game._get_tile(log_owner_pos_after_move[0], log_owner_pos_after_move[1])
                    do_log(f"Owner actual position after move: {log_owner_pos_after_move}, Tile: {log_owner_tile_after_move}\n\n")
                    
                    next_owner_state_tensor = get_dqn_state_representation(game, GRID_HEIGHT, GRID_WIDTH)
                    
                    # Calculate step-based reward: 0 by default, penalty if in a room
                    reward = 0

                    if game.owner_pos:
                        owner_r_after_move, owner_c_after_move = game.owner_pos
                        owner_tile_type_after_move = game._get_tile(owner_r_after_move, owner_c_after_move)
                        if owner_tile_type_after_move == HotelGenerator.ROOM_FLOOR:
                            reward += REWARD_IN_ROOM

                    if game.game_over:
                        reward += REWARD_CATCH_PLAYER if game.winner == "OWNER" else REWARD_PLAYER_ESCAPES
                    
                    if not moved_successfully and chosen_owner_action != Game.WAIT : 
                        reward += REWARD_BUMP_WALL
                    
                    if game.player_pos and game.owner_pos:
                        dist_to_player = math.sqrt((game.player_pos[0] - game.owner_pos[0])**2 + (game.player_pos[1] - game.owner_pos[1])**2)
                        if 0.5 < dist_to_player <= PROXIMITY_THRESHOLD: # 0.5 to avoid division by zero
                            reward += REWARD_PROXIMITY_MAX / dist_to_player
                    
                    do_log(f"Owner received reward for this turn: {reward}\n") # Added reward logging
                    episode_reward += reward
                    
                    owner_agent.store_transition(prev_owner_state_tensor_for_replay, chosen_owner_action, reward, next_owner_state_tensor, game.game_over)
                    
                    current_owner_state_tensor = next_owner_state_tensor 
                    game_turn_count += 1
                    total_steps_taken += 1

                if game.game_over:
                    do_log(f"--- GAME OVER ---\n")
                    do_log(f"Winner: {game.winner}\n")
                    do_log(game.render_grid_to_string(player_pov=False) + "\n\n")
                    break 
                
                if total_steps_taken > LEARN_START_STEPS and total_steps_taken % LEARN_EVERY_N_STEPS == 0:
                    owner_agent.learn()
            
            if not game.game_over and game_turn_count >= MAX_TURNS_PER_EPISODE:
                do_log(f"--- MAX TURNS ({MAX_TURNS_PER_EPISODE}) REACHED ---\n")
                do_log(game.render_grid_to_string(player_pov=False) + "\n\n")

            if log_file_handle: # Close the file if it was opened for this episode
                log_file_handle.close()
                log_file_handle = None # Good practice to reset

            owner_agent.decay_exploration()
            total_rewards_per_episode.append(episode_reward)

            postfix_stats = {
                "Epsilon": f"{owner_agent.epsilon:.4f}",
                "Steps": total_steps_taken,
                "Ep Reward": f"{episode_reward:.2f}"
            }
            if len(total_rewards_per_episode) >= PLOTTING_FREQUENCY:
                avg_reward = np.mean(total_rewards_per_episode[-PLOTTING_FREQUENCY:])
                postfix_stats[f"Avg Rwd ({PLOTTING_FREQUENCY})"] = f"{avg_reward:.2f}"
            elif total_rewards_per_episode:
                avg_reward = np.mean(total_rewards_per_episode)
                postfix_stats[f"Avg Rwd (All)"] = f"{avg_reward:.2f}"
            episode_pbar.set_postfix(postfix_stats)

            if (episode + 1) % PLOTTING_FREQUENCY == 0:
                periodic_model_save_path = os.path.join(MODEL_DIR, f"dqn_owner_agent_ep{episode+1}.pth")
                owner_agent.save_model(periodic_model_save_path)
                
                periodic_plot_filename = os.path.join(PLOTS_DIR, f"dqn_rewards_ep{episode+1}.png")
                plt.figure(figsize=(10,5))
                plt.plot(total_rewards_per_episode)
                plt.title(f"DQN Owner's Rewards up to Episode {episode+1}")
                plt.xlabel("Episode")
                plt.ylabel("Total Reward for Owner")
                plt.grid(True)
                plt.savefig(periodic_plot_filename)
                plt.close() 

    print("\n--- DQN TRAINING COMPLETE ---")
    final_model_save_path = os.path.join(MODEL_DIR, "dqn_owner_agent_final.pth")
    owner_agent.save_model(final_model_save_path)
    print(f"Final DQN model saved as {final_model_save_path}")

    final_plot_filename = os.path.join(PLOTS_DIR, "dqn_final_rewards_plot.png")
    plt.figure(figsize=(10,5))
    plt.plot(total_rewards_per_episode)
    plt.title("DQN Owner's Rewards per Episode (Full Training)")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward for Owner")
    plt.grid(True)
    plt.savefig(final_plot_filename)
    print(f"Final DQN rewards plot saved as {final_plot_filename}")
    # plt.show() 
    print(f"Periodic episode traces logged to '{LOGS_DIR}' directory.")


def test_trained_agent(model_filename="dqn_owner_agent_final.pth"):
    MODEL_DIR = "models"
    model_filepath = os.path.join(MODEL_DIR, model_filename)

    print(f"\n--- RUNNING WITH TRAINED DQN AGENT (FROM {model_filepath}) ---")
    
    GRID_HEIGHT = 25 
    GRID_WIDTH = 30
    PLAYER_MOVES_PER_TURN = 2

    game_params = { 'max_rooms': 8, 'room_min_size': 3, 'room_max_size': 5 } 
    MAX_TURNS_PER_EPISODE = 400 

    game_actions_list = [Game.MOVE_NORTH, Game.MOVE_SOUTH, Game.MOVE_EAST, Game.MOVE_WEST, Game.WAIT]
    
    trained_agent = DQNAgent(
        game_actions_list=game_actions_list,
        input_channels=NUM_DQN_CHANNELS,
        grid_height=GRID_HEIGHT,
        grid_width=GRID_WIDTH,
        exploration_rate_initial=0.0, 
        min_exploration_rate=0.0 
    )
    if not os.path.exists(model_filepath):
        print(f"ERROR: Model file not found at {model_filepath}. Cannot run test.")
        return
        
    trained_agent.load_model(model_filepath)
    trained_agent.q_network.eval() 

    game = Game(width=GRID_WIDTH, height=GRID_HEIGHT, player_moves_per_turn=PLAYER_MOVES_PER_TURN, generator_params=game_params)
    game_turn_count = 0 # Counts owner turns
    
    current_owner_state_tensor = get_dqn_state_representation(game, GRID_HEIGHT, GRID_WIDTH)

    while not game.game_over and game_turn_count < MAX_TURNS_PER_EPISODE:
        print(f"\n--- Game Turn {game_turn_count + 1} ---")
        current_entity = game.get_current_turn_entity()
        
        if current_entity == "PLAYER":
            game.print_grid_with_entities(player_pov=True)
            action_map = {'n': Game.MOVE_NORTH, 's': Game.MOVE_SOUTH, 'e': Game.MOVE_EAST, 'w': Game.MOVE_WEST, 'x': Game.WAIT}
            usr_input = input("Player action (n,s,e,w,x for wait): ").lower()
            action = action_map.get(usr_input)
            
            if action: 
                game.handle_player_turn(action)
            else: 
                print("Invalid action. Player waits.")
                game.handle_player_turn(Game.WAIT)
            current_owner_state_tensor = get_dqn_state_representation(game, GRID_HEIGHT, GRID_WIDTH) 
        
        elif current_entity == "OWNER":
            print("Owner (trained DQN) is thinking...")
            valid_owner_actions = [] 
            for owner_act_key in game_actions_list:
                if owner_act_key == Game.WAIT:
                    valid_owner_actions.append(owner_act_key)
                    continue
                dr_o, dc_o = Game.ACTION_DELTAS[owner_act_key]
                if game.owner_pos:
                    next_r_o, next_c_o = game.owner_pos[0] + dr_o, game.owner_pos[1] + dc_o
                    if game._is_walkable(next_r_o, next_c_o, "OWNER"):
                        valid_owner_actions.append(owner_act_key)
            if not valid_owner_actions and game.owner_pos: valid_owner_actions.append(Game.WAIT)
            elif not game.owner_pos: valid_owner_actions.append(Game.WAIT)


            owner_action = trained_agent.choose_action(current_owner_state_tensor.unsqueeze(0), game, valid_owner_actions)
            print(f"Owner chose: {owner_action}")
            game.handle_owner_turn(owner_action)
            current_owner_state_tensor = get_dqn_state_representation(game, GRID_HEIGHT, GRID_WIDTH)
            game_turn_count += 1
        
        if current_entity == "OWNER" or game.get_current_turn_entity() == "OWNER": # Show grid after owner or if player's turn ended
             game.print_grid_with_entities(player_pov=False)

    print("\n--- FINAL TRAINED DQN AGENT GAME STATE ---")
    game.print_grid_with_entities(player_pov=False)
    if game.game_over: 
        print(f"Result: {game.winner} wins after {game_turn_count} owner turns!")
    elif game_turn_count >= MAX_TURNS_PER_EPISODE:
        print(f"Max turns ({MAX_TURNS_PER_EPISODE}) reached in test run. No winner.")


if __name__ == '__main__':
    train_agent()