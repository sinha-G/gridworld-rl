import math
import random
import json
from gridworld_generator import HotelGenerator
from game import Game 


class QLearningAgent:
    def __init__(self, actions, learning_rate=0.1, discount_factor=0.9, 
                 exploration_rate=1.0, exploration_decay_rate=0.999, min_exploration_rate=0.01):
        self.actions = list(actions) # Ensure it's a list e.g. [Game.MOVE_NORTH, ...]
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.epsilon_decay = exploration_decay_rate
        self.epsilon_min = min_exploration_rate
        self.q_table = {}  # Using dict for sparse Q-table: {(state_tuple): {action_str: q_value}}

    def get_q_value(self, state, action):
        # State is expected to be a tuple, e.g., owner's position (r, c)
        return self.q_table.get(state, {}).get(action, 0.0)

    def choose_action(self, state, game_instance):
        # state is owner's current position (r, c)
        # game_instance is the Game object to check for valid moves
        
        valid_actions = []
        for act in self.actions:
            if act == Game.WAIT: # Wait is generally always valid
                valid_actions.append(act)
                continue
            
            dr, dc = Game.ACTION_DELTAS[act]
            next_r, next_c = state[0] + dr, state[1] + dc
            
            owner_on_door = game_instance._get_tile(state[0], state[1]) == HotelGenerator.DOOR
            target_is_door = game_instance._get_tile(next_r, next_c) == HotelGenerator.DOOR if game_instance._is_valid_pos(next_r,next_c) else False
            
            if owner_on_door and target_is_door: # Cannot move door to door directly
                continue
            if game_instance._is_walkable(next_r, next_c, "OWNER"):
                valid_actions.append(act)
        
        if not valid_actions: # Should not happen if WAIT is always an option and self.actions includes WAIT
            return Game.WAIT

        if random.random() < self.epsilon:
            return random.choice(valid_actions)  # Explore
        else:
            # Exploit: choose the best known valid action for the current state
            q_values_for_state = self.q_table.get(state, {})
            
            best_q = -float('inf')
            best_action = random.choice(valid_actions) # Default to random valid if no q-values or all are equal

            # Find the action with the max Q-value among valid actions
            # Shuffle to break ties randomly if multiple actions have the same max Q-value
            random.shuffle(valid_actions)
            found_best = False
            for action in valid_actions:
                q_val = q_values_for_state.get(action, 0.0)
                if q_val > best_q:
                    best_q = q_val
                    best_action = action
                    found_best = True
            
            if not found_best and valid_actions: # If all Q-values are 0 or state not seen
                 return random.choice(valid_actions)
            return best_action


    def update_q_table(self, state, action, reward, next_state, game_instance, done):
        old_q_value = self.get_q_value(state, action)
        
        next_max_q = 0.0
        if not done:
            # Find max Q-value for the next state over valid actions
            q_values_for_next_state = self.q_table.get(next_state, {})
            
            possible_next_q_values = []
            for act_prime in self.actions: # Consider all possible actions
                # Check if act_prime is valid from next_state
                is_next_action_valid = False
                if act_prime == Game.WAIT:
                    is_next_action_valid = True
                else:
                    dr, dc = Game.ACTION_DELTAS[act_prime]
                    n_r, n_c = next_state[0] + dr, next_state[1] + dc
                    owner_on_door = game_instance._get_tile(next_state[0], next_state[1]) == HotelGenerator.DOOR
                    target_is_door = game_instance._get_tile(n_r, n_c) == HotelGenerator.DOOR if game_instance._is_valid_pos(n_r,n_c) else False

                    if not (owner_on_door and target_is_door) and game_instance._is_walkable(n_r, n_c, "OWNER"):
                        is_next_action_valid = True
                
                if is_next_action_valid:
                    possible_next_q_values.append(q_values_for_next_state.get(act_prime, 0.0))
            
            if possible_next_q_values:
                next_max_q = max(possible_next_q_values)

        new_q_value = old_q_value + self.lr * (reward + self.gamma * next_max_q - old_q_value)
        
        if state not in self.q_table:
            self.q_table[state] = {}
        self.q_table[state][action] = new_q_value

    def decay_exploration(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save_q_table(self, filepath="q_table.json"):
        # Q-table keys are tuples, convert to strings for JSON
        saveable_q_table = {str(k): v for k, v in self.q_table.items()}
        with open(filepath, 'w') as f:
            json.dump(saveable_q_table, f)
        print(f"Q-table saved to {filepath}")

    def load_q_table(self, filepath="q_table.json"):
        try:
            with open(filepath, 'r') as f:
                loaded_q_table_str_keys = json.load(f)
                # Convert string keys back to tuples
                self.q_table = {eval(k): v for k, v in loaded_q_table_str_keys.items()}
            print(f"Q-table loaded from {filepath}")
        except FileNotFoundError:
            print(f"No Q-table found at {filepath}, starting fresh.")
        except json.JSONDecodeError:
            print(f"Error decoding Q-table from {filepath}, starting fresh.")
    
    def get_owner_state_representation(game_instance):
        owner_r, owner_c = game_instance.owner_pos
        
        # Player relative position (if seen)
        player_seen_in_vision = False
        rel_player_dr, rel_player_dc = None, None # Or some other default
        
        owner_vision_data = game_instance.get_owner_vision_data() # Assuming this returns a dict
        
        player_actual_pos = game_instance.player_pos # Get player's true position
        
        # Check if player is in owner's vision
        # The vision data might directly tell you "PLAYER" at a coord
        # Or you might need to check if player_actual_pos is in owner_vision_data keys
        if player_actual_pos:
            for loc, entity_type in owner_vision_data.items():
                if entity_type == "PLAYER": # Or if loc == player_actual_pos and owner can see it
                    player_seen_in_vision = True
                    rel_player_dr = player_actual_pos[0] - owner_r
                    rel_player_dc = player_actual_pos[1] - owner_c
                    break
        
        # Discretize relative positions to keep state space smaller
        # e.g., map rel_player_dr to categories: "North", "South", "Same_Row", "Far_North", etc.
        # For simplicity here, we might just use the raw relative offset if seen, or a placeholder
        
        # Example: Tile types immediately around owner
        # This requires the owner to "know" its immediate surroundings,
        # which it does via game.grid if we assume it can sense adjacent tiles.
        # Or, it could be based on its *vision* of adjacent tiles.
        # Let's assume for now it can sense its immediate grid neighbors.
        
        # For a tabular method, you need to be careful about how many distinct values each feature can take.
        # For instance, rel_player_dr could be many values. You might bin them:
        # e.g., if dr < -2: "Far_North", if dr == -1: "North", if dr == 0: "Same_R", etc.
        
        # Simplified example:
        state_features = [
            # owner_r, owner_c, # Maybe remove these if maps change too much and local perception is key
            "player_seen:" + str(player_seen_in_vision)
        ]
        if player_seen_in_vision:
            # Discretize relative positions
            if rel_player_dr < 0: state_features.append("player_dir_r:N")
            elif rel_player_dr > 0: state_features.append("player_dir_r:S")
            else: state_features.append("player_dir_r:0")

            if rel_player_dc < 0: state_features.append("player_dir_c:W")
            elif rel_player_dc > 0: state_features.append("player_dir_c:E")
            else: state_features.append("player_dir_c:0")
            
            # Could add discretized distance too
            # dist = math.sqrt(rel_player_dr**2 + rel_player_dc**2)
            # if dist < 3: state_features.append("player_dist:Close") else: state_features.append("player_dist:Far")

        # Add features about immediate surroundings (walls/openings)
        # This part needs careful thought about what the owner "knows" or "sees"
        # For now, let's assume it can sense adjacent tiles from its vision data or game.grid
        for dr_check, dc_check, direction_name in [(-1,0,"N"), (1,0,"S"), (0,-1,"W"), (0,1,"E")]:
            check_r, check_c = owner_r + dr_check, owner_c + dc_check
            tile_type_in_direction = "BOUNDARY" # Default if out of bounds
            if game_instance._is_valid_pos(check_r, check_c):
                # What does the owner perceive there?
                # Option A: From its vision data
                # tile_type_in_direction = owner_vision_data.get((check_r, check_c), HotelGenerator.WALL) # Assume unseen is wall-like
                # Option B: From the true grid (if owner has local sensors)
                actual_tile = game_instance.grid[check_r][check_c]
                if actual_tile == HotelGenerator.WALL:
                    tile_type_in_direction = "WALL"
                elif actual_tile in [HotelGenerator.ROOM_FLOOR, HotelGenerator.HALLWAY, HotelGenerator.DOOR, HotelGenerator.HIDING_SPOT, HotelGenerator.EXIT]:
                    tile_type_in_direction = "OPEN"
                # Add more specific types if needed
            state_features.append(f"adj_{direction_name}:{tile_type_in_direction}")
            
        return tuple(sorted(state_features)) # Convert to tuple, sort for canonical representation
