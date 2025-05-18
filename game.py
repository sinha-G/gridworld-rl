import math
import random
from gridworld_generator import HotelGenerator

class Game:
    MOVE_NORTH = "MOVE_NORTH"
    MOVE_SOUTH = "MOVE_SOUTH"
    MOVE_EAST = "MOVE_EAST"
    MOVE_WEST = "MOVE_WEST"
    WAIT = "WAIT"

    ACTION_DELTAS = {
        MOVE_NORTH: (-1, 0),
        MOVE_SOUTH: (1, 0),
        MOVE_EAST: (0, 1),
        MOVE_WEST: (0, -1),
        WAIT: (0, 0)
    }

    def __init__(self, width, height, player_moves_per_turn=2, player_vision_radius=5, owner_vision_radius=5, generator_params=None):
        if generator_params is None:
            generator_params = {} # Default empty dict
        self.generator = HotelGenerator(width, height)
        self.grid = []
        self.rooms_data = [] 

        self.player_pos = None 
        self.owner_pos = None  
        self.exit_pos = None

        self.player_moves_per_turn = player_moves_per_turn
        self.player_moves_taken_this_turn = 0

        self.player_vision_radius = player_vision_radius
        self.owner_vision_radius = owner_vision_radius
        self.player_hiding_vision_radius = max(1, player_vision_radius // 2)

        self.game_over = False
        self.winner = None 
        self.sound_alerts = [] 

        self._initialize_game(generator_params)

    def _initialize_game(self, generator_params):
        self.grid = self.generator.generate_hotel(**generator_params)
        self.rooms_data = self.generator.rooms

        player_start_pos = None
        owner_start_pos = None
        
        for r, row in enumerate(self.grid):
            for c, tile in enumerate(row):
                if tile == HotelGenerator.PLAYER:
                    player_start_pos = (r, c)
                    # We'll replace 'P' with its underlying tile after finding all special tiles
                elif tile == HotelGenerator.OWNER:
                    owner_start_pos = (r, c)
                    # We'll replace 'O' similarly
                elif tile == HotelGenerator.EXIT:
                    self.exit_pos = (r,c)

        # Determine underlying tiles for P and O before removing them
        if player_start_pos:
            self.player_pos = player_start_pos
            self.grid[player_start_pos[0]][player_start_pos[1]] = self._determine_underlying_tile(player_start_pos[0], player_start_pos[1])
        else:
            print("Warning: Player not found in generated grid. Attempting fallback placement.")
            self.player_pos = self._find_random_empty_tile([HotelGenerator.ROOM_FLOOR, HotelGenerator.HALLWAY])
            if self.player_pos is None: raise Exception("Could not place Player.")


        if owner_start_pos:
            self.owner_pos = owner_start_pos
            self.grid[owner_start_pos[0]][owner_start_pos[1]] = self._determine_underlying_tile(owner_start_pos[0], owner_start_pos[1])
        else:
            print("Warning: Owner not found in generated grid. Attempting fallback placement.")
            exclude_list = [self.player_pos] if self.player_pos else []
            self.owner_pos = self._find_random_empty_tile([HotelGenerator.ROOM_FLOOR, HotelGenerator.HALLWAY], exclude_pos=exclude_list)
            if self.owner_pos is None: raise Exception("Could not place Owner.")

        if not self.exit_pos:
            print("CRITICAL: Exit not found in generated grid! Attempting fallback.")
            self.exit_pos = self._find_random_empty_tile([HotelGenerator.WALL], on_border=True) # Place exit on a border wall
            if self.exit_pos:
                self.grid[self.exit_pos[0]][self.exit_pos[1]] = HotelGenerator.EXIT
            else:
                 raise Exception("Could not place Exit.")


        self.player_moves_taken_this_turn = 0
        self.game_over = False
        self.winner = None
        self.sound_alerts = []

    def _determine_underlying_tile(self, r, c):
        # Check if inside any room's floor area
        for room in self.rooms_data:
            if room['r'] <= r < room['r'] + room['height'] and \
               room['c'] <= c < room['c'] + room['width']:
                # If the generator could place P/O on H, this would need to check self.generator.grid
                # But assuming P/O are placed on what *becomes* ROOM_FLOOR or HALLWAY
                return HotelGenerator.ROOM_FLOOR
        return HotelGenerator.HALLWAY

    def _find_random_empty_tile(self, allowed_tile_types, exclude_pos=None, on_border=False):
        if exclude_pos is None: exclude_pos = []
        candidates = []
        for r_idx in range(self.generator.height):
            for c_idx in range(self.generator.width):
                if on_border and not (r_idx == 0 or r_idx == self.generator.height - 1 or c_idx == 0 or c_idx == self.generator.width - 1):
                    continue
                if self.grid[r_idx][c_idx] in allowed_tile_types and (r_idx, c_idx) not in exclude_pos:
                    candidates.append((r_idx, c_idx))
        return random.choice(candidates) if candidates else None

    def _is_valid_pos(self, r, c):
        return 0 <= r < self.generator.height and 0 <= c < self.generator.width

    def _get_tile(self, r, c):
        return self.grid[r][c] if self._is_valid_pos(r, c) else None

    def get_room_player_is_in(self, player_r, player_c):
        """Checks if the player at (player_r, player_c) is inside the floor area of any primary room."""
        if not self.rooms_data:
            return None
        for room_info in self.rooms_data:
            # Consider only primary room definitions (those without 'door_pos' key or where 'is_door_entry' is False)
            if not room_info.get('door_pos') and not room_info.get('is_door_entry'): # Check for primary room
                r, c, h, w = room_info['r'], room_info['c'], room_info['height'], room_info['width']
                # Check if (player_r, player_c) is within the room's floor boundaries
                # Ensure it's strictly inside, not on the wall perimeter
                if r <= player_r < r + h and \
                   c <= player_c < c + w and \
                   self.grid[player_r][player_c] == HotelGenerator.ROOM_FLOOR:
                    return room_info # Return the primary room object
        return None

    def get_doors_for_room(self, room_obj):
        """
        Finds all door tiles associated with a given primary room_obj
        and returns their details including connected_hallway.
        room_obj is a dictionary for a primary room.
        """
        if not room_obj or not self.rooms_data:
            return []

        doors_details = []
        room_r, room_c, room_h, room_w = room_obj['r'], room_obj['c'], room_obj['height'], room_obj['width']

        # Iterate over the perimeter of the room_obj
        # Top and Bottom walls
        for col in range(room_c, room_c + room_w):
            for row_val in [room_r, room_r + room_h - 1]:
                if self._is_valid_pos(row_val, col) and self.grid[row_val][col] == HotelGenerator.DOOR:
                    door_specific_room_data = self._get_room_at_door(row_val, col)
                    if door_specific_room_data and door_specific_room_data.get('connected_hallway'):
                        doors_details.append({
                            'door_pos': (row_val, col),
                            'connected_hallway': door_specific_room_data['connected_hallway']
                        })
        # Left and Right walls (avoid double counting corners)
        for row in range(room_r + 1, room_r + room_h - 1):
            for col_val in [room_c, room_c + room_w - 1]:
                if self._is_valid_pos(row, col_val) and self.grid[row][col_val] == HotelGenerator.DOOR:
                    door_specific_room_data = self._get_room_at_door(row, col_val)
                    if door_specific_room_data and door_specific_room_data.get('connected_hallway'):
                         doors_details.append({
                            'door_pos': (row, col_val),
                            'connected_hallway': door_specific_room_data['connected_hallway']
                        })
        # Deduplicate if any door was found twice (e.g. if room is 1x1 and door is a corner)
        # This is a simple deduplication based on door_pos
        final_doors = []
        seen_door_pos = set()
        for door_info in doors_details:
            if door_info['door_pos'] not in seen_door_pos:
                final_doors.append(door_info)
                seen_door_pos.add(door_info['door_pos'])
        return final_doors


    def is_pos_in_room_or_door(self, r, c, room_obj, target_door_pos):
        """
        Checks if (r,c) is within the floor of room_obj or is the target_door_pos.
        Used for constraining player movement when exiting a room.
        """
        if not room_obj: return False
        # Player can always move onto the target door tile itself
        if (r,c) == target_door_pos and self.grid[r][c] == HotelGenerator.DOOR:
            return True
        
        # Check if (r,c) is within the room's floor boundaries
        # Ensure it's strictly inside, not on the wall perimeter, and is a ROOM_FLOOR tile
        is_in_room_floor = (room_obj['r'] <= r < room_obj['r'] + room_obj['height'] and \
                             room_obj['c'] <= c < room_obj['c'] + room_obj['width'] and \
                             self.grid[r][c] == HotelGenerator.ROOM_FLOOR)
        return is_in_room_floor

    def _is_walkable(self, r, c, entity_type="PLAYER"):
        if not self._is_valid_pos(r, c): return False
        tile = self.grid[r][c]
        walkable_tiles = [
            HotelGenerator.HALLWAY, HotelGenerator.ROOM_FLOOR,
            HotelGenerator.HIDING_SPOT, HotelGenerator.DOOR, HotelGenerator.EXIT
        ]
        if tile in walkable_tiles:
            # Owner cannot walk directly *through* a door in one step if not already on it.
            # This is handled more explicitly in _handle_owner_move.
            return True
        return False

    def _get_room_at_door(self, door_r, door_c):
        """
        Finds the specific room data entry associated with a door at (door_r, door_c).
        This entry should contain 'connected_hallway'.
        """
        for room in self.rooms_data:
            if room.get('has_door') and room.get('door_pos') == (door_r, door_c):
                return room
        return None

    def _handle_player_move(self, new_r, new_c):
        target_tile = self._get_tile(new_r, new_c)

        if target_tile == HotelGenerator.DOOR:
            room_connected = self._get_room_at_door(new_r, new_c)
            if not room_connected: return False 

            current_tile_of_player = self._get_tile(self.player_pos[0], self.player_pos[1])
            
            # Player moving from Hallway (or standing on door from hall side) into Room
            if current_tile_of_player == HotelGenerator.HALLWAY or \
               (current_tile_of_player == HotelGenerator.DOOR and self.player_pos == room_connected.get('door_pos')): 
                for dr_adj, dc_adj in self.ACTION_DELTAS.values(): 
                    adj_r, adj_c = new_r + dr_adj, new_c + dc_adj
                    if self._is_valid_pos(adj_r, adj_c) and \
                       room_connected['r'] <= adj_r < room_connected['r'] + room_connected['height'] and \
                       room_connected['c'] <= adj_c < room_connected['c'] + room_connected['width'] and \
                       self.grid[adj_r][adj_c] == HotelGenerator.ROOM_FLOOR:
                        self.player_pos = (adj_r, adj_c)
                        # Player used the door at (new_r, new_c)
                        self.sound_alerts.append({'type': 'DOOR_USAGE_PLAYER', 'pos': (new_r, new_c), 'room_id': id(room_connected)})
                        return True
                return False 
            else: # Moving from Room into Hallway
                self.player_pos = room_connected['connected_hallway']
                # Player used the door at (new_r, new_c)
                self.sound_alerts.append({'type': 'DOOR_USAGE_PLAYER', 'pos': (new_r, new_c), 'room_id': id(room_connected)})
                return True
        else:
            self.player_pos = (new_r, new_c)
            return True

    def _handle_owner_move(self, new_r, new_c):
        target_tile = self._get_tile(new_r, new_c)
        owner_current_r, owner_current_c = self.owner_pos
        owner_current_tile = self._get_tile(owner_current_r, owner_current_c)

        if owner_current_tile == HotelGenerator.DOOR: # Owner is moving *from* a door tile
            self.owner_pos = (new_r, new_c)
            room_connected = self._get_room_at_door(owner_current_r, owner_current_c)
            if room_connected:
                self.sound_alerts.append({'type': 'DOOR_CLOSING', 'pos': (owner_current_r, owner_current_c), 'room_id': id(room_connected)})
            return True
        elif target_tile == HotelGenerator.DOOR: # Owner is moving *onto* a door tile
            self.owner_pos = (new_r, new_c)
            room_connected = self._get_room_at_door(new_r, new_c)
            if room_connected:
                self.sound_alerts.append({'type': 'DOOR_OPENING', 'pos': (new_r, new_c), 'room_id': id(room_connected)})
            return True # Owner's action ends, they are now on the door tile
        else: # Standard move
            self.owner_pos = (new_r, new_c)
            return True

    def _move_entity(self, entity_pos, action_key, entity_type="PLAYER"):
        if action_key == self.WAIT:
            # For WAIT, position doesn't change, but it's a valid action.
            # Specific handling for player/owner wait if needed (e.g. consuming a turn)
            if entity_type == "PLAYER": self.player_pos = entity_pos
            else: self.owner_pos = entity_pos
            return True

        dr, dc = self.ACTION_DELTAS.get(action_key, (0,0))
        current_r, current_c = entity_pos
        new_r, new_c = current_r + dr, current_c + dc

        if not self._is_walkable(new_r, new_c, entity_type):
            return False

        if entity_type == "PLAYER":
            return self._handle_player_move(new_r, new_c)
        elif entity_type == "OWNER":
            # Owner's door logic is more nuanced:
            # If owner is on a door, they can only move to an adjacent non-door.
            # If owner is not on a door, they can move to an adjacent door (1st step) or other walkable.
            owner_on_door = self._get_tile(current_r, current_c) == HotelGenerator.DOOR
            target_is_door = self._get_tile(new_r, new_c) == HotelGenerator.DOOR

            if owner_on_door and target_is_door: # Trying to move from door to door (invalid)
                return False
            
            return self._handle_owner_move(new_r, new_c)
        return False

    def _check_game_over_conditions(self):
        if self.player_pos == self.exit_pos:
            self.game_over = True
            self.winner = "PLAYER"
            return

        if self.player_pos == self.owner_pos:
            player_on_hiding_spot = self._get_tile(self.player_pos[0], self.player_pos[1]) == HotelGenerator.HIDING_SPOT
            if not player_on_hiding_spot: # Caught in the open or owner moved onto non-hiding player
                self.game_over = True
                self.winner = "OWNER"
            # If player is on hiding spot and owner is on same tile, owner "checked" it.
            elif player_on_hiding_spot and self.player_pos == self.owner_pos:
                 self.game_over = True
                 self.winner = "OWNER"

    def _get_visible_tiles(self, center_r, center_c, radius, entity_type="PLAYER"):
        visible = set()
        # Corrected iteration for circle
        for r_offset in range(-radius, radius + 1):
            for c_offset in range(-radius, radius + 1):
                if r_offset**2 + c_offset**2 <= radius**2:
                    r, c = center_r + r_offset, center_c + c_offset
                    if self._is_valid_pos(r, c):
                        # Add Line of Sight (LoS) check here
                        # For now, basic: if it's a wall, it blocks unless it's the tile itself.
                        # A real LoS would trace a line from center to (r,c) and check for walls.
                        # If self._get_tile(r,c) == HotelGenerator.WALL and (r,c) != (center_r, center_c):
                        # if not self._has_clear_los(center_r, center_c, r, c):
                        # continue
                        visible.add((r,c))
        return visible

    def _has_clear_los(self, r0, c0, r1, c1):
        """Checks if there's a clear line of sight between two points."""
        # Uses Bresenham's line algorithm or similar.
        # Returns True if no walls (excluding endpoints) are on the line.
        # This is a simplified version. A full implementation is more involved.
        dr = abs(r1 - r0)
        dc = abs(c1 - c0)
        r = r0
        c = c0
        sr = 1 if r1 > r0 else -1
        sc = 1 if c1 > c0 else -1
        err = dr - dc

        while True:
            if r == r1 and c == c1:
                break 
            
            # Check current point (r,c) if it's not the start point
            if not (r == r0 and c == c0): # Don't check the start point itself for being a wall
                 tile = self._get_tile(r,c)
                 if tile == HotelGenerator.WALL:
                     return False # Obstruction

            e2 = 2 * err
            if e2 > -dc:
                err -= dc
                r += sr
            if e2 < dr:
                err += dr
                c += sc
        return True

    def get_player_vision_data(self):
        current_player_tile = self._get_tile(self.player_pos[0], self.player_pos[1])
        radius = self.player_hiding_vision_radius if current_player_tile == HotelGenerator.HIDING_SPOT else self.player_vision_radius
        
        visible_coords = self._get_visible_tiles(self.player_pos[0], self.player_pos[1], radius, "PLAYER")
        
        seen_world = {}
        for r_coord, c_coord in visible_coords:
            if not self._has_clear_los(self.player_pos[0], self.player_pos[1], r_coord, c_coord):
                # If LoS is blocked, player might see a wall or nothing (depends on game rules)
                # For now, let's assume they see the first wall that blocks LoS, or the tile if LoS is clear.
                # This part needs careful thought for what the player actually perceives.
                # A simple approach: if LoS is blocked, they don't see that specific (r_coord, c_coord).
                # However, _get_visible_tiles + _has_clear_los should handle this.
                # If _has_clear_los is accurate, we only process tiles with clear LoS.
                continue # Skip if LoS is blocked by a wall to this specific tile

            tile_on_grid = self.grid[r_coord][c_coord]
            if (r_coord, c_coord) == self.owner_pos:
                seen_world[(r_coord, c_coord)] = "OWNER"
            else:
                seen_world[(r_coord, c_coord)] = tile_on_grid
        
        return seen_world, self.sound_alerts

    def get_owner_vision_data(self):
        visible_coords = self._get_visible_tiles(self.owner_pos[0], self.owner_pos[1], self.owner_vision_radius, "OWNER")
        seen_world = {}
        owner_is_on_door = self._get_tile(self.owner_pos[0], self.owner_pos[1]) == HotelGenerator.DOOR

        for r_coord, c_coord in visible_coords:
            if not self._has_clear_los(self.owner_pos[0], self.owner_pos[1], r_coord, c_coord):
                continue

            # If owner is on a door, their vision into the next area might be restricted.
            # This is a complex LoS problem. For now, if LoS is clear, they see.
            # A more advanced rule: if owner_is_on_door, and (r_coord, c_coord) is "past" the door frame, block.

            tile_on_grid = self.grid[r_coord][c_coord]
            if (r_coord, c_coord) == self.player_pos:
                player_actual_tile = self.grid[self.player_pos[0]][self.player_pos[1]]
                if player_actual_tile == HotelGenerator.HIDING_SPOT:
                    seen_world[(r_coord, c_coord)] = HotelGenerator.HIDING_SPOT # Sees spot, not player
                else:
                    seen_world[(r_coord, c_coord)] = "PLAYER"
            else:
                seen_world[(r_coord, c_coord)] = tile_on_grid
        return seen_world

    def handle_player_turn(self, action_key):
        if self.game_over or self.player_moves_taken_this_turn >= self.player_moves_per_turn:
            return False

        self.sound_alerts = [] # Clear alerts at start of player's action
        moved = self._move_entity(self.player_pos, action_key, "PLAYER")
        if moved:
            self.player_moves_taken_this_turn += 1
            self._check_game_over_conditions()
        return moved

    def handle_owner_turn(self, action_key):
        if self.game_over: return False
        
        self.player_moves_taken_this_turn = 0 # Reset for next player turn
        self.sound_alerts = [] # Clear before owner's move, new sounds might be generated by owner

        moved = self._move_entity(self.owner_pos, action_key, "OWNER")
        if moved:
            self._check_game_over_conditions()
        return moved

    def get_current_turn_entity(self):
        return "PLAYER" if self.player_moves_taken_this_turn < self.player_moves_per_turn else "OWNER"

    def print_grid_with_entities(self, player_pov=False):
        display_grid = [row[:] for row in self.grid]
        
        player_vision_map = None
        if player_pov:
            player_vision_map, _ = self.get_player_vision_data()

        for r in range(self.generator.height):
            for c in range(self.generator.width):
                if player_pov:
                    if (r,c) in player_vision_map:
                        tile_to_display = player_vision_map[(r,c)]
                        if tile_to_display == "OWNER": display_grid[r][c] = HotelGenerator.OWNER
                        elif tile_to_display == "PLAYER": display_grid[r][c] = HotelGenerator.PLAYER # Should be self
                        else: display_grid[r][c] = tile_to_display
                    else:
                        display_grid[r][c] = '?' # Fog of war
                else: # Omniscient view
                    if (r,c) == self.player_pos: display_grid[r][c] = HotelGenerator.PLAYER
                    elif (r,c) == self.owner_pos: display_grid[r][c] = HotelGenerator.OWNER
                    # Exit is already on self.grid

        # Ensure P/O are shown if player_pov and they are at their own loc
        if player_pov and self.player_pos in player_vision_map :
             display_grid[self.player_pos[0]][self.player_pos[1]] = HotelGenerator.PLAYER


        for row in display_grid:
            print("".join(row))
        
        print("-" * self.generator.width)
        print(f"Player: {self.player_pos} (Tile: {self._get_tile(self.player_pos[0], self.player_pos[1]) if self.player_pos else 'N/A'}) "
              f"(Moves left: {self.player_moves_per_turn - self.player_moves_taken_this_turn})")
        print(f"Owner: {self.owner_pos} (Tile: {self._get_tile(self.owner_pos[0], self.owner_pos[1]) if self.owner_pos else 'N/A'})")
        print(f"Exit: {self.exit_pos}")
        if self.sound_alerts: print(f"Sound Alerts: {self.sound_alerts}")
        if self.game_over: print(f"GAME OVER! Winner: {self.winner}")
