import random

class HotelGenerator:
    # Tile Types
    WALL = "#"
    HALLWAY = " "
    ROOM_FLOOR = "."
    DOOR = "D"
    HIDING_SPOT = "H"
    PLAYER = "P"
    OWNER = "O"
    EXIT = "E"

    # Directions for carving/checking neighbors (North, East, South, West)
    DIRECTIONS = [(-1, 0), (0, 1), (1, 0), (0, -1)]

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.grid = []
        self.rooms = [] # List to store room data dictionaries

    def _initialize_grid(self):
        """Fills the grid with walls and resets room list."""
        self.grid = [[self.WALL for _ in range(self.width)] for _ in range(self.height)]
        self.rooms = []

    def _is_valid(self, r, c, margin=0):
        """Checks if coordinates (r, c) are within grid boundaries, with an optional margin."""
        return margin <= r < self.height - margin and \
               margin <= c < self.width - margin

    def print_grid(self):
        """Prints the current state of the grid to the console."""
        if not self.grid:
            print("Grid not generated yet.")
            return
        for row in self.grid:
            print("".join(row))
        print("-" * self.width)

    def generate_hotel(self, straightness_hallways, hall_loops, 
                       max_hallway_perc=0.20, max_rooms=10, room_min_size=3, 
                       room_max_size=4, max_hiding_spots_per_room=1):
        """
        Generates the full hotel layout.
        Returns the grid (list of lists of strings).
        """
        self._initialize_grid()

        # print("1. Generating hallways...")
        self._generate_hallways_dfs_like(straightness_factor=straightness_hallways, 
                                         num_additional_loops=hall_loops,
                                         max_hallway_percentage=max_hallway_perc) # Pass it here

        # print("2. Placing rooms...")
        self._place_rooms_off_hallways(max_rooms_to_place=max_rooms,
                                       min_dim=room_min_size, max_dim=room_max_size)

        # print("3. Adding doors to rooms...")
        self._add_doors_to_all_rooms()
        
        # self.rooms = [room for room in self.rooms if room.get('has_door')]

        # if not self.rooms:
        #     print("Warning: No rooms with doors were successfully generated. Player/Owner/Exit may not be placed correctly.")
        # else:
        #     print(f"Successfully generated {len(self.rooms)} rooms with doors.")
        
        # print("4. Placing hiding spots...")
        self._place_hiding_spots_in_rooms(max_spots_per_room=max_hiding_spots_per_room)

        # print("5. Placing player and owner...")
        self._place_characters()

        # print("6. Placing exit...")
        self._place_exit_tile() 

        # print("7. Validating hallway adjacencies...")
        # if not self._validate_hallway_adjacencies():
            # print("Warning: Some hallway tiles have invalid adjacencies.")
        
        # print("Hotel generation complete!")
        return self.grid

    def _generate_hallways_dfs_like(self, straightness_factor=0.75, 
                                    num_additional_loops=0, 
                                    max_hallway_percentage=0.20): # Added max_hallway_percentage
        """
        Generates 1-tile wide hallways using a DFS-like approach with a bias for straightness.
        Stops when a certain percentage of the grid is converted to hallways.
        Then, optionally adds more loops.
        """
        
        margin = 1 
        start_r = (self.height // 2) | 1 
        start_c = (self.width // 2) | 1
        if not self._is_valid(start_r,start_c, margin):
            start_r = margin if margin < self.height -1 - margin else self.height // 2
            start_c = margin if margin < self.width -1 - margin else self.width // 2

        self.grid[start_r][start_c] = self.HALLWAY
        
        hallway_cell_count = 1 
        # Calculate target based on the carvable area (inside margins)
        carvable_height = self.height - 2 * margin
        carvable_width = self.width - 2 * margin
        if carvable_height <= 0 : carvable_height = self.height # Failsafe
        if carvable_width <= 0 : carvable_width = self.width   # Failsafe
        total_carvable_cells = carvable_height * carvable_width
        
        target_hallway_cells = int(total_carvable_cells * max_hallway_percentage)
        # Ensure a minimum reasonable number of hallway cells are targeted
        min_target_hallways = (self.height + self.width)       # Heuristic minimum
        if target_hallway_cells < min_target_hallways : target_hallway_cells = min_target_hallways
        if target_hallway_cells < 1 : target_hallway_cells = 1 # Absolute minimum

        stack = [(start_r, start_c, -1)] 
        
        while stack and hallway_cell_count < target_hallway_cells: # Check budget
            curr_r, curr_c, prev_dir_idx = stack[-1]
            possible_moves = [] 

            shuffled_directions = list(self.DIRECTIONS)
            random.shuffle(shuffled_directions)

            for idx, (dr, dc) in enumerate(shuffled_directions):
                wall_between_r, wall_between_c = curr_r + dr, curr_c + dc
                next_hall_r, next_hall_c = curr_r + dr * 2, curr_c + dc * 2

                if self._is_valid(next_hall_r, next_hall_c, margin) and \
                   self.grid[next_hall_r][next_hall_c] == self.WALL and \
                   self._is_valid(wall_between_r, wall_between_c) and \
                   self.grid[wall_between_r][wall_between_c] == self.WALL:
                    possible_moves.append((dr, dc, idx))
            
            if possible_moves:
                chosen_dr, chosen_dc, chosen_dir_idx = -1,-1,-1
                went_straight = False
                if prev_dir_idx != -1: 
                    for move_dr, move_dc, move_idx in possible_moves:
                        if move_idx == prev_dir_idx: 
                            if random.random() < straightness_factor:
                                chosen_dr, chosen_dc, chosen_dir_idx = move_dr, move_dc, move_idx
                                went_straight = True
                            break 
                
                if not went_straight: 
                    non_straight_options = [m for m in possible_moves if m[2] != prev_dir_idx]
                    if non_straight_options:
                        chosen_dr, chosen_dc, chosen_dir_idx = random.choice(non_straight_options)
                    elif possible_moves: 
                        chosen_dr, chosen_dc, chosen_dir_idx = random.choice(possible_moves)
                    else: 
                        stack.pop() 
                        continue

                # Check budget before carving the 2 new hallway cells
                if hallway_cell_count + 2 > target_hallway_cells:
                    # Not enough budget for this full step, effectively stop exploring this path
                    # Or we could just stop the whole process if we want a hard cap.
                    # For now, let's make it stop this path and try to finish other branches
                    # if any are left on stack that are still under budget.
                    # A simpler approach is to break the main while loop if very close to budget
                    # For now, just stop carving on this path to prevent large overshoots.
                    stack.pop() # Remove current path head
                    continue    # Try another path from stack if available

                wall_to_carve_r, wall_to_carve_c = curr_r + chosen_dr, curr_c + chosen_dc
                new_hall_r, new_hall_c = curr_r + chosen_dr * 2, curr_c + chosen_dc * 2
                                
                self.grid[wall_to_carve_r][wall_to_carve_c] = self.HALLWAY
                hallway_cell_count += 1
                if hallway_cell_count >= target_hallway_cells: # Check budget after first carve
                    self.grid[new_hall_r][new_hall_c] = self.HALLWAY # Carve the second cell too
                    hallway_cell_count +=1
                    break # Break from the while stack loop

                self.grid[new_hall_r][new_hall_c] = self.HALLWAY
                hallway_cell_count += 1
                
                stack.append((new_hall_r, new_hall_c, chosen_dir_idx))
            else:
                stack.pop()
        
        # print(f"Hallway generation: Target hallway cells ~{target_hallway_cells}, Actual carved hallway cells: {hallway_cell_count}")

        if num_additional_loops > 0 and hallway_cell_count > 0: # Ensure some hallways exist before trying to add loops
            loop_candidates = []
            # Iterate an inner part of the grid to find H-W-H patterns
            # Reduced margin for loop candidates to allow loops closer to edge hallways
            loop_margin = margin 
            for r in range(loop_margin, self.height - loop_margin): 
                for c in range(loop_margin, self.width - loop_margin):
                    if not self._is_valid(r,c): continue # Should not happen with loop_margin logic
                    if self.grid[r][c] == self.WALL:
                        # Check horizontal H-W-H: grid[r][c-1]=H, grid[r][c]=W, grid[r][c+1]=H
                        # AND ensure the perpendiculars are walls to prefer opening up simple corridors
                        if self._is_valid(r,c-1) and self.grid[r][c-1] == self.HALLWAY and \
                           self._is_valid(r,c+1) and self.grid[r][c+1] == self.HALLWAY and \
                           self._is_valid(r-1,c) and self.grid[r-1][c] == self.WALL and \
                           self._is_valid(r+1,c) and self.grid[r+1][c] == self.WALL:
                            loop_candidates.append((r,c))
                        # Check vertical H-W-H: grid[r-1][c]=H, grid[r][c]=W, grid[r+1][c]=H
                        # AND ensure the perpendiculars are walls
                        elif self._is_valid(r-1,c) and self.grid[r-1][c] == self.HALLWAY and \
                             self._is_valid(r+1,c) and self.grid[r+1][c] == self.HALLWAY and \
                             self._is_valid(r,c-1) and self.grid[r][c-1] == self.WALL and \
                             self._is_valid(r,c+1) and self.grid[r][c+1] == self.WALL:
                            loop_candidates.append((r,c))
            
            random.shuffle(loop_candidates)
            loops_created = 0
            for i in range(min(num_additional_loops, len(loop_candidates))):
                lr, lc = loop_candidates[i]
                # Re-check condition before carving, in case another loop carving changed things
                is_h_hwh = self._is_valid(r,c-1) and self.grid[lr][lc-1] == self.HALLWAY and \
                           self._is_valid(r,c+1) and self.grid[lr][lc+1] == self.HALLWAY and \
                           self._is_valid(r-1,c) and self.grid[lr-1][lc] == self.WALL and \
                           self._is_valid(r+1,c) and self.grid[lr+1][lc] == self.WALL
                is_v_hwh = self._is_valid(r-1,c) and self.grid[lr-1][lc] == self.HALLWAY and \
                           self._is_valid(r+1,c) and self.grid[lr+1][lc] == self.HALLWAY and \
                           self._is_valid(r,c-1) and self.grid[lr][lc-1] == self.WALL and \
                           self._is_valid(r,c+1) and self.grid[lr][lc+1] == self.WALL
                if self.grid[lr][lc] == self.WALL and (is_h_hwh or is_v_hwh):
                    self.grid[lr][lc] = self.HALLWAY
                    loops_created +=1
            # print(f"Attempted to create {num_additional_loops} additional loops, successfully created {loops_created}.")


    def _can_place_room_at(self, room_r, room_c, room_h, room_w,
                               door_site_r, door_site_c,
                               hall_conn_r, hall_conn_c):
        """
        Checks if a room (floor and its walls) can be placed without conflicts.
        - room_r, room_c: top-left of the room's FLOOR.
        - room_h, room_w: dimensions of the room's FLOOR.
        - door_site_r, door_site_c: the WALL tile that will become the DOOR.
        - hall_conn_r, hall_conn_c: the HALLWAY tile the door connects to.
        """
        # 1. Check overall bounds for room floor and its direct walls
        # The room's walls are at room_r-1, room_r+room_h, room_c-1, room_c+room_w
        if not (self._is_valid(room_r - 1, room_c - 1) and    # Top-left corner of wall structure
                  self._is_valid(room_r + room_h, room_c + room_w)): # Bottom-right corner of wall structure
            # print(f"Debug: Overall bounds check failed for room at {room_r},{room_c} ({room_h}x{room_w})")
            return False

        # 2. Check the hallway connection tile (must be existing HALLWAY)
        if not (self._is_valid(hall_conn_r, hall_conn_c) and \
                self.grid[hall_conn_r][hall_conn_c] == self.HALLWAY):
            # print(f"Debug: Hallway connection tile ({hall_conn_r},{hall_conn_c}) is not HALLWAY.")
            return False

        # 3. Check the door site tile (must be existing WALL)
        if not (self._is_valid(door_site_r, door_site_c) and \
                self.grid[door_site_r][door_site_c] == self.WALL):
            # print(f"Debug: Door site tile ({door_site_r},{door_site_c}) is not WALL.")
            return False

        # 4. Check adjacency of door site and hallway connection (Manhattan distance should be 1)
        if abs(door_site_r - hall_conn_r) + abs(door_site_c - hall_conn_c) != 1:
             # This would typically indicate a logical error in how these parameters were derived by the caller
            #  print(f"Debug: Door site ({door_site_r},{door_site_c}) not strictly adjacent to hall connection ({hall_conn_r},{hall_conn_c}).")
             return False

        # 5. Check the room's proposed floor area and its own walls
        for r_check in range(room_r - 1, room_r + room_h + 1):
            for c_check in range(room_c - 1, room_c + room_w + 1):
                current_grid_tile_val = self.grid[r_check][c_check]

                is_part_of_floor = (room_r <= r_check < room_r + room_h and
                                    room_c <= c_check < room_c + room_w)
                is_the_door_site = (r_check == door_site_r and c_check == door_site_c)
                is_the_hall_connection = (r_check == hall_conn_r and c_check == hall_conn_c)

                if is_part_of_floor:
                    if current_grid_tile_val != self.WALL:
                        # print(f"Debug: Proposed floor tile ({r_check},{c_check}) is not WALL, but '{current_grid_tile_val}'.")
                        return False # Floor area must be carved from wall
                elif is_the_door_site:
                    # This was already checked in step 3 to be WALL.
                    pass
                elif is_the_hall_connection:
                    # The room's own structure (floor/walls) should not overlap the hallway tile it connects to.
                    # If (r_check,c_check) is the hall_conn_tile, it means the room's bounding box includes it, which is wrong.
                    # print(f"Debug: Room structure at ({r_check},{c_check}) conflicts with hall connection tile ({hall_conn_r},{hall_conn_c}).")
                    return False
                else: # This is a proposed wall of the room (not the door_site wall)
                    if current_grid_tile_val != self.WALL:
                        # print(f"Debug: Proposed room wall tile ({r_check},{c_check}) is not WALL, but '{current_grid_tile_val}'.")
                        return False # Room's other walls must also be on existing WALLs

        # 6. Crucial Geometry Check: Ensure the door_site_is_validly_part_of_the_room's_perimeter.
        # Based on how floor_r, floor_c were calculated in _place_rooms_off_hallways relative to door_site_
        # this check validates that alignment.
        door_on_north_wall = (door_site_r == room_r - 1    and room_c <= door_site_c < room_c + room_w)
        door_on_south_wall = (door_site_r == room_r + room_h and room_c <= door_site_c < room_c + room_w)
        door_on_west_wall  = (door_site_c == room_c - 1    and room_r <= door_site_r < room_r + room_h)
        door_on_east_wall  = (door_site_c == room_c + room_w and room_r <= door_site_r < room_r + room_h)

        if not (door_on_north_wall or door_on_south_wall or door_on_west_wall or door_on_east_wall):
            # This indicates a mismatch between the room's floor coordinates and the door's position.
            # print(f"Debug: Door site ({door_site_r},{door_site_c}) is not geometrically aligned as a direct wall for room floor at ({room_r},{room_c}) dim ({room_h}x{room_w}).")
            return False
        return True

    def _place_rooms_off_hallways(self, max_rooms_to_place, min_dim, max_dim):
        """
        Identifies potential door locations (walls next to hallways) and tries to build rooms there.
        """
        potential_sites = [] # Store as ((hall_r, hall_c), (door_dir_r, door_dir_c from hall_r,hall_c))
        for r in range(self.height):
            for c in range(self.width):
                if self.grid[r][c] == self.HALLWAY:
                    for dr, dc in self.DIRECTIONS: # For each direction from hallway tile
                        wall_cand_r, wall_cand_c = r + dr, c + dc
                        if self._is_valid(wall_cand_r, wall_cand_c) and \
                           self.grid[wall_cand_r][wall_cand_c] == self.WALL:
                            potential_sites.append(((r,c), (dr,dc)))
        
        random.shuffle(potential_sites)
        rooms_placed_count = 0

        for (hall_r, hall_c), (door_dir_r, door_dir_c) in potential_sites:
            if rooms_placed_count >= max_rooms_to_place: break

            door_site_r, door_site_c = hall_r + door_dir_r, hall_c + door_dir_c

            for attempt_num in range(5): # Increased attempts with different sizes
                room_h = random.randint(min_dim, max_dim)
                room_w = random.randint(min_dim, max_dim)

                floor_r, floor_c = -1, -1 

                if door_dir_r == -1: 
                    floor_r = door_site_r - room_h
                    floor_c = door_site_c - (room_w // 2)
                elif door_dir_r == 1: 
                    floor_r = door_site_r + 1
                    floor_c = door_site_c - (room_w // 2)
                elif door_dir_c == -1: 
                    floor_r = door_site_r - (room_h // 2)
                    floor_c = door_site_c - room_w
                elif door_dir_c == 1: 
                    floor_r = door_site_r - (room_h // 2)
                    floor_c = door_site_c + 1
                
                # --- PRELIMINARY CHECK ---
                floor_is_valid = self._is_valid(floor_r, floor_c)
                floor_end_is_valid = self._is_valid(floor_r + room_h - 1, floor_c + room_w - 1)

                # print(f"Attempt {attempt_num+1}: door_site({door_site_r},{door_site_c}), dir({door_dir_r},{door_dir_c}), size({room_h}x{room_w}) -> floor_coords({floor_r},{floor_c})") # General Info Print

                if not (floor_is_valid and floor_end_is_valid):
                    # THIS IS THE CRITICAL DEBUG PRINT WE NEED TO SEE FOR NEGATIVE floor_r
                    # print(f"DEBUG_PRELIM_CHECK: floor_r={floor_r}, floor_c={floor_c}, room_h={room_h}, room_w={room_w}. floor_is_valid={floor_is_valid}, floor_end_is_valid={floor_end_is_valid}. SKIPPING.")
                    continue 
                
                # If preliminary check passed, print info before calling _can_place_room_at
                # print(f"DEBUG_PRELIM_PASSED: floor_r={floor_r}, floor_c={floor_c}. Calling _can_place_room_at.")

                if self._can_place_room_at(floor_r, floor_c, room_h, room_w,
                                           door_site_r, door_site_c, hall_r, hall_c):
                    # (This part seems okay if _can_place_room_at is correct)
                    for r_offset in range(room_h):
                        for c_offset in range(room_w):
                            self.grid[floor_r + r_offset][floor_c + c_offset] = self.ROOM_FLOOR
                    
                    self.rooms.append({
                        'r': floor_r, 'c': floor_c, 'height': room_h, 'width': room_w,
                        'door_pos': (door_site_r, door_site_c),
                        'connected_hallway': (hall_r, hall_c),
                        'has_door': False
                    })
                    rooms_placed_count += 1
                    break
    
    def _add_doors_to_all_rooms(self):
        """
        Iterates through all placed rooms and attempts to convert their
        'door_pos' into an actual DOOR tile.
        Ensures each room gets exactly one door if conditions are met.
        """
        for room in self.rooms:
            if room['has_door']: continue # Already has a door (shouldn't happen here)

            dr, dc = room['door_pos']
            hr, hc = room['connected_hallway'] # Hallway this door should connect to

            # Verify conditions: door candidate is WALL, connects to the specified HALLWAY,
            # and is adjacent to this room's FLOOR.
            adj_to_this_room_floor = False
            for r_offset, c_offset in self.DIRECTIONS:
                check_r, check_c = dr + r_offset, dc + c_offset
                if room['r'] <= check_r < room['r'] + room['height'] and \
                   room['c'] <= check_c < room['c'] + room['width'] and \
                   self.grid[check_r][check_c] == self.ROOM_FLOOR:
                    adj_to_this_room_floor = True
                    break
            
            if self._is_valid(dr,dc) and self.grid[dr][dc] == self.WALL and \
               self._is_valid(hr,hc) and self.grid[hr][hc] == self.HALLWAY and \
               adj_to_this_room_floor:
                
                # Check if the hallway tile is indeed adjacent to the door candidate
                is_hall_adj_to_door_cand = False
                for move_r, move_c in self.DIRECTIONS:
                    if (dr + move_r, dc + move_c) == (hr, hc):
                        is_hall_adj_to_door_cand = True
                        break
                
                if is_hall_adj_to_door_cand:
                    self.grid[dr][dc] = self.DOOR
                    room['door_pos'] = (dr, dc) # Store actual door position
                    room['has_door'] = True
                else:
                    print(f"Warning: Door candidate ({dr},{dc}) for room at ({room['r']},{room['c']}) not adjacent to its designated hallway ({hr},{hc}).")
            else:
                print(f"Warning: Could not place door for room at ({room['r']},{room['c']}) at candidate ({dr},{dc}). Conditions not met.")
                print(f"  Door tile: {self.grid[dr][dc] if self._is_valid(dr,dc) else 'OOB'}")
                print(f"  Hall tile: {self.grid[hr][hc] if self._is_valid(hr,hc) else 'OOB'}")
                print(f"  Adj to room floor: {adj_to_this_room_floor}")

    def _get_placeable_tiles_in_room(self, room_data):
        """Returns a list of (r, c) tuples for tiles within a room where items/chars can be placed."""
        placeable = []
        door_pos = room_data.get('door_pos')

        for r_offset in range(room_data['height']):
            for c_offset in range(room_data['width']):
                r, c = room_data['r'] + r_offset, room_data['c'] + c_offset
                if self.grid[r][c] == self.ROOM_FLOOR: # Only on actual floor tiles
                    is_too_close_to_door = False
                    if door_pos:
                        # Check if (r,c) is the door_pos itself or directly adjacent to it
                        if (r,c) == door_pos or \
                           (abs(r - door_pos[0]) + abs(c - door_pos[1]) == 1) : # Manhattan dist 1
                           is_too_close_to_door = True
                    if not is_too_close_to_door:
                        placeable.append((r,c))
        return placeable

    def _place_hiding_spots_in_rooms(self, max_spots_per_room=2):
        """Places 0 to max_spots_per_room hiding spots in each valid room."""
        for room in [r for r in self.rooms if r.get('has_door')]:
            if not room.get('has_door'): continue # Skip rooms without a door

            placeable_tiles = self._get_placeable_tiles_in_room(room)
            if not placeable_tiles: continue

            num_to_place = random.randint(0, min(max_spots_per_room, len(placeable_tiles)))
            
            for _ in range(num_to_place):
                spot_r, spot_c = random.choice(placeable_tiles)
                self.grid[spot_r][spot_c] = self.HIDING_SPOT
                placeable_tiles.remove((spot_r, spot_c)) # Avoid placing multiple spots on same tile

    def _place_characters(self):
        """Places the Player and Owner in different, random, valid rooms."""
        valid_rooms_for_spawn = [r for r in self.rooms if r.get('has_door')]

        if len(valid_rooms_for_spawn) == 0:
            # print("Warning: No valid rooms to spawn characters. Attempting hallway spawn.")
            self._spawn_character_in_hallway(self.PLAYER)
            self._spawn_character_in_hallway(self.OWNER)
            return
        
        if len(valid_rooms_for_spawn) == 1:
            # print("Warning: Only one valid room. Player and Owner will spawn in the same room if possible, or hallway.")
            room_choice = valid_rooms_for_spawn[0]
            spawn_tiles = self._get_placeable_tiles_in_room(room_choice)
            random.shuffle(spawn_tiles)
            if spawn_tiles:
                pr, pc = spawn_tiles.pop()
                self.grid[pr][pc] = self.PLAYER
            else: self._spawn_character_in_hallway(self.PLAYER)
            
            if spawn_tiles:
                or_val, oc_val = spawn_tiles.pop()
                self.grid[or_val][oc_val] = self.OWNER
            else: self._spawn_character_in_hallway(self.OWNER)
            return

        # len(valid_rooms_for_spawn) >= 2
        player_room, owner_room = random.sample(valid_rooms_for_spawn, 2)

        player_spawn_tiles = self._get_placeable_tiles_in_room(player_room)
        if player_spawn_tiles:
            pr, pc = random.choice(player_spawn_tiles)
            self.grid[pr][pc] = self.PLAYER
        else:
            # print(f"Warning: No placeable tiles in Player's chosen room ({player_room['r']},{player_room['c']}). Spawning in hallway.")
            self._spawn_character_in_hallway(self.PLAYER)

        owner_spawn_tiles = self._get_placeable_tiles_in_room(owner_room)
        if owner_spawn_tiles:
            # Ensure owner doesn't spawn on player if by some fluke they ended up in same tile after all.
            # This check is mostly for the single room case, but good to have if rooms could overlap.
            chosen_owner_tile = random.choice(owner_spawn_tiles)
            if self.grid[chosen_owner_tile[0]][chosen_owner_tile[1]] == self.ROOM_FLOOR:
                 self.grid[chosen_owner_tile[0]][chosen_owner_tile[1]] = self.OWNER
            else: # Tile taken (e.g. by player, if rooms were same and only 1 spot) - try another
                owner_spawn_tiles.remove(chosen_owner_tile)
                if owner_spawn_tiles:
                    chosen_owner_tile = random.choice(owner_spawn_tiles)
                    self.grid[chosen_owner_tile[0]][chosen_owner_tile[1]] = self.OWNER
                else:
                    print(f"Warning: No placeable tiles left in Owner's chosen room ({owner_room['r']},{owner_room['c']}) after player. Spawning in hallway.")
                    self._spawn_character_in_hallway(self.OWNER)
        else:
            print(f"Warning: No placeable tiles in Owner's chosen room ({owner_room['r']},{owner_room['c']}). Spawning in hallway.")
            self._spawn_character_in_hallway(self.OWNER)

    def _spawn_character_in_hallway(self, char_tile):
        """Fallback to spawn a character in a random empty hallway tile."""
        hallway_tiles = []
        for r in range(self.height):
            for c in range(self.width):
                if self.grid[r][c] == self.HALLWAY: # Empty hallway tile
                    hallway_tiles.append((r,c))
        if hallway_tiles:
            sr, sc = random.choice(hallway_tiles)
            self.grid[sr][sc] = char_tile
            # print(f"Placed {char_tile} in hallway at ({sr},{sc}).")
        else:
            print(f"CRITICAL: No hallway tiles available to spawn {char_tile}!")

    def _place_exit_tile(self):
        """Places the Exit tile, prioritizing walls adjacent to exactly one hallway tile,
        not adjacent to doors or room floors, and preferably on the border."""
        possible_exit_sites = [] # List of ( (exit_r, exit_c), is_border_wall_flag )

        for r_wall in range(self.height):
            for c_wall in range(self.width):
                if self.grid[r_wall][c_wall] == self.WALL:
                    hallway_neighbors = 0
                    problematic_neighbors = 0 # Counts adjacent Doors or Room Floors

                    for dr, dc in self.DIRECTIONS:
                        nr, nc = r_wall + dr, c_wall + dc
                        if self._is_valid(nr, nc):
                            neighbor_tile = self.grid[nr][nc]
                            if neighbor_tile == self.HALLWAY:
                                hallway_neighbors += 1
                            elif neighbor_tile == self.DOOR or neighbor_tile == self.ROOM_FLOOR:
                                problematic_neighbors += 1
                    
                    if hallway_neighbors == 1 and problematic_neighbors == 0:
                        is_border = (r_wall == 0 or r_wall == self.height - 1 or \
                                     c_wall == 0 or c_wall == self.width - 1)
                        possible_exit_sites.append(((r_wall, c_wall), is_border))

        chosen_exit_site = None

        if possible_exit_sites:
            # Prefer border walls that meet the new criteria
            border_exit_coords = [site_coords for site_coords, is_b in possible_exit_sites if is_b]
            if border_exit_coords:
                chosen_exit_site = random.choice(border_exit_coords)
            else:
                # No border walls found, so pick from any wall meeting the new criteria
                all_possible_coords = [site_coords for site_coords, _ in possible_exit_sites]
                if all_possible_coords: # Should be true if possible_exit_sites is not empty
                    chosen_exit_site = random.choice(all_possible_coords)
        
        if chosen_exit_site:
            self.grid[chosen_exit_site[0]][chosen_exit_site[1]] = self.EXIT
            # print(f"Exit placed at {chosen_exit_site} based on new criteria.")
        else:
            # Fallback: Try original logic if new criteria yield no results
            # print("Warning: New exit criteria yielded no valid spots. Falling back to original logic.")
            original_possible_exit_walls = []
            for r in range(self.height):
                for c in range(self.width):
                    if self.grid[r][c] == self.HALLWAY:
                        for dr, dc in self.DIRECTIONS:
                            wall_r, wall_c = r + dr, c + dc
                            if self._is_valid(wall_r, wall_c) and self.grid[wall_r][wall_c] == self.WALL:
                                is_border = (wall_r == 0 or wall_r == self.height - 1 or \
                                             wall_c == 0 or wall_c == self.width - 1)
                                original_possible_exit_walls.append(((wall_r, wall_c), is_border))
            
            fallback_chosen_site = None
            if original_possible_exit_walls:
                border_fallback_coords = [site_coords for site_coords, is_b in original_possible_exit_walls if is_b]
                if border_fallback_coords:
                    fallback_chosen_site = random.choice(border_fallback_coords)
                else:
                    all_fallback_coords = [site_coords for site_coords, _ in original_possible_exit_walls]
                    if all_fallback_coords:
                        fallback_chosen_site = random.choice(all_fallback_coords)
            
            if fallback_chosen_site:
                self.grid[fallback_chosen_site[0]][fallback_chosen_site[1]] = self.EXIT
                # print(f"Exit placed at {fallback_chosen_site} using original fallback logic.")
            else:
                # Further Fallback: Place on any border hallway if still no exit
                # print("Warning: Original exit logic also failed. Attempting to place on any border hallway.")
                border_hallways = []
                for r_idx in range(self.height):
                    for c_idx in range(self.width):
                        if (r_idx == 0 or r_idx == self.height - 1 or c_idx == 0 or c_idx == self.width - 1) and \
                           self.grid[r_idx][c_idx] == self.HALLWAY:
                            border_hallways.append((r_idx, c_idx))
                
                if border_hallways:
                    exit_r, exit_c = random.choice(border_hallways)
                    self.grid[exit_r][exit_c] = self.EXIT
                    # print(f"Placed EXIT directly on border hallway tile ({exit_r},{exit_c}) as final fallback.")
                else:
                    print("CRITICAL: No suitable location for EXIT found even with all fallbacks!")

    def _validate_hallway_adjacencies(self):
        """
        Checks if hallway tiles are only adjacent (non-diagonally) to allowed tile types.
        Allowed: HALLWAY, WALL, DOOR, EXIT, PLAYER, OWNER.
        """
        allowed_neighbors = [self.HALLWAY, self.WALL, self.DOOR, self.EXIT, self.PLAYER, self.OWNER]
        all_valid = True
        for r in range(self.height):
            for c in range(self.width):
                if self.grid[r][c] == self.HALLWAY:
                    for dr, dc in self.DIRECTIONS:
                        nr, nc = r + dr, c + dc
                        if self._is_valid(nr, nc):
                            neighbor_tile = self.grid[nr][nc]
                            if neighbor_tile not in allowed_neighbors:
                                print(f"Validation Error: Hallway at ({r},{c}) is adjacent to invalid tile '{neighbor_tile}' at ({nr},{nc}).")
                                all_valid = False
        return all_valid

# --- Example Usage ---
if __name__ == '__main__':
    hotel_gen = HotelGenerator(height=10, width=10) 
    try:
        grid = hotel_gen.generate_hotel(
            straightness_hallways=0.9,   # higher for straigher hallways
            hall_loops=0,                # e.g., 24 loops for 40x30 grid
            max_hallway_perc=0.05,       # proportion of grid to be hallways
            max_rooms=0,                 # number of rooms to place
            room_min_size=2,
            room_max_size=3,
            max_hiding_spots_per_room=1
        )
        hotel_gen.print_grid()
        player_found = any(hotel_gen.PLAYER in row for row in grid)
        owner_found = any(hotel_gen.OWNER in row for row in grid)
        exit_found = any(hotel_gen.EXIT in row for row in grid)
        print(f"Player found: {player_found}, Owner found: {owner_found}, Exit found: {exit_found}")
        print(f"Total rooms with doors: {len([room for room in hotel_gen.rooms if room.get('has_door')])}")
        
    except Exception as e: 
        print(f"An unexpected error occurred: {e}")
        