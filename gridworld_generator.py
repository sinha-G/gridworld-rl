import numpy as np
import random
import math

class DSU:
    """Disjoint Set Union data structure."""
    def __init__(self, n_elements):
        self.parent = list(range(n_elements))
        self.num_sets = n_elements

    def find(self, i):
        """Find the representative (root) of the set containing element i with path compression."""
        if self.parent[i] == i:
            return i # Added return statement
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i, j):
        """Merge the sets containing elements i and j. Returns True if merged, False if already in same set."""
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            self.parent[root_i] = root_j # Standard union by making one root parent of other
            self.num_sets -= 1
            return True
        return False


class GridWorldGenerator:
    """
    Generates gridworlds with rooms and hallways, aiming for hotel/school-like structures with loops.
    """
    TILE_WALL = 0
    TILE_FLOOR = 1
    TILE_DOOR = 2
    TILE_HIDING_SPOT = 3
    WALL_BUFFER_SIZE = 3

    def __init__(self, width=50, height=50, 
                 room_attempts_per_round=50, 
                 max_placement_rounds=3,    
                 room_min_size=3, room_max_size=7, 
                 min_rooms=10,              
                 extra_connection_attempts=5):
        
        # Validate dimensions against room size and buffer
        if width < room_max_size + 2 * self.WALL_BUFFER_SIZE or \
           height < room_max_size + 2 * self.WALL_BUFFER_SIZE:
            raise ValueError(
                f"Grid dimensions ({width}x{height}) are too small for room_max_size ({room_max_size}) "
                f"with WALL_BUFFER_SIZE ({self.WALL_BUFFER_SIZE}).\n"
                f"Need at least {room_max_size + 2 * self.WALL_BUFFER_SIZE} in each dimension."
            )

        self.width = width
        self.height = height
        self.room_attempts_per_round = room_attempts_per_round
        self.max_placement_rounds = max_placement_rounds
        self.room_min_size = room_min_size
        self.room_max_size = room_max_size
        self.min_rooms = min_rooms
        self.extra_connection_attempts = extra_connection_attempts
        
        self.grid = np.full((height, width), self.TILE_WALL, dtype=int)
        self.rooms = [] 
        self.player_start_pos = None
        self.owner_start_pos = None
        self.exit_pos = None

    def generate(self):
        """Generates the gridworld map."""
        self.grid = np.full((self.height, self.width), self.TILE_WALL, dtype=int)
        self.rooms = []
        self.player_start_pos = None
        self.owner_start_pos = None
        self.exit_pos = None

        # Check again in generate, in case __init__ was bypassed or params changed
        if self.width < self.room_max_size + 2 * self.WALL_BUFFER_SIZE or \
           self.height < self.room_max_size + 2 * self.WALL_BUFFER_SIZE:
            print(f"Warning: Grid dimensions too small for room_max_size and WALL_BUFFER_SIZE. No rooms will be placed.")
            return self.grid


        total_attempts_so_far = 0
        for i in range(self.max_placement_rounds):
            self._place_rooms()
            total_attempts_so_far += self.room_attempts_per_round
            if len(self.rooms) >= self.min_rooms:
                break
            # print(f"Round {i+1}: Placed {len(self.rooms)} rooms so far.")
        
        if not self.rooms:
            # print("Warning: No rooms were placed. Check parameters and grid size.")
            # Fallback: if grid is very small but valid, try to place one tiny room if possible
            if self.width >= self.room_min_size + 2 * self.WALL_BUFFER_SIZE and \
               self.height >= self.room_min_size + 2 * self.WALL_BUFFER_SIZE:
                rw = self.room_min_size
                rh = self.room_min_size
                rx = self.WALL_BUFFER_SIZE
                ry = self.WALL_BUFFER_SIZE
                single_room = (rx, ry, rw, rh)
                if self._room_fits(single_room): # Should fit if grid is large enough
                    self._carve_room(single_room)
                    self.rooms.append(single_room)
                    # print("Fallback: Placed one small room.")
            if not self.rooms: # Still no rooms
                 return self.grid # Return empty grid if no rooms could be placed
        
        if len(self.rooms) < 2:
            if self.rooms: 
                self._place_entities()
                self._place_hiding_spots()
            return self.grid

        self._connect_rooms()
        self._place_entities() 
        self._place_hiding_spots() 
        return self.grid

    def _place_rooms(self):
        """Attempts to place rooms on the grid for one round."""
        for _ in range(self.room_attempts_per_round):
            rw = random.randint(self.room_min_size, self.room_max_size)
            rh = random.randint(self.room_min_size, self.room_max_size)

            # Ensure room placement respects WALL_BUFFER_SIZE from grid edges
            # Max rx is such that rx + rw - 1 <= self.width - 1 - WALL_BUFFER_SIZE
            # So, rx <= self.width - rw - WALL_BUFFER_SIZE
            # Min rx is WALL_BUFFER_SIZE
            if self.width - rw - self.WALL_BUFFER_SIZE < self.WALL_BUFFER_SIZE or \
               self.height - rh - self.WALL_BUFFER_SIZE < self.WALL_BUFFER_SIZE:
                continue # Not enough space for this room size with buffers

            rx = random.randint(self.WALL_BUFFER_SIZE, self.width - rw - self.WALL_BUFFER_SIZE -1) # -1 because randint is inclusive for end
            ry = random.randint(self.WALL_BUFFER_SIZE, self.height - rh - self.WALL_BUFFER_SIZE -1) # -1 for inclusivity

            new_room = (rx, ry, rw, rh)
            if self._room_fits(new_room):
                self._carve_room(new_room)
                self.rooms.append(new_room)

    def _room_fits(self, room_to_check):
        """
        Checks if a new room, respecting WALL_BUFFER_SIZE, overlaps with existing rooms
        or is too close to grid edges (already handled by rx, ry generation in _place_rooms).
        """
        rx_c, ry_c, rw_c, rh_c = room_to_check

        # Check against grid boundaries (already implicitly handled by rx,ry generation, but good for safety)
        if rx_c < self.WALL_BUFFER_SIZE or \
           ry_c < self.WALL_BUFFER_SIZE or \
           rx_c + rw_c > self.width - self.WALL_BUFFER_SIZE or \
           ry_c + rh_c > self.height - self.WALL_BUFFER_SIZE:
            return False # Should not happen if rx,ry are generated correctly

        # Check against existing rooms for overlap, including the WALL_BUFFER_SIZE
        # A room (x,y,w,h) occupies floor tiles from x to x+w-1 and y to y+h-1.
        # The forbidden zone around an existing room (erx, ery, erw, erh) for a new room is:
        # x: [erx - WALL_BUFFER_SIZE, erx + erw - 1 + WALL_BUFFER_SIZE]
        # y: [ery - WALL_BUFFER_SIZE, ery + erh - 1 + WALL_BUFFER_SIZE]
        
        # Coordinates for the room_to_check's actual floor space
        x1_check = rx_c
        y1_check = ry_c
        x2_check = rx_c + rw_c - 1 
        y2_check = ry_c + rh_c - 1

        for erx, ery, erw, erh in self.rooms:
            # Define the forbidden zone for room_to_check based on the existing room + buffer
            x1_existing_buffered = erx - self.WALL_BUFFER_SIZE
            y1_existing_buffered = ery - self.WALL_BUFFER_SIZE
            x2_existing_buffered = erx + erw - 1 + self.WALL_BUFFER_SIZE 
            y2_existing_buffered = ery + erh - 1 + self.WALL_BUFFER_SIZE

            # Standard AABB overlap test:
            # Overlap exists if (RectA.X1 < RectB.X2 && RectA.X2 > RectB.X1 &&
            #                    RectA.Y1 < RectB.Y2 && RectA.Y2 > RectB.Y1)
            overlap_x = (x1_check <= x2_existing_buffered and x2_check >= x1_existing_buffered)
            overlap_y = (y1_check <= y2_existing_buffered and y2_check >= y1_existing_buffered)

            if overlap_x and overlap_y:
                return False # New room overlaps with the buffered zone of an existing room
        return True

    def _carve_room(self, room):
        """Sets tiles for a room to TILE_FLOOR."""
        rx, ry, rw, rh = room
        self.grid[ry : ry + rh, rx : rx + rw] = self.TILE_FLOOR

    def _get_room_centers(self):
        """Calculates the center coordinates of all placed rooms."""
        centers = []
        for i, (x, y, w, h) in enumerate(self.rooms):
            centers.append((x + w // 2, y + h // 2, i)) # Store room index
        return centers

    def _connect_rooms(self):
        """Connects rooms using MST and adds extra connections for loops."""
        if len(self.rooms) < 2:
            return

        room_centers_with_indices = self._get_room_centers()
        num_rooms = len(self.rooms)
        dsu = DSU(num_rooms)

        edges = []
        for i in range(num_rooms):
            for j in range(i + 1, num_rooms):
                c1_x, c1_y, _ = room_centers_with_indices[i]
                c2_x, c2_y, _ = room_centers_with_indices[j]
                dist = math.sqrt((c1_x - c2_x)**2 + (c1_y - c2_y)**2)
                edges.append((dist, i, j)) # (distance, room_idx1, room_idx2)
        
        edges.sort() # Sort by distance for Kruskal's

        # Connect rooms using MST (Kruskal's algorithm)
        mst_edges_count = 0
        for dist, u_idx, v_idx in edges:
            if dsu.union(u_idx, v_idx):
                center1 = (room_centers_with_indices[u_idx][0], room_centers_with_indices[u_idx][1])
                center2 = (room_centers_with_indices[v_idx][0], room_centers_with_indices[v_idx][1])
                self._carve_corridor(center1, center2, u_idx, v_idx) 
                mst_edges_count += 1
                if dsu.num_sets == 1: # All rooms are connected
                    break
        
        # Add extra connections to create loops
        added_loops = 0
        loop_attempts = 0
        max_loop_gen_attempts = self.extra_connection_attempts * 5 

        while added_loops < self.extra_connection_attempts and loop_attempts < max_loop_gen_attempts:
            if num_rooms < 2: break 
            
            idx1, idx2 = random.sample(range(num_rooms), 2) 
            
            center1 = (room_centers_with_indices[idx1][0], room_centers_with_indices[idx1][1])
            center2 = (room_centers_with_indices[idx2][0], room_centers_with_indices[idx2][1])
            
            self._carve_corridor(center1, center2, idx1, idx2) 
            added_loops += 1
            loop_attempts += 1
        
    def _get_room_index_at_coord(self, tile_x, tile_y):
        """
        Returns the index of the room that geometrically contains the given tile coordinates.
        Checks against defined room boundaries (self.rooms).
        """
        for i, (r_x, r_y, r_w, r_h) in enumerate(self.rooms):
            if r_x <= tile_x < r_x + r_w and r_y <= tile_y < r_y + r_h:
                return i
        return None

    def _carve_corridor(self, c1, c2, room1_idx, room2_idx):
        """
        Carves an L-shaped corridor between two points, placing doors where corridors meet
        the specified room1 or room2.
        """
        x1, y1 = c1
        x2, y2 = c2

        # Helper to carve a single tile, deciding if it's a door or floor
        def carve_tile(px, py):
            if not (0 <= py < self.height and 0 <= px < self.width):
                return

            current_tile_type = self.grid[py, px]

            if current_tile_type == self.TILE_WALL:
                is_valid_door_location_for_target_rooms = False
                # Check cardinal neighbors for floor of room1 or room2
                for dx_n, dy_n in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    adj_x, adj_y = px + dx_n, py + dy_n
                    if 0 <= adj_y < self.height and 0 <= adj_x < self.width and \
                       self.grid[adj_y, adj_x] == self.TILE_FLOOR:
                        
                        adj_room_idx = self._get_room_index_at_coord(adj_x, adj_y)
                        if adj_room_idx is not None and \
                           (adj_room_idx == room1_idx or adj_room_idx == room2_idx):
                            is_valid_door_location_for_target_rooms = True
                            break 
                
                if is_valid_door_location_for_target_rooms:
                    # Check cardinal neighbors for existing doors to prevent DD, D D, etc.
                    #                                                               D
                    place_as_floor_due_to_adj_door = False
                    for dx_adj, dy_adj in [(0,1), (0,-1), (1,0), (-1,0)]:
                        near_x, near_y = px + dx_adj, py + dy_adj
                        if 0 <= near_y < self.height and 0 <= near_x < self.width and \
                           self.grid[near_y, near_x] == self.TILE_DOOR:
                            place_as_floor_due_to_adj_door = True
                            break
                    
                    if place_as_floor_due_to_adj_door:
                        self.grid[py, px] = self.TILE_FLOOR
                    else:
                        self.grid[py, px] = self.TILE_DOOR
                else: # Not a door candidate for target rooms (e.g. adjacent to unrelated room, or no room)
                    self.grid[py, px] = self.TILE_FLOOR
            # If it's already a door or floor (e.g. from another corridor or room carving), leave it.
            elif current_tile_type == self.TILE_DOOR or current_tile_type == self.TILE_FLOOR:
                pass
            # Potentially handle other existing tile types if necessary

        # Carve L-shaped path
        if random.random() < 0.5: # Horizontal then Vertical
            # Horizontal segment from (x1,y1) to (x2,y1)
            for x_coord in range(min(x1, x2), max(x1, x2) + 1):
                carve_tile(x_coord, y1)
            # Vertical segment from (x2,y1) to (x2,y2)
            for y_coord in range(min(y1, y2), max(y1, y2) + 1):
                carve_tile(x2, y_coord)
        else: # Vertical then Horizontal
            # Vertical segment from (x1,y1) to (x1,y2)
            for y_coord in range(min(y1, y2), max(y1, y2) + 1):
                carve_tile(x1, y_coord)
            # Horizontal segment from (x1,y2) to (x2,y2)
            for x_coord in range(min(x1, x2), max(x1, x2) + 1):
                carve_tile(x_coord, y2)

    def _get_random_floor_in_room(self, room_details, exclude_pos_list=None):
        """Gets a random TILE_FLOOR coordinate within a given room, optionally excluding positions."""
        if exclude_pos_list is None:
            exclude_pos_list = []
        
        rx, ry, rw, rh = room_details
        possible_locations = []
        for y_offset in range(rh):
            for x_offset in range(rw):
                px, py = rx + x_offset, ry + y_offset
                if self.grid[py, px] == self.TILE_FLOOR and (px, py) not in exclude_pos_list:
                    possible_locations.append((px, py))
        
        if possible_locations:
            return random.choice(possible_locations)
        else: # Fallback if no suitable floor tile found (e.g., all excluded or room has no floor)
            # Try again without exclusion if exclusion list was the cause
            if exclude_pos_list:
                return self._get_random_floor_in_room(room_details, exclude_pos_list=[])
            # If still no floor, this is an issue, return a placeholder (e.g. room center)
            # print(f"Warning: Could not find a suitable TILE_FLOOR in room {room_details} for entity placement.")
            return (rx + rw // 2, ry + rh // 2)

    def _is_pos_in_room(self, pos, room_details):
        """Checks if a position (x,y) is within the floor area of a room."""
        if pos is None: return False
        px, py = pos
        rx, ry, rw, rh = room_details
        return rx <= px < rx + rw and ry <= py < ry + rh

    def _place_entities(self):
        """Places player, owner, and exit in random rooms."""
        if not self.rooms:
            # print("Warning: No rooms to place entities.")
            return

        shuffled_rooms = list(self.rooms)
        random.shuffle(shuffled_rooms)

        # Place Player
        player_room = shuffled_rooms[0]
        self.player_start_pos = self._get_random_floor_in_room(player_room)

        # Place Owner
        if len(shuffled_rooms) > 1:
            owner_room = shuffled_rooms[1]
            # Ensure owner is not in the exact same spot as player if in the same room by chance (unlikely with distinct rooms)
            self.owner_start_pos = self._get_random_floor_in_room(owner_room, 
                                                                  exclude_pos_list=[self.player_start_pos] if owner_room == player_room else [])
        else: # Only one room
            self.owner_start_pos = self._get_random_floor_in_room(player_room, exclude_pos_list=[self.player_start_pos])

        # Place Exit
        # Try to place exit in a third room if available, otherwise pick any room.
        if len(shuffled_rooms) > 2:
            exit_room = shuffled_rooms[2]
        else: # Pick any room, could be player's or owner's
            exit_room = random.choice(self.rooms)

        exclude_for_exit = []
        if self.player_start_pos and self._is_pos_in_room(self.player_start_pos, exit_room):
            exclude_for_exit.append(self.player_start_pos)
        if self.owner_start_pos and self._is_pos_in_room(self.owner_start_pos, exit_room):
            exclude_for_exit.append(self.owner_start_pos)
        self.exit_pos = self._get_random_floor_in_room(exit_room, exclude_pos_list=exclude_for_exit)

    def _place_hiding_spots(self):
        """Places 0-2 hiding spots in each room."""
        if not self.rooms:
            return

        for room_details in self.rooms:
            rx, ry, rw, rh = room_details
            num_hiding_spots_to_place = random.randint(0, 2)
            placed_spots = 0
            
            possible_hiding_locations = []
            for y_offset in range(rh):
                for x_offset in range(rw):
                    tile_x, tile_y = rx + x_offset, ry + y_offset
                    pos = (tile_x, tile_y)

                    # Must be an original floor tile
                    if self.grid[tile_y, tile_x] == self.TILE_FLOOR:
                        # Cannot be where an entity is
                        if pos == self.player_start_pos or \
                           pos == self.owner_start_pos or \
                           pos == self.exit_pos:
                            continue

                        # Must not be adjacent to a door (to avoid blocking)
                        is_clear_of_doors = True
                        for dy_n, dx_n in [(0,1), (0,-1), (1,0), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]: # Check 8 neighbors
                            adj_x, adj_y = tile_x + dx_n, tile_y + dy_n
                            if 0 <= adj_y < self.height and 0 <= adj_x < self.width and \
                               self.grid[adj_y, adj_x] == self.TILE_DOOR:
                                is_clear_of_doors = False
                                break
                        
                        if is_clear_of_doors:
                            possible_hiding_locations.append(pos)
            
            random.shuffle(possible_hiding_locations)
            
            for spot_x, spot_y in possible_hiding_locations:
                if placed_spots < num_hiding_spots_to_place:
                    self.grid[spot_y, spot_x] = self.TILE_HIDING_SPOT
                    placed_spots += 1
                else:
                    break

def print_grid(grid_array, player_pos=None, owner_pos=None, exit_pos=None):
    """ASCII-print the grid, optionally showing entity positions."""
    chars = {
        GridWorldGenerator.TILE_WALL: "#", 
        GridWorldGenerator.TILE_FLOOR: ".", 
        GridWorldGenerator.TILE_DOOR: "D",
        GridWorldGenerator.TILE_HIDING_SPOT: "H"
    }
    
    # Create a character grid from the numerical grid
    char_grid = []
    for r_idx, row in enumerate(grid_array):
        char_row = []
        for c_idx, val in enumerate(row):
            char_row.append(chars.get(val, "?"))
        char_grid.append(char_row)

    # Overlay entities if positions are provided
    if player_pos and 0 <= player_pos[1] < len(char_grid) and 0 <= player_pos[0] < len(char_grid[0]):
        char_grid[player_pos[1]][player_pos[0]] = "P"
    if owner_pos and 0 <= owner_pos[1] < len(char_grid) and 0 <= owner_pos[0] < len(char_grid[0]):
        char_grid[owner_pos[1]][owner_pos[0]] = "O"
    if exit_pos and 0 <= exit_pos[1] < len(char_grid) and 0 <= exit_pos[0] < len(char_grid[0]):
        char_grid[exit_pos[1]][exit_pos[0]] = "E"

    for row in char_grid:
        print("".join(row))


if __name__ == "__main__":
    print("Generating a gridworld with doors, hiding spots, player, owner, and exit:")
    try:
        generator = GridWorldGenerator(
            width=70, 
            height=40, 
            room_attempts_per_round=200, # Increased attempts due to stricter placement
            max_placement_rounds=5,    # Potentially more rounds needed
            room_min_size=4, 
            room_max_size=6, # Max size might need to be smaller relative to grid
            min_rooms=8, # Might achieve fewer rooms with buffer                
            extra_connection_attempts=10
        )
        
        grid_map = generator.generate()
        if grid_map is not None and generator.rooms: # Check if generation was successful
            print_grid(grid_map, generator.player_start_pos, generator.owner_start_pos, generator.exit_pos)
            print(f"Number of rooms successfully placed: {len(generator.rooms)}")
            print(f"Grid dimensions: {generator.width}x{generator.height}")
            print(f"Player start: {generator.player_start_pos}")
            print(f"Owner start: {generator.owner_start_pos}")
            print(f"Exit position: {generator.exit_pos}")
        else:
            print("Failed to generate a valid grid or place rooms.")

    except ValueError as e:
        print(f"Error: {e}")