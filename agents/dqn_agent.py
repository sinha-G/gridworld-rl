import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
from collections import deque, namedtuple
import os # For saving/loading models

# Define the Experience tuple for the Replay Buffer
Experience = namedtuple('Experience', 
                        ('state_tensor', 'action_index', 'reward', 'next_state_tensor', 'done'))

class DQNNetwork(nn.Module):
    """
    Convolutional Neural Network for Q-value approximation.
    """
    def __init__(self, input_channels, height, width, num_actions, dropout_rate=0.2):
        super(DQNNetwork, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=5, stride=1, padding=2) # Maintains H, W
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1) # Maintains H, W
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1) # Maintains H, W
        self.bn3 = nn.BatchNorm2d(64)
        
        # Calculate the size of the flattened features after conv layers
        # Assuming padding maintains dimensions and stride is 1
        conv_out_h = height 
        conv_out_w = width
        linear_input_size = 64 * conv_out_h * conv_out_w 

        # Fully connected layers
        self.fc1 = nn.Linear(linear_input_size, 256) # Increased hidden units
        self.dropout = nn.Dropout(dropout_rate) # Added dropout layer
        
        # Dueling DQN: Value stream
        self.fc_value = nn.Linear(256, 1)
        # Dueling DQN: Advantage stream
        self.fc_advantage = nn.Linear(256, num_actions)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1) # Flatten the output of conv layers
        x_fc1 = F.relu(self.fc1(x))
        x_fc1_dropped = self.dropout(x_fc1) # Apply dropout

        state_value = self.fc_value(x_fc1)
        action_advantage = self.fc_advantage(x_fc1)

        # Combine value and advantage streams
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
        q_values = state_value + (action_advantage - action_advantage.mean(dim=1, keepdim=True))
        return q_values


class DQNAgent:
    def __init__(self, 
                 game_actions_list, # List of game action constants [Game.MOVE_NORTH, ...]
                 input_channels, 
                 grid_height, 
                 grid_width,
                 learning_rate=0.0005, # Adjusted learning rate
                 discount_factor=0.99, # Gamma
                 exploration_rate_initial=1.0, # Epsilon
                 exploration_decay_rate=0.9995, 
                 min_exploration_rate=0.01, # Lower min epsilon
                 replay_buffer_size=50000, # Increased buffer size
                 batch_size=64, 
                 target_update_frequency=1000): # Update target less frequently initially
        
        self.game_actions_list = game_actions_list # e.g., [Game.MOVE_NORTH, Game.MOVE_SOUTH, ...]
        self.num_actions = len(game_actions_list)
        self.gamma = discount_factor
        self.epsilon = exploration_rate_initial
        self.epsilon_decay = exploration_decay_rate
        self.epsilon_min = min_exploration_rate
        self.batch_size = batch_size
        self.target_update_frequency = target_update_frequency
        self.learn_step_counter = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"DQNAgent using device: {self.device}")

        self.q_network = DQNNetwork(input_channels, grid_height, grid_width, self.num_actions).to(self.device)
        self.target_network = DQNNetwork(input_channels, grid_height, grid_width, self.num_actions).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval() # Target network is not trained directly

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.replay_buffer = deque(maxlen=replay_buffer_size)

    def get_action_index(self, game_action_constant):
        """Helper to get index from a game action constant."""
        try:
            return self.game_actions_list.index(game_action_constant)
        except ValueError:
            return -1 # Should not happen if action is valid

    def choose_action(self, state_tensor):
        """
        Chooses an action using epsilon-greedy policy from all possible game actions.
        state_tensor: PyTorch tensor (1, C, H, W) representing the current grid state.
        """
        if random.random() < self.epsilon:
            # Exploration: Choose a random valid game action
            return random.choice(self.game_actions_list)
        else:
            # Exploitation: Choose the best valid action based on Q-network
            with torch.no_grad():
                # state_tensor should be (1, C, H, W)
                q_values_all_actions = self.q_network(state_tensor.to(self.device))[0] 

            # Find the action_index corresponding to the max Q-value
            best_action_idx = torch.argmax(q_values_all_actions).item()
            return self.game_actions_list[best_action_idx]

    def store_transition(self, state_tensor, action_constant, reward, next_state_tensor, done):
        """
        Stores an experience in the replay buffer.
        state_tensor: PyTorch tensor for current state.
        action_constant: The game action constant (e.g., Game.MOVE_NORTH).
        reward: Float.
        next_state_tensor: PyTorch tensor for next state. Can be None if terminal.
        done: Boolean.
        """
        action_idx = self.get_action_index(action_constant)
        if action_idx == -1:
            print(f"Warning: Action constant {action_constant} not found in agent's list.")
            return 
        # Ensure tensors are detached from graph and on CPU for storage if they came from GPU
        # Or ensure they are in a format that can be easily batched later.
        # Storing them as is, assuming they are already processed, is fine for now.
        self.replay_buffer.append(Experience(state_tensor.cpu(), 
                                             action_idx, 
                                             reward, 
                                             next_state_tensor.cpu() if next_state_tensor is not None else None, 
                                             done))

    def learn(self):
        """Samples from replay buffer and performs a Q-network update."""
        if len(self.replay_buffer) < self.batch_size:
            return # Not enough experiences to learn

        experiences = random.sample(self.replay_buffer, self.batch_size)
        batch = Experience(*zip(*experiences))

        # Convert batch components to tensors on the correct device
        # States and next_states are already tensors, just need to stack and move to device
        state_batch = torch.cat([s.unsqueeze(0) for s in batch.state_tensor]).to(self.device)
        action_batch = torch.tensor(batch.action_index, dtype=torch.long).unsqueeze(1).to(self.device)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32).unsqueeze(1).to(self.device)
        done_batch = torch.tensor(batch.done, dtype=torch.float32).unsqueeze(1).to(self.device)

        # Handle non-final next states
        non_final_mask = torch.tensor(tuple(s is not None for s in batch.next_state_tensor), dtype=torch.bool).to(self.device)
        
        non_final_next_states_list = [s.unsqueeze(0) for s in batch.next_state_tensor if s is not None]
        if not non_final_next_states_list:
            # All next states in this batch are terminal
            # Create an empty tensor with the correct shape for next_state_q_values calculation
            # This case might need careful handling if it occurs often with small batches
            # For now, we proceed, and next_state_q_values will remain zeros for these.
            next_state_q_values = torch.zeros(self.batch_size, device=self.device)
        else:
            non_final_next_states = torch.cat(non_final_next_states_list).to(self.device)
            # V(s_{t+1}) = max_a Q_target(s_{t+1}, a)
            next_state_q_values_all_actions = torch.zeros(self.batch_size, device=self.device) # Initialize for all
            with torch.no_grad():
                 # Get Q values from target network only for non_final_next_states
                next_state_q_values_all_actions[non_final_mask] = self.target_network(non_final_next_states).max(1)[0]
            next_state_q_values = next_state_q_values_all_actions


        # Expected Q values: R + gamma * V(s_{t+1}) * (1 - done)
        expected_q_values = reward_batch + (self.gamma * next_state_q_values.unsqueeze(1) * (1 - done_batch))

        # Q(s_t, a_t) from the Q-network
        current_q_values = self.q_network(state_batch).gather(1, action_batch)
        
        # Loss
        loss = F.mse_loss(current_q_values, expected_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0) # Gradient clipping
        self.optimizer.step()

        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_update_frequency == 0:
            self._update_target_network()

    def _update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_exploration(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save_model(self, filepath="dqn_model.pth"):
        """Saves the Q-network's state dictionary."""
        # Ensure directory exists if filepath includes directories
        dir_name = os.path.dirname(filepath)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
            
        torch.save(self.q_network.state_dict(), filepath)
        # print(f"DQN model saved to {filepath}")

    def load_model(self, filepath="dqn_model.pth"):
        """Loads the Q-network's state dictionary."""
        if os.path.exists(filepath):
            try:
                self.q_network.load_state_dict(torch.load(filepath, map_location=self.device))
                self._update_target_network() # Sync target network
                self.q_network.eval() # Set to evaluation mode if loaded for inference
                print(f"DQN model loaded from {filepath}")
            except Exception as e:
                print(f"Error loading DQN model from {filepath}: {e}. Starting fresh.")
        else:
            print(f"No DQN model found at {filepath}. Starting fresh.")
