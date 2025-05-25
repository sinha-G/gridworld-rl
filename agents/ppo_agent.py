import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import os

class ActorCriticNetwork(nn.Module):
    def __init__(self, input_channels, height, width, num_actions, dropout_rate=0.1):
        super(ActorCriticNetwork, self).__init__()
        # Convolutional layers (can be similar to your DQN network)
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        conv_out_h = height
        conv_out_w = width
        linear_input_size = 64 * conv_out_h * conv_out_w

        self.fc_shared = nn.Linear(linear_input_size, 512)
        # self.dropout = nn.Dropout(dropout_rate)

        # Actor head
        self.actor_fc = nn.Linear(512, num_actions)
        # Critic head
        self.critic_fc = nn.Linear(512, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1) # Flatten
        
        x_shared = F.relu(self.fc_shared(x))
        # x_shared_dropped = self.dropout(x_shared)

        action_probs = F.softmax(self.actor_fc(x_shared), dim=-1)
        state_values = self.critic_fc(x_shared)
        
        return action_probs, state_values

class PPOAgent:
    def __init__(self,
                 game_actions_list,
                 input_channels,
                 grid_height,
                 grid_width,
                 learning_rate=0.0001,
                 gamma=0.99, # Discount factor
                 gae_lambda=0.95, # Lambda for GAE
                 ppo_clip_epsilon=0.2, # Epsilon for PPO clipping
                 ppo_epochs=10, # Number of epochs for PPO update
                 mini_batch_size=64,
                 entropy_coefficient=0.01,
                 value_loss_coefficient=0.5):
        
        self.game_actions_list = game_actions_list
        self.num_actions = len(game_actions_list)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ppo_clip_epsilon = ppo_clip_epsilon
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self.entropy_coefficient = entropy_coefficient
        self.value_loss_coefficient = value_loss_coefficient

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"PPOAgent using device: {self.device}")

        self.network = ActorCriticNetwork(input_channels, grid_height, grid_width, self.num_actions).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)

        # Temporary storage for an episode or a batch of transitions
        self.memory = {
            "states": [], "actions": [], "log_probs": [],
            "rewards": [], "next_states": [], "dones": [], "values": []
        }
        self.action_map = {action_const: i for i, action_const in enumerate(self.game_actions_list)}
        self.index_to_action_map = {i: action_const for i, action_const in enumerate(self.game_actions_list)}


    def get_action_index(self, game_action_constant):
        return self.action_map.get(game_action_constant, -1)

    def get_action_constant(self, action_index):
        return self.index_to_action_map.get(action_index)

    def choose_action(self, state_tensor):
        """
        Chooses an action based on the current policy.
        state_tensor: PyTorch tensor (1, C, H, W)
        Returns: game_action_constant, action_log_prob, state_value
        """
        state_tensor = state_tensor.to(self.device)
        with torch.no_grad():
            action_probs, state_value = self.network(state_tensor)
        
        dist = Categorical(action_probs)
        action_index = dist.sample()
        action_log_prob = dist.log_prob(action_index)
        
        game_action_constant = self.get_action_constant(action_index.item())
        return game_action_constant, action_log_prob.cpu(), state_value.cpu()

    def store_transition(self, state, action_constant, log_prob, value, reward, next_state, done):
        self.memory["states"].append(state.cpu())
        action_idx = self.get_action_index(action_constant)
        self.memory["actions"].append(torch.tensor(action_idx, dtype=torch.long))
        self.memory["log_probs"].append(log_prob)
        self.memory["values"].append(value)
        self.memory["rewards"].append(torch.tensor(reward, dtype=torch.float32))
        self.memory["next_states"].append(next_state.cpu() if next_state is not None else torch.zeros_like(state.cpu())) # Store zero tensor if None
        self.memory["dones"].append(torch.tensor(done, dtype=torch.float32))

    def clear_memory(self):
        for key in self.memory:
            self.memory[key].clear()

    def learn(self):
        if not self.memory["states"]:
            return 0, 0, 0 # No data to learn from

        # Convert lists to tensors
        states = torch.stack(self.memory["states"]).to(self.device)
        actions = torch.stack(self.memory["actions"]).to(self.device).squeeze()
        old_log_probs = torch.stack(self.memory["log_probs"]).to(self.device).squeeze().detach()
        rewards = torch.stack(self.memory["rewards"]).to(self.device).squeeze()
        # next_states = torch.stack(self.memory["next_states"]).to(self.device) # Not directly used in GAE like this
        dones = torch.stack(self.memory["dones"]).to(self.device).squeeze()
        old_values = torch.stack(self.memory["values"]).to(self.device).squeeze().detach()

        # Calculate GAE and Returns
        advantages = torch.zeros_like(rewards).to(self.device)
        returns = torch.zeros_like(rewards).to(self.device)
        gae = 0
        
        # Estimate value of the last next_state if not done
        last_state_for_value = self.memory["next_states"][-1].unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, last_value = self.network(last_state_for_value)
        last_value = last_value.squeeze()

        for t in reversed(range(len(rewards))):
            if dones[t]:
                delta = rewards[t] - old_values[t]
                gae = 0 # Reset GAE at terminal state
            else:
                next_val = old_values[t+1] if t < len(rewards) - 1 else last_value # Value of S_t+1
                delta = rewards[t] + self.gamma * next_val - old_values[t]
            
            gae = delta + self.gamma * self.gae_lambda * (1.0 - dones[t]) * gae # dones[t] ensures gae is 0 if current state was terminal
            advantages[t] = gae
            returns[t] = advantages[t] + old_values[t] # Q_target = Advantage + V(s_t)

        # Normalize advantages (optional but often helpful)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Normalize returns (optional, but can help with stability)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8) 

        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0

        num_samples = len(states)
        indices = np.arange(num_samples)

        for _ in range(self.ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, num_samples, self.mini_batch_size):
                end = start + self.mini_batch_size
                batch_indices = indices[start:end]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                # Get new action_probs, values, and entropy from the network
                action_probs, current_values = self.network(batch_states)
                current_values = current_values.squeeze()
                
                dist = Categorical(action_probs)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                # Policy (Actor) Loss - PPO Clipped Objective
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.ppo_clip_epsilon, 1.0 + self.ppo_clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value (Critic) Loss
                value_loss = F.mse_loss(current_values, batch_returns)
                
                # Total Loss
                loss = policy_loss + self.value_loss_coefficient * value_loss - self.entropy_coefficient * entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5) # Gradient clipping
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy.item()
        
        avg_policy_loss = total_policy_loss / (self.ppo_epochs * (num_samples // self.mini_batch_size))
        avg_value_loss = total_value_loss / (self.ppo_epochs * (num_samples // self.mini_batch_size))
        avg_entropy_loss = total_entropy_loss / (self.ppo_epochs * (num_samples // self.mini_batch_size))

        self.clear_memory() # Clear memory after update
        return avg_policy_loss, avg_value_loss, avg_entropy_loss

    def save_model(self, filepath="ppo_model.pth"):
        dir_name = os.path.dirname(filepath)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
        torch.save(self.network.state_dict(), filepath)

    def load_model(self, filepath="ppo_model.pth"):
        if os.path.exists(filepath):
            try:
                self.network.load_state_dict(torch.load(filepath, map_location=self.device))
                self.network.eval() # Set to evaluation mode
                print(f"PPO model loaded from {filepath}")
            except Exception as e:
                print(f"Error loading PPO model from {filepath}: {e}. Starting fresh.")
        else:
            print(f"No PPO model found at {filepath}. Starting fresh.")
    
    # PPO doesn't use epsilon-greedy exploration in the same way as DQN.
    # Exploration is driven by the stochastic policy and entropy bonus.
    # So, decay_exploration is not typically needed for PPO.
    def decay_exploration(self):
        pass # Not used in this PPO setup
