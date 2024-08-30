import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from gym import Env
from typing import Any
from gym.spaces import Discrete, Box
from time import sleep
import time
import uuid

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Output layer for the means
        self.fc_mean = nn.Linear(hidden_dim, action_dim)
        
        # Optional: Output layer for the log standard deviations
        # This gives each action dimension its own log std dev
        self.fc_log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        
        # Means for the Gaussian distribution
        means = self.fc_mean(x)
        
        # Log std deviations for the Gaussian distribution
        log_stds = self.fc_log_std(x)
        
        # Return both means and log stds
        return means, log_stds


class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim=128):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class PPO:
    def __init__(self, env: Env, policy, value, lr=1e-3, gamma=0.99, eps_clip=0.2, K_epochs=4):
        self.env = env
        self.policy = policy
        self.value = value
        self.optimizer = optim.Adam(list(policy.parameters()) + list(value.parameters()), lr=lr)
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        means, log_stds = self.policy(state)
        
        stds = torch.exp(log_stds)
        
        dist = torch.distributions.Normal(means, stds)
        sampled_action = dist.sample()
        log_prob = dist.log_prob(sampled_action).sum(dim=1)
        
        return sampled_action.squeeze(0).detach().numpy(), log_prob.squeeze(0).detach()

    def compute_returns(self, rewards, dones, next_value):
        returns = []
        R = next_value
        for step in reversed(range(len(rewards))):
            R = rewards[step] + self.gamma * R * (1 - dones[step])
            returns.insert(0, R)
        return returns

    def optimize(self, states, actions, log_probs, returns):
        for _ in range(self.K_epochs):
            states = torch.tensor(states, dtype=torch.float32)
            actions = torch.tensor(actions, dtype=torch.float32)
            log_probs = torch.tensor(log_probs, dtype=torch.float32)
            returns = torch.tensor(returns, dtype=torch.float32)

            means, log_stds = self.policy(states)
            
            # Clamping log standard deviations
            log_stds = torch.clamp(log_stds, min=-20, max=2)
            
            stds = torch.exp(log_stds)

            # Check for NaNs before creating the distribution
            if torch.isnan(means).any() or torch.isnan(stds).any():
                continue  # Skip this iteration if NaNs are found

            dist = torch.distributions.Normal(means, stds)
            new_log_probs = dist.log_prob(actions).sum(dim=1)
            ratios = torch.exp(new_log_probs - log_probs)
            state_values = self.value(states).squeeze()
            advantages = returns - state_values.detach()

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2).mean() + 0.5 * nn.MSELoss()(state_values, returns)

            self.optimizer.zero_grad()
            loss.backward()
            
            # Apply gradient clipping
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
            
            self.optimizer.step()

    def train(self, max_episodes=1000):
        for episode in range(max_episodes):
            if (episode+1) % 10 == 0:
                self.env.render()
            print(self.env.list_clients())

            state = self.env.reset()  # Start a new episode

            # Select an action for the initial state
            action, log_prob = self.select_action(state)

            # Perform the action and get the reward
            next_state, reward, done, _ = self.env.step(action)

            # Store the state, action, log_prob, and reward
            states = [state]
            actions = [action]
            log_probs = [log_prob]
            episode_rewards = [reward]

            # Since there's only one step, the return is just the reward
            returns = [reward]

            # Optimize the policy using this single-step experience
            self.optimize(states, actions, log_probs, returns)

            total_reward = sum(episode_rewards)
            print(f"Episode {episode+1}/{max_episodes}, Total Reward: {total_reward}")

class Environment(Env):
    def __init__(self, client_pool: Any, param_ranges: np.ndarray):
        super(Environment, self).__init__()

        self.param_ranges = param_ranges
        self.action_space = Box(low=param_ranges[:, 0], high=param_ranges[:, 1], dtype=np.float32)
        print(f"Action space: {self.action_space}")
        self.observation_space = Box(low=param_ranges[:, 0], high=param_ranges[:, 1], dtype=np.float32)
        self.state = np.random.uniform(param_ranges[:, 0], param_ranges[:, 1])
        print(self.state)
        self.client = client_pool
        self.result_buffer = {}
    
    def step(self, action):
        print(f"Taking action: {action}")
        self.state = np.clip(action, self.param_ranges[:, 0], self.param_ranges[:, 1])
        
        rollout_id = str(uuid.uuid4())  # Generate a unique ID for this rollout
        rollout = {
            "id": rollout_id,
            "parameters": self.state
        }

        self.client.distribute_tasks([rollout])

        # Wait for results, with a timeout
        start_time = time.time()
        results_received = False
        while not results_received and time.time() - start_time <= 5:
          client_results = self.client.gather_results()
          for result in client_results:
              result_data, result_reward = result  # Unpack the tuple
              if result_data["id"] == rollout_id:
                
                self.result_buffer[result_data["id"]] = result
                results_received = True
                

        # Get the results for the current rollout
        print(f"Results received: {self.result_buffer}")
        if rollout_id in self.result_buffer:
            reward = self.result_buffer[rollout_id][1]
            self.result_buffer.pop(rollout_id)
        else:
            reward = 0  # Or some other default value

        done = True
        return self.state, reward, done, {}

    
    def reset(self):
        self.state = np.random.uniform(self.param_ranges[:, 0], self.param_ranges[:, 1])
        return self.state
    
    

    def render(self, mode='human'):
        print(f"Current parameters: {self.state}")
    
    def close(self):
        pass

    def list_clients(self):
        return self.client.clients
    