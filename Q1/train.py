import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
from collections import deque, namedtuple
import os
import time

# Set random seeds for reproducibility
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Experience replay buffer
class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
    
    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        
        return (states, actions, rewards, next_states, dones)
    
    def __len__(self):
        return len(self.memory)

# Actor (Policy) Network
class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=256, init_w=3e-3):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        
        # Initialize final layer weights
        self.fc3.weight.data.uniform_(-init_w, init_w)
        self.fc3.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        # Apply tanh to bound output to [-1, 1]
        # Then scale to action space range (-2.0, 2.0)
        return torch.tanh(self.fc3(x)) * 2.0  # Scale to Pendulum's action range

# Critic (Q-Value) Network
class Critic(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=256, init_w=3e-3):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size + action_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        
        # Initialize final layer weights
        self.fc3.weight.data.uniform_(-init_w, init_w)
        self.fc3.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state, action):
        xs = F.relu(self.fc1(state))
        x = torch.cat([xs, action], dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Ornstein-Uhlenbeck noise for exploration
class OUNoise:
    def __init__(self, size, mu=0.0, theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.size = size
        self.reset()
        
    def reset(self):
        self.state = self.mu.copy()
        
    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx
        return self.state

# DDPG Agent
class DDPGAgent:
    def __init__(self, state_size, action_size, hidden_size=256, buffer_size=1000000, batch_size=64,
                 gamma=0.99, tau=1e-3, actor_lr=1e-4, critic_lr=1e-3):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma  # discount factor
        self.tau = tau  # soft update parameter
        
        # Actor Networks (local and target)
        self.actor_local = Actor(state_size, action_size, hidden_size).to(device)
        self.actor_target = Actor(state_size, action_size, hidden_size).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=actor_lr)
        
        # Critic Networks (local and target)
        self.critic_local = Critic(state_size, action_size, hidden_size).to(device)
        self.critic_target = Critic(state_size, action_size, hidden_size).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=critic_lr, weight_decay=1e-2)
        
        # Initialize target networks with local network weights
        self.soft_update(self.actor_local, self.actor_target, 1.0)
        self.soft_update(self.critic_local, self.critic_target, 1.0)
        
        # Replay buffer
        self.memory = ReplayBuffer(buffer_size, batch_size)
        self.batch_size = batch_size
        
        # Noise process for exploration
        self.noise = OUNoise(action_size)
        
    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        
        if add_noise:
            action += self.noise.sample()
        
        # Clip to ensure action is within bounds
        return np.clip(action, -2.0, 2.0)
    
    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn, if enough samples are available in memory
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)
            
    def learn(self, experiences):
        """Update policy and value parameters using batch of experience tuples."""
        states, actions, rewards, next_states, dones = experiences
        
        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, self.tau)
        self.soft_update(self.actor_local, self.actor_target, self.tau)                     
    
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters: θ_target = τ*θ_local + (1 - τ)*θ_target"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
    
    def save(self, filename):
        """Save the models"""
        torch.save({
            'actor_state_dict': self.actor_local.state_dict(),
            'critic_state_dict': self.critic_local.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, filename)
    
    def load(self, filename):
        """Load the models"""
        if os.path.isfile(filename):
            checkpoint = torch.load(filename)
            self.actor_local.load_state_dict(checkpoint['actor_state_dict'])
            self.critic_local.load_state_dict(checkpoint['critic_state_dict'])
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            
            # Copy to target networks
            self.soft_update(self.actor_local, self.actor_target, 1.0)
            self.soft_update(self.critic_local, self.critic_target, 1.0)
            print(f"Loaded model from {filename}")
        else:
            print(f"No model found at {filename}")

def make_env():
    # Create Pendulum-v1 environment
    env = gym.make("Pendulum-v1")
    return env

def train(n_episodes=2000, max_t=1000, print_every=100, save_path='model.pth'):
    """DDPG training algorithm"""
    env = make_env()
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    
    agent = DDPGAgent(state_size, action_size)
    
    scores = []
    recent_scores = deque(maxlen=100)
    
    for i_episode in range(1, n_episodes+1):
        state, _ = env.reset()
        agent.noise.reset()
        score = 0
        
        for t in range(max_t):
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            
            if done:
                break
        
        # Save scores
        scores.append(score)
        recent_scores.append(score)
        
        # Print progress
        if i_episode % print_every == 0:
            print(f'Episode {i_episode}\tAverage Score: {np.mean(recent_scores):.2f}')
        
        # Save checkpoint when performance improves
        if i_episode % 100 == 0:
            agent.save(save_path)
            print(f'Model saved to {save_path}')
    
    return scores, agent

def test_agent(agent, n_episodes=100, render=False):
    """Test the trained agent"""
    env = make_env()
    rewards = []
    
    for i in range(n_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.act(state, add_noise=False)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            state = next_state
            episode_reward += reward
            
            if render:
                env.render()
        
        rewards.append(episode_reward)
    
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    score = mean_reward - std_reward
    
    print(f"Test Results over {n_episodes} episodes:")
    print(f"Mean Reward: {mean_reward:.2f}")
    print(f"Std Reward: {std_reward:.2f}")
    print(f"Score (Mean - Std): {score:.2f}")
    
    return rewards

def plot_scores(scores, rolling_window=100):
    """Plot scores and rolling mean"""
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(scores)), scores, label='Score')
    
    # Calculate rolling mean
    rolling_mean = []
    for i in range(len(scores)):
        rolling_mean.append(np.mean(scores[max(0, i-rolling_window):(i+1)]))
    
    plt.plot(np.arange(len(rolling_mean)), rolling_mean, label=f'{rolling_window}-episode Rolling Mean')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.legend()
    plt.savefig('training_scores.png')
    plt.show()

if __name__ == "__main__":
    start_time = time.time()
    
    # Train agent
    print("Starting training...")
    scores, agent = train(n_episodes=1000, save_path='model.pth')
    
    # Plot results
    plot_scores(scores)
    
    # Test agent
    print("\nTesting agent performance...")
    test_rewards = test_agent(agent, n_episodes=100)
    
    # Save final model
    agent.save('final_model.pth')
    
    print(f"Total runtime: {(time.time() - start_time)/60:.2f} minutes")