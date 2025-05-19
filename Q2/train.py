import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import os
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from dmc import make_dmc_env

# Set random seed for reproducibility
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Actor-Critic Network
class ActorCritic(nn.Module):
    def __init__(self, obs_size, act_size):
        super(ActorCritic, self).__init__()
        
        self.shared = nn.Sequential(
            nn.Linear(obs_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        self.actor_mean = nn.Linear(64, act_size)
        self.actor_logstd = nn.Parameter(torch.zeros(act_size))
        self.critic = nn.Linear(64, 1)
        
    def forward(self, x):
        features = self.shared(x)
        mean = self.actor_mean(features)
        log_std = self.actor_logstd.expand_as(mean)
        std = torch.exp(log_std)
        dist = Normal(mean, std)
        value = self.critic(features)
        return dist, value

# PPO Agent
class PPO:
    def __init__(self, obs_size, act_size, lr=3e-4, gamma=0.99, clip_eps=0.2, epochs=10, batch_size=64):
        self.device = device
        self.model = ActorCritic(obs_size, act_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size
        
    def compute_gae(self, rewards, values, next_value, dones, gae_lambda=0.95):
        advantages = []
        gae = 0
        values = values + [next_value]
        
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * values[i + 1] * (1 - dones[i]) - values[i]
            gae = delta + self.gamma * gae_lambda * (1 - dones[i]) * gae
            advantages.insert(0, gae)
        
        return advantages
    
    def update(self, states, actions, log_probs_old, returns, advantages):
        states = np.array(states)
        actions = np.array(actions)
        log_probs_old = np.array(log_probs_old)
        returns = np.array(returns)
        advantages = np.array(advantages)
        
        # Normalize advantages (improves training stability)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        indices = np.random.permutation(len(states))
        
        for _ in range(self.epochs):
            for start in range(0, len(states), self.batch_size):
                idx = indices[start:start + self.batch_size]
                
                batch_states = torch.FloatTensor(states[idx]).to(self.device)
                batch_actions = torch.FloatTensor(actions[idx]).to(self.device)
                batch_log_probs_old = torch.FloatTensor(log_probs_old[idx]).to(self.device)
                batch_returns = torch.FloatTensor(returns[idx]).to(self.device)
                batch_advantages = torch.FloatTensor(advantages[idx]).to(self.device)
                
                dist, value = self.model(batch_states)
                log_probs = dist.log_prob(batch_actions).sum(-1)
                entropy = dist.entropy().mean()
                
                ratio = torch.exp(log_probs - batch_log_probs_old)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * batch_advantages
                
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = (batch_returns - value.squeeze()).pow(2).mean()
                
                # Total loss - actor loss + value loss - entropy bonus
                loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
    
    def get_action(self, state, deterministic=False):
        """Get action from policy given state"""
        # Handle different state formats
        if isinstance(state, tuple):
            state = state[0]  # Extract state from tuple if needed
            
        # Ensure state is properly formatted as a tensor
        if isinstance(state, np.ndarray):
            state_tensor = torch.FloatTensor(state).to(self.device)
        else:
            # If it's not a numpy array, try to convert it
            state_tensor = torch.FloatTensor([state]).to(self.device)
        
        with torch.no_grad():
            dist, _ = self.model(state_tensor)
            
            if deterministic:
                # For deterministic actions, just use the mean
                action = dist.mean
            else:
                # Sample from the distribution
                action = dist.sample()
            
            # Apply tanh to bound actions to [-1, 1]
            action = torch.tanh(action)
            
        return action.cpu().numpy().squeeze()
        
    def save(self, path):
        """Save the model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
    
    def load(self, path):
        """Load the model"""
        if os.path.isfile(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"Loaded model from {path}")
        else:
            print(f"No model found at {path}")

def make_env():
    # Create environment with state observations
    env_name = "cartpole-balance"
    env = make_dmc_env(env_name, np.random.randint(0, 1000000), flatten=True, use_pixels=False)
    return env

def evaluate(agent, env, num_episodes=10):
    """Evaluate the agent"""
    rewards = []
    
    for _ in range(num_episodes):
        # Reset environment
        state = env.reset(seed=np.random.randint(0, 1000000))
        # Handle tuple state
        if isinstance(state, tuple):
            state = state[0]
            
        episode_reward = 0
        done = False
        
        while not done:
            # Get deterministic action
            action = agent.get_action(state, deterministic=True)
            
            # Take step in environment
            step_result = env.step(action)
            
            # Handle different return formats
            if len(step_result) == 4:  # Old gym format
                next_state, reward, done, _ = step_result
            elif len(step_result) == 5:  # New gym format
                next_state, reward, terminated, truncated, _ = step_result
                done = terminated or truncated
            
            # Update state and reward
            state = next_state
            episode_reward += reward
        
        rewards.append(episode_reward)
    
    # Compute statistics
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    score = mean_reward - std_reward
    
    return rewards, mean_reward, std_reward, score

def train(max_episodes=1000, max_steps=1000, update_freq=2048, save_path='ppo_model.pth'):
    """PPO training algorithm"""
    # Create environment
    env = make_env()
    
    # Get state and action dimensions
    obs_size = env.observation_space.shape[0]
    act_size = env.action_space.shape[0]
    
    # Initialize PPO agent
    agent = PPO(
        obs_size=obs_size,
        act_size=act_size,
        lr=3e-4,
        gamma=0.99,
        clip_eps=0.2,
        epochs=10,
        batch_size=64
    )
    
    # Training parameters
    episode_rewards = []
    states, actions, rewards, log_probs, values, dones = [], [], [], [], [], []
    best_score = -float('inf')
    update_counter = 0
    
    # Progress bar for episodes
    pbar = tqdm(range(max_episodes), desc="Training", unit="episode")
    for episode in pbar:
        # Reset environment
        state = env.reset(seed=np.random.randint(0, 1000000))
        if isinstance(state, tuple):
            state = state[0]
            
        episode_reward = 0
        
        for step in range(max_steps):
            # Convert state to tensor
            state_tensor = torch.FloatTensor(state).to(device)
            
            # Get action from policy
            dist, value = agent.model(state_tensor)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(-1)
            
            # Apply tanh for action bounds
            action_bounded = torch.tanh(action)
            action_np = action_bounded.cpu().detach().numpy()
            
            # Take step in environment
            step_result = env.step(action_np)
            
            # Handle different return formats
            if len(step_result) == 4:  # Old gym format
                next_state, reward, done, _ = step_result
            elif len(step_result) == 5:  # New gym format
                next_state, reward, terminated, truncated, _ = step_result
                done = terminated or truncated
            
            # Store experience
            states.append(state)
            actions.append(action.cpu().detach().numpy())  # Store original action
            rewards.append(reward)
            log_probs.append(log_prob.item())
            values.append(value.item())
            dones.append(1 if done else 0)
            
            # Update state and reward
            state = next_state
            episode_reward += reward
            update_counter += 1
            
            if done:
                break
        
        # Store episode reward
        episode_rewards.append(episode_reward)
        
        # Compute next value for incomplete episode
        state_tensor = torch.FloatTensor(state).to(device)
        _, next_value = agent.model(state_tensor)
        next_value = next_value.item()
        
        # Update policy if reached update frequency
        if update_counter >= update_freq:
            # Compute returns and advantages
            advantages = agent.compute_gae(rewards, values, next_value, dones)
            returns = [adv + val for adv, val in zip(advantages, values)]
            
            # Update policy
            agent.update(states, actions, log_probs, returns, advantages)
            
            # Clear experience buffer
            states, actions, rewards, log_probs, values, dones = [], [], [], [], [], []
            update_counter = 0
            
            # Evaluate agent
            _, mean_reward, std_reward, score = evaluate(agent, env, num_episodes=5)
            
            # Update progress bar with evaluation info
            pbar.set_postfix({
                "Episode": episode,
                "Mean Reward": f"{mean_reward:.2f}",
                "Score": f"{score:.2f}"
            })
            
            # Save best model
            if score > best_score:
                best_score = score
                agent.save(save_path)
                print(f"\nSaved best model with score {score:.2f} at episode {episode}")
                
            # Early stopping condition
            if score > 950:
                print(f"\nEnvironment solved at episode {episode} with score {score:.2f}!")
                break
    
    # Save final model
    agent.save(f'final_{save_path}')
    print(f"Final model saved to final_{save_path}")
    
    return episode_rewards, agent

def plot_scores(scores, rolling_window=100):
    """Plot scores and rolling mean"""
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(scores)), scores, label='Score', alpha=0.4)
    
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
    scores, agent = train(
        max_episodes=1000,
        max_steps=1000,
        update_freq=2048,
        save_path='ppo_model.pth'
    )
    
    # Plot results
    plot_scores(scores)
    
    # Test the best model
    print("\nTesting best model performance...")
    agent.load('ppo_model.pth')  # Load best model
    rewards, mean_reward, std_reward, score = evaluate(agent, make_env(), num_episodes=100)
    
    print(f"Test Results over 100 episodes:")
    print(f"Mean Reward: {mean_reward:.2f}")
    print(f"Std Reward: {std_reward:.2f}")
    print(f"Score (Mean - Std): {score:.2f}")
    
    print(f"Total runtime: {(time.time() - start_time)/60:.2f} minutes")