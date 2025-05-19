import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import os

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

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

class Agent:
    def __init__(self):
        # Set device to CPU for evaluation
        self.device = torch.device("cpu")
        
        # Define state and action dimensions
        self.obs_size = 5  # CartPole Balance has 5D state
        self.act_size = 1  # CartPole Balance has 1D action
        
        # Create model
        self.model = ActorCritic(self.obs_size, self.act_size).to(self.device)
        self.model.eval()  # Set to evaluation mode
        
        # Load saved model
        self.load_model()
    
    def load_model(self):
        """Attempt to load model from various possible file paths"""
        # Define all possible model file paths
        model_paths = [
            'ppo_model.pth', 
            'final_ppo_model.pth',
            'ppo_model.pth_state_dict',
            'final_ppo_model.pth_state_dict',
            'model.pth',
            'final_model.pth'
        ]
        
        loaded = False
        
        # Try each path
        for path in model_paths:
            if not os.path.exists(path):
                continue
                
            try:
                checkpoint = torch.load(path, map_location=self.device)
                
                # Handle different checkpoint formats
                if isinstance(checkpoint, dict):
                    if 'model_state_dict' in checkpoint:
                        self.model.load_state_dict(checkpoint['model_state_dict'])
                    elif 'network_state_dict' in checkpoint:
                        self.model.load_state_dict(checkpoint['network_state_dict'])
                    elif 'state_dict' in checkpoint:
                        self.model.load_state_dict(checkpoint['state_dict'])
                    else:
                        continue  # Try next file
                else:
                    # Direct state dict
                    self.model.load_state_dict(checkpoint)
                    
                print(f"Successfully loaded model from: {path}")
                loaded = True
                break
            except Exception as e:
                print(f"Failed to load from {path}: {e}")
        
        if not loaded:
            print("Could not load any model, using untrained weights")
    
    def act(self, observation):
        """Return action for given observation"""
        # Handle different observation formats
        if isinstance(observation, tuple):
            observation = observation[0]
        
        # Convert observation to tensor
        if isinstance(observation, np.ndarray):
            state = torch.FloatTensor(observation).to(self.device)
        else:
            state = torch.FloatTensor([observation]).to(self.device)
        
        # Get deterministic action
        with torch.no_grad():
            dist, _ = self.model(state)
            action = dist.mean  # Use mean for deterministic action
            action = torch.tanh(action)  # Constrain to [-1, 1]
        
        return action.cpu().numpy().squeeze()