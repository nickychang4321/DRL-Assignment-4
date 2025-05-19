import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# Set seed for reproducibility
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Device configuration
device = torch.device("cpu")  # Using CPU for evaluation

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

class Agent:
    def __init__(self):
        # Define state and action dimensions for Pendulum
        self.state_size = 3
        self.action_size = 1
        
        # Initialize actor network
        self.actor = Actor(self.state_size, self.action_size).to(device)
        
        # Load pre-trained model
        self.load_model()
    
    def load_model(self):
        """Load the pre-trained model"""
        model_path = 'model.pth'
        
        if os.path.isfile(model_path):
            checkpoint = torch.load(model_path, map_location=device)
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.actor.eval()  # Set the network to evaluation mode
            print(f"Model loaded from {model_path}")
        else:
            print(f"No model found at {model_path}, using untrained model")
    
    def act(self, observation):
        """
        Select an action based on the current policy
        Args:
            observation: the current state observation
        Returns:
            action: the selected action
        """
        # Convert observation to tensor
        state = torch.FloatTensor(observation).to(device)
        
        # Get action from actor network (no gradient needed for inference)
        with torch.no_grad():
            action = self.actor(state).cpu().numpy()
        
        # Ensure action is within the action space bounds
        action = np.clip(action, -2.0, 2.0)
        
        return action