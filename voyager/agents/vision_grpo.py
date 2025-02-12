import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
import numpy as np
from typing import Dict, List, Tuple, Optional

class GRPONetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, group_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.group_dim = group_dim
        
        # Feature extractor shared between policy and value
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Group-specific policy heads
        self.group_policy_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim)
            ) for _ in range(group_dim)
        ])
        
        # Value function
        self.value = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Group embeddings
        self.group_embeddings = nn.Embedding(group_dim, hidden_dim)

    def forward(self, state: torch.Tensor, group_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.feature_extractor(state)
        
        # Get group-specific features
        group_embeds = self.group_embeddings(group_ids)
        features = features + group_embeds
        
        # Get actions for each group
        action_logits = torch.stack([
            head(features[group_ids == i]) 
            for i, head in enumerate(self.group_policy_heads)
        ])
        
        value = self.value(features)
        return action_logits, value

class GRPO:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        group_dim: int,
        learning_rate: float = 3e-4,
        clip_ratio: float = 0.2,
        target_kl: float = 0.01,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01
    ):
        self.network = GRPONetwork(state_dim, action_dim, group_dim)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
        
        self.clip_ratio = clip_ratio
        self.target_kl = target_kl
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        
        # Group-specific statistics
        self.group_advantages = {i: [] for i in range(group_dim)}
        self.group_returns = {i: [] for i in range(group_dim)}

    def compute_group_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        group_ids: torch.Tensor,
        gamma: float = 0.99,
        lambda_: float = 0.95
    ) -> Dict[int, torch.Tensor]:
        """Compute advantages for each group using GAE"""
        advantages = {}
        
        for group_id in range(self.network.group_dim):
            group_mask = group_ids == group_id
            if not group_mask.any():
                continue
                
            group_rewards = rewards[group_mask]
            group_values = values[group_mask]
            group_dones = dones[group_mask]
            
            # Compute GAE
            advantages[group_id] = self._compute_gae(
                group_rewards, group_values, group_dones, gamma, lambda_
            )
            
            # Normalize advantages within group
            advantages[group_id] = (advantages[group_id] - advantages[group_id].mean()) / (advantages[group_id].std() + 1e-8)
            
        return advantages

    def _compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        gamma: float,
        lambda_: float
    ) -> torch.Tensor:
        """Compute Generalized Advantage Estimation"""
        gae = 0
        advantages = torch.zeros_like(rewards)
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
                
            delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + gamma * lambda_ * (1 - dones[t]) * gae
            advantages[t] = gae
            
        return advantages

    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: Dict[int, torch.Tensor],
        returns: torch.Tensor,
        group_ids: torch.Tensor,
        num_epochs: int = 10
    ) -> Dict[str, float]:
        """Update policy using GRPO"""
        metrics = []
        
        for _ in range(num_epochs):
            # Get current policy distributions and values
            action_logits, values = self.network(states, group_ids)
            
            # Compute policy loss for each group
            total_policy_loss = 0
            total_value_loss = 0
            total_entropy_loss = 0
            
            for group_id in range(self.network.group_dim):
                group_mask = group_ids == group_id
                if not group_mask.any():
                    continue
                
                group_actions = actions[group_mask]
                group_old_log_probs = old_log_probs[group_mask]
                group_advantages = advantages[group_id]
                
                # Get new action distributions
                dist = Categorical(logits=action_logits[group_id])
                new_log_probs = dist.log_prob(group_actions)
                
                # Compute ratio and clipped ratio
                ratio = torch.exp(new_log_probs - group_old_log_probs)
                clipped_ratio = torch.clamp(
                    ratio, 1 - self.clip_ratio, 1 + self.clip_ratio
                )
                
                # Compute losses
                policy_loss = -torch.min(
                    ratio * group_advantages,
                    clipped_ratio * group_advantages
                ).mean()
                
                value_loss = F.mse_loss(values[group_mask], returns[group_mask])
                entropy_loss = -dist.entropy().mean()
                
                # Accumulate losses
                total_policy_loss += policy_loss
                total_value_loss += value_loss
                total_entropy_loss += entropy_loss
                
                # Check KL divergence
                approx_kl = ((ratio - 1) - torch.log(ratio)).mean().item()
                if approx_kl > self.target_kl:
                    break
            
            # Compute total loss
            total_loss = (
                total_policy_loss +
                self.value_coef * total_value_loss +
                self.entropy_coef * total_entropy_loss
            )
            
            # Optimize
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            
            metrics.append({
                'policy_loss': total_policy_loss.item(),
                'value_loss': total_value_loss.item(),
                'entropy_loss': total_entropy_loss.item(),
                'total_loss': total_loss.item(),
                'approx_kl': approx_kl
            })
        
        return {k: np.mean([m[k] for m in metrics]) for k in metrics[0].keys()}

    def get_action(
        self,
        state: torch.Tensor,
        group_id: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get action from policy"""
        with torch.no_grad():
            action_logits, _ = self.network(state, group_id)
            dist = Categorical(logits=action_logits[group_id])
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
        return action, log_prob



# Key features of this GRPO implementation:

# 1. **Group-Specific Policy Heads**
# - Separate policy networks for each group
# - Shared feature extractor
# - Group embeddings for better group differentiation

# 2. **Relative Advantage Computation**
# - Advantages computed within groups
# - Group-specific normalization
# - GAE (Generalized Advantage Estimation) support

# 3. **GRPO Update Mechanism**
# - Clipped objective like PPO
# - Group-specific policy updates
# - KL divergence monitoring per group

# 4. **Additional Features**
# - Value function for baseline estimation
# - Entropy regularization
# - Configurable hyperparameters

# To use this implementation:
# # Initialize GRPO
# grpo = GRPO(
#     state_dim=8,
#     action_dim=4,
#     group_dim=3,  # Number of groups
#     learning_rate=3e-4,
#     clip_ratio=0.2
# )

# Training loop
# for episode in range(num_episodes):
#     states, actions, rewards, dones = [], [], [], []
#     group_ids = []
    
#     # Collect experience
#     state = env.reset()
#     group_id = determine_group(state)  # Your group assignment function
    
#     while not done:
#         # Get action from policy
#         action, log_prob = grpo.get_action(
#             torch.FloatTensor(state),
#             torch.LongTensor([group_id])
#         )
        
#         # Execute action
#         next_state, reward, done, _ = env.step(action.item())
        
#         # Store experience
#         states.append(state)
#         actions.append(action)
#         rewards.append(reward)
#         dones.append(done)
#         group_ids.append(group_id)
        
#         state = next_state
#         group_id = determine_group(state)
    
#     # Convert to tensors
#     states = torch.FloatTensor(states)
#     actions = torch.LongTensor(actions)
#     rewards = torch.FloatTensor(rewards)
#     dones = torch.FloatTensor(dones)
#     group_ids = torch.LongTensor(group_ids)
    
    # Compute advantages and returns
    # advantages = grpo.compute_group_advantages(rewards, values, dones, group_ids)
    # returns = rewards + values  # Simplified returns computation
    
    # # Update policy
    # metrics = grpo.update(
    #     states, actions, old_log_probs, advantages, returns, group_ids
    # )


# 1. Add more advanced features like multi-agent coordination?
# 2. Implement specific Minecraft-related group structures?
# 3. Add additional optimization techniques?