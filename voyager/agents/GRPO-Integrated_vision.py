import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
from collections import deque
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple

@dataclass
class WorldState:
    visual_embedding: torch.Tensor
    spatial_state: torch.Tensor
    inventory: Dict[str, int]
    entity_states: List[Dict]
    temporal_context: torch.Tensor

# Vision Transformer for Processing Environment Inputs
class ViT(nn.Module):
    def __init__(self, image_size=256, patch_size=16, num_classes=512, dim=1024):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, dim, patch_size, patch_size)
        num_patches = (image_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=8), num_layers=6
        )
        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        b, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.transformer(x)
        return self.mlp_head(x[:, 0])

# Transformer-Based Memory for Past Events
class TransformerMemory(nn.Module):
    def __init__(self, memory_size=1000, embedding_dim=512):
        super().__init__()
        self.memory = nn.Parameter(torch.randn(memory_size, embedding_dim))
        self.attention = nn.MultiheadAttention(embedding_dim, num_heads=8)

    def read(self, query):
        attn_output, _ = self.attention(query, self.memory, self.memory)
        return attn_output

# GRPO-Based Hierarchical Controller for Decision Making
class GRPOController(nn.Module):
    def __init__(self, state_dim=512, num_actions=64):
        super().__init__()
        self.policy_network = nn.Linear(state_dim, num_actions)
        self.value_network = nn.Linear(state_dim, 1)

    def forward(self, state):
        action_logits = self.policy_network(state)
        action_probs = torch.softmax(action_logits, dim=-1)
        return action_probs

    def compute_grpo_loss(self, action_probs, rewards):
        relative_advantage = rewards - rewards.mean()
        loss = -torch.sum(action_probs * relative_advantage)
        return loss

# MultiModal Encoder: Fusing Vision, Spatial, and Inventory Data
class MultiModalEncoder(nn.Module):
    def __init__(self, visual_dim=512, spatial_dim=256, inventory_dim=128):
        super().__init__()
        self.visual_net = ViT()
        self.spatial_net = nn.Linear(spatial_dim, 512)
        self.inventory_net = nn.Linear(inventory_dim, 512)
        self.fusion_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=512, nhead=8), num_layers=4
        )

    def forward(self, visual, spatial, inventory):
        visual_enc = self.visual_net(visual)
        spatial_enc = self.spatial_net(spatial)
        inventory_enc = self.inventory_net(inventory)
        combined = torch.stack([visual_enc, spatial_enc, inventory_enc])
        return self.fusion_transformer(combined)

# Full AI Agent System with GRPO-Based Training
class GRPOAgent:
    def __init__(self, checkpoint_dir="checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)

        self.vision_transformer = ViT()
        self.memory = TransformerMemory()
        self.controller = GRPOController()
        self.encoder = MultiModalEncoder()

        self.optimizer = optim.Adam(
            list(self.vision_transformer.parameters()) +
            list(self.controller.parameters()) +
            list(self.encoder.parameters()),
            lr=0.001
        )

        self.training_step = 0
        self.replay_buffer = deque(maxlen=1000)

    def process_observation(self, visual_input, spatial_input, inventory):
        with torch.no_grad():
            multimodal_embedding = self.encoder(visual_input, spatial_input, inventory)
            world_state = WorldState(
                visual_embedding=multimodal_embedding[0],
                spatial_state=multimodal_embedding[1],
                inventory=inventory,
                entity_states=[],
                temporal_context=self.memory.read(multimodal_embedding.mean(0))
            )
            return world_state

    def select_action(self, state: WorldState) -> Tuple[int, float]:
        with torch.no_grad():
            action_probs = self.controller(
                torch.cat([state.visual_embedding, state.spatial_state, state.temporal_context])
            )
            action_dist = Categorical(action_probs)
            action = action_dist.sample()
            return action.item(), action_dist.log_prob(action).item()

    def grpo_update(self, batch):
        states, actions, rewards = zip(*batch)
        action_probs = self.controller(states)

        relative_rewards = rewards - np.mean(rewards)
        loss = -torch.sum(action_probs * relative_rewards)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self, episodes=500):
        for _ in range(episodes):
            batch = []
            for _ in range(10):  # Collect 10 experiences per episode
                state = torch.randn(512)
                action, log_prob = self.select_action(state)
                reward = np.random.rand()  # Simulated reward
                batch.append((state, action, reward))

            self.grpo_update(batch)
            self.training_step += 1

    def save_checkpoint(self):
        checkpoint = {
            'controller': self.controller.state_dict(),
            'encoder': self.encoder.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'training_step': self.training_step
        }
        torch.save(checkpoint, self.checkpoint_dir / 'model.pt')

    def load_checkpoint(self):
        checkpoint = torch.load(self.checkpoint_dir / 'model.pt')
        self.controller.load_state_dict(checkpoint['controller'])
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.training_step = checkpoint['training_step']

# Run Training
agent = GRPOAgent()
agent.train(episodes=100)
agent.save_checkpoint()
