from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from rl.encoding import ACTION_DIM, STATE_DIM


class DominoNet(nn.Module):
    """Actor-critic MLP for Cuban Dominos (Double-9, 4P Pairs).

    Forward returns (action_probs, value) where action_probs is a masked
    softmax over the ACTION_DIM action space.
    """

    def __init__(self, hidden_dim: int = 256, state_dim: int = STATE_DIM, num_layers: int = 4) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.num_layers = num_layers

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # Residual blocks (num_layers - 2 of them, since input and output are separate)
        self.res_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            )
            for _ in range(num_layers - 2)
        ])

        # Output projection (narrows to hidden_dim // 2)
        narrow = hidden_dim // 2
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, narrow),
            nn.LayerNorm(narrow),
            nn.ReLU(),
        )

        self.policy_head = nn.Linear(narrow, ACTION_DIM)
        self.value_head = nn.Linear(narrow, 1)

    def forward(
        self,
        state: torch.Tensor,        # (B, STATE_DIM)
        action_mask: torch.Tensor,  # (B, ACTION_DIM) bool — True = valid
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (action_probs (B, ACTION_DIM), value (B, 1))."""
        h = self.input_proj(state)
        for block in self.res_blocks:
            h = h + block(h)  # residual connection
        h = self.output_proj(h)
        logits = self.policy_head(h)
        logits = logits.masked_fill(~action_mask, float("-inf"))
        probs = F.softmax(logits, dim=-1)
        value = self.value_head(h)
        return probs, value
