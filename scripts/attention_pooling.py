"""Learned temporal pooling modules for utterance classification."""

from __future__ import annotations

import math

import torch
from torch import nn

ATTENTION_POOLING = {"attentive_global", "ordered8", "attentive_combined"}


def sequence_mask(lengths: torch.Tensor, width: int) -> torch.Tensor:
    """Return a Boolean mask for valid sequence positions."""
    positions = torch.arange(width, device=lengths.device).unsqueeze(0)
    return positions < lengths.unsqueeze(1)


class AttentiveStatisticsPooling(nn.Module):
    """Learn a scalar weight for every frame, then pool mean and deviation."""

    def __init__(self, width: int, attention_size: int = 128) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(width, attention_size),
            nn.Tanh(),
            nn.Linear(attention_size, 1),
        )

    def forward(
        self, hidden: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        valid = sequence_mask(lengths, hidden.shape[1])
        logits = self.score(hidden).squeeze(-1).masked_fill(~valid, -torch.inf)
        attention = logits.softmax(dim=1)
        mean = torch.einsum("bt,btd->bd", attention, hidden)
        variance = torch.einsum(
            "bt,btd->bd", attention, (hidden - mean.unsqueeze(1)).square()
        )
        return torch.cat((mean, variance.clamp_min(1e-7).sqrt()), dim=1), attention


class OrderedAttentionPooling(nn.Module):
    """Pool a sequence with content-aware heads having learnable ordered windows."""

    def __init__(
        self,
        width: int,
        heads: int = 8,
        frame_projection_size: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.frame_projection_size = frame_projection_size
        attention_size = 128
        self.keys = nn.Sequential(nn.Linear(width, attention_size), nn.Tanh())
        self.queries = nn.Parameter(torch.empty(heads, attention_size))
        nn.init.normal_(self.queries, std=0.02)
        initial_centers = torch.linspace(0.08, 0.92, heads)
        self.center_logits = nn.Parameter(torch.logit(initial_centers))
        self.log_widths = nn.Parameter(torch.full((heads,), math.log(0.22)))
        self.frame_project = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, frame_projection_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self, hidden: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch, frames, _ = hidden.shape
        valid = sequence_mask(lengths, frames)
        keys = self.keys(hidden)
        content = torch.einsum("btd,hd->bht", keys, self.queries) / math.sqrt(
            keys.shape[-1]
        )
        positions = torch.arange(frames, device=hidden.device).view(1, 1, -1)
        denominators = (lengths - 1).clamp_min(1).view(-1, 1, 1)
        relative_positions = positions / denominators
        centers = self.center_logits.sigmoid().view(1, -1, 1)
        widths = self.log_widths.exp().clamp(0.05, 1.0).view(1, -1, 1)
        position_bias = -0.5 * ((relative_positions - centers) / widths).square()
        logits = (content + position_bias).masked_fill(~valid.unsqueeze(1), -torch.inf)
        attention = logits.softmax(dim=2)
        projected = self.frame_project(hidden)
        summaries = torch.einsum("bht,btd->bhd", attention, projected)
        empirical_centers = (attention * relative_positions).sum(dim=2)
        normalized = attention / attention.square().sum(dim=2, keepdim=True).sqrt()
        similarity = torch.bmm(normalized, normalized.transpose(1, 2))
        identity = torch.eye(self.heads, device=hidden.device).unsqueeze(0)
        diversity_loss = (similarity - identity).square().mean()
        ordering_loss = torch.relu(
            empirical_centers[:, :-1] - empirical_centers[:, 1:] + 0.03
        ).mean()
        entropy = -(attention.clamp_min(1e-8).log() * attention).sum(dim=2)
        return summaries.flatten(1), {
            "attention": attention,
            "centers": empirical_centers,
            "entropy": entropy,
            "diversity_loss": diversity_loss,
            "ordering_loss": ordering_loss,
        }
