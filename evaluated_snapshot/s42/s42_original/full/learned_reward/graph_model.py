import torch
from torch import nn


class GraphBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.message = nn.Sequential(
            nn.Linear(3 * width, 2 * width), nn.SiLU(),
            nn.Linear(2 * width, width),
        )
        self.update = nn.Sequential(
            nn.Linear(2 * width, 2 * width), nn.SiLU(),
            nn.Linear(2 * width, width),
        )
        self.norm = nn.LayerNorm(width)

    def forward(self, nodes, action):
        b, n, d = nodes.shape
        receiver = nodes[:, :, None, :].expand(b, n, n, d)
        sender = nodes[:, None, :, :].expand(b, n, n, d)
        control = action[:, None, None, :].expand(b, n, n, d)
        messages = self.message(
            torch.cat([receiver, sender, control], dim=-1)
        )
        mask = (~torch.eye(
            n, device=nodes.device, dtype=torch.bool
        ))[None, :, :, None]
        aggregated = (messages * mask).sum(dim=2) / (n - 1)
        return self.norm(
            nodes + self.update(
                torch.cat([nodes, aggregated], dim=-1)
            )
        )


class LYGraphDynamics(nn.Module):
    def __init__(
        self, feature_dim, feature_nodes=16, state_dim=20,
        action_dim=12, horizon=16, width=384, blocks=4
    ):
        super().__init__()
        self.config = dict(
            feature_dim=feature_dim, feature_nodes=feature_nodes,
            state_dim=state_dim, action_dim=action_dim,
            horizon=horizon, width=width, blocks=blocks,
        )
        self.feature_in = nn.Linear(feature_dim, width)
        self.state_in = nn.Linear(state_dim, width)
        self.action_in = nn.Sequential(
            nn.Linear(horizon * action_dim, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )
        self.node_embedding = nn.Parameter(
            torch.zeros(feature_nodes + 1, width)
        )
        self.blocks = nn.ModuleList(
            [GraphBlock(width) for _ in range(blocks)]
        )
        self.feature_out = nn.Linear(width, feature_dim)
        self.state_out = nn.Sequential(
            nn.Linear(width, width), nn.SiLU(),
            nn.Linear(width, state_dim),
        )

    def forward(self, features, state, actions):
        nodes = torch.cat([
            self.feature_in(features),
            self.state_in(state)[:, None],
        ], dim=1)
        nodes = nodes + self.node_embedding[None]
        control = self.action_in(actions.flatten(1))
        for block in self.blocks:
            nodes = block(nodes, control)
        return dict(
            features=features + self.feature_out(nodes[:, :-1]),
            state=state + self.state_out(nodes[:, -1]),
        )
