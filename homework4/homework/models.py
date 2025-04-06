from pathlib import Path

import torch
import torch.nn as nn

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


class MLPPlanner(nn.Module):
    def __init__(self, n_track: int = 10, n_waypoints: int = 3):
        super().__init__()
        self.n_track = n_track
        self.n_waypoints = n_waypoints

        input_dim = n_track * 6  # left(2), right(2), diff(2) => total 6 per point

        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, n_waypoints * 2)
        )

    def forward(self, track_left: torch.Tensor, track_right: torch.Tensor, **kwargs) -> torch.Tensor:
        # Normalize: subtract ego-center (first left point)
        origin = track_left[:, 0:1, :]  # shape (B, 1, 2)
        track_left = track_left - origin
        track_right = track_right - origin

        # Add relative difference
        track_diff = track_left - track_right

        # Concatenate [left, right, diff] → (B, N, 6)
        x = torch.cat([track_left, track_right, track_diff], dim=-1)

        # Flatten → (B, N*6)
        x = x.view(x.size(0), -1)

        out = self.model(x)  # (B, N*2)
        return out.view(-1, self.n_waypoints, 2)



import torch
import torch.nn as nn

import torch
import torch.nn as nn


class TransformerPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints
        self.d_model = d_model

        # Input projection: (x, z) → d_model
        self.input_proj = nn.Linear(2, d_model)

        # Positional encodings
        self.query_embed = nn.Embedding(n_waypoints, d_model)
        self.positional_encoding = nn.Parameter(torch.randn(n_track * 2, d_model))

        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # Enables (B, N, D) input format
        )

        # Output projection to (x, z) coordinates
        self.output_proj = nn.Linear(d_model, 2)

    def forward(
        self,
        track_left: torch.Tensor,   # (B, n_track, 2)
        track_right: torch.Tensor,  # (B, n_track, 2)
        **kwargs,
    ) -> torch.Tensor:
        B, N, _ = track_left.shape

        # Concatenate left and right tracks → (B, 2N, 2)
        track = torch.cat([track_left, track_right], dim=1)

        # Project to d_model → (B, 2N, d_model)
        track_feat = self.input_proj(track)

        # Add positional encodings
        pos_enc = self.positional_encoding[: track_feat.shape[1]]
        track_feat = track_feat + pos_enc[None, :, :]

        # Prepare decoder queries → (B, n_waypoints, d_model)
        query_embed = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)

        # Run transformer
        memory = self.transformer.encoder(track_feat)
        out = self.transformer.decoder(query_embed, memory)

        # Project to (x, z) → (B, n_waypoints, 2)
        waypoints = self.output_proj(out)

        return waypoints



class CNNPlanner(nn.Module):
    def __init__(self, in_channels=3, num_waypoints=3):
        super().__init__()
        self.num_waypoints = num_waypoints

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=5, stride=2, padding=2),  # (B, 16, H/2, W/2)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),  # (B, 32, H/4, W/4)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),  # (B, 64, H/8, W/8)
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),  # (B, 64, 4, 4)
        )

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, num_waypoints * 2),
        )

    # def forward(self, x):
    #     x = self.features(x)
    #     x = self.regressor(x)
    #     return x.view(-1, self.num_waypoints, 2)

    def forward(self, image: torch.Tensor, **kwargs):
        x = self.features(image)
        x = self.regressor(x)
        return x.view(-1, self.num_waypoints, 2)




MODEL_FACTORY = {
    "mlp_planner": MLPPlanner,
    "transformer_planner": TransformerPlanner,
    "cnn_planner": CNNPlanner,
}


def load_model(
    model_name: str,
    with_weights: bool = False,
    **model_kwargs,
) -> torch.nn.Module:
    """
    Called by the grader to load a pre-trained model by name
    """
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(m)

    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m


def save_model(model: torch.nn.Module) -> str:
    """
    Use this function to save your model in train.py
    """
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Naive way to estimate model size
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024
