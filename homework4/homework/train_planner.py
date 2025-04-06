import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets.road_dataset import load_data
from models import TransformerPlanner, save_model
from metrics import PlannerMetric
from models import MLPPlanner, TransformerPlanner, CNNPlanner, save_model
from models import load_model



def train_planner(
    dataset_path: str = "drive_data",
    num_epochs: int = 20,
    lr: float = 1e-3,
    batch_size: int = 16,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    print("Loading data...")
    train_loader = load_data(
        Path(dataset_path) / "train",
        transform_pipeline="default",
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = load_data(
        Path(dataset_path) / "val",
        transform_pipeline="default",
        batch_size=batch_size,
        shuffle=False,
    )

    model = load_model(args.model_name).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss(reduction="none")

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            img = batch["image"].to(device)
            track_right = batch["track_right"].to(device)
            target = batch["waypoints"].to(device)
            mask = batch["waypoints_mask"].to(device)

            optimizer.zero_grad()
            # Determine model type based on args
            if isinstance(model, CNNPlanner):
                output = model(img)
            else:
                track_left = batch["track_left"].to(device)
                track_right = batch["track_right"].to(device)
                output = model(track_left=track_left, track_right=track_right)

            loss = criterion(output, target)
            loss = (loss * mask.unsqueeze(-1)).mean()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                img = batch["image"].to(device)
                track_right = batch["track_right"].to(device)
                target = batch["waypoints"].to(device)
                mask = batch["waypoints_mask"].to(device)

                # Determine model type based on args
                if isinstance(model, CNNPlanner):
                    output = model(img)
                else:
                    track_left = batch["track_left"].to(device)
                    track_right = batch["track_right"].to(device)
                    output = model(track_left=track_left, track_right=track_right)

                loss = criterion(output, target)
                loss = (loss * mask.unsqueeze(-1)).mean()
                val_loss += loss.item()

            val_loss /= len(val_loader)

        print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    save_model(model)
    print("✅ Planner model saved successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="transformer_planner",
                    choices=["mlp_planner", "transformer_planner", "cnn_planner"])

    parser.add_argument("--dataset_path", type=str, default="drive_data")
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    train_planner(
        dataset_path=args.dataset_path,
        num_epochs=args.num_epochs,
        lr=args.lr,
        batch_size=args.batch_size,
    )
    print("Training completed.")
    print("Model saved successfully.")