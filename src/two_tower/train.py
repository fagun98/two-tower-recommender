"""Training script for two-tower model."""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from two_tower.data import InteractionDataset, MovieLensDataset
from two_tower.model import TwoTowerModel, in_batch_negative_loss


def train_epoch(model, dataloader, genre_features, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0

    for user_ids, item_ids in tqdm(dataloader, desc="Training"):
        user_ids = user_ids.to(device)
        item_ids = item_ids.to(device)

        # Get genre features for these items
        batch_genres = genre_features[item_ids].to(device)

        # Forward pass
        user_emb, item_emb = model(user_ids, item_ids, batch_genres)

        # Compute loss
        loss = in_batch_negative_loss(user_emb, item_emb)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def train(
    data_dir: str = "data/ml-latest-small",
    embedding_dim: int = 64,
    hidden_dim: int = 128,
    batch_size: int = 256,
    num_epochs: int = 10,
    learning_rate: float = 0.001,
    checkpoint_dir: str = "checkpoints",
):
    """Train the two-tower model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    print("Loading data...")
    dataset = MovieLensDataset(data_dir)
    dataset.load()
    dataset.split_leave_last_out()

    # Create dataloaders
    train_dataset = InteractionDataset(dataset.train_data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Get genre features
    genre_features = torch.FloatTensor(dataset.get_item_features())

    # Create model
    print("Creating model...")
    model = TwoTowerModel(
        num_users=dataset.get_num_users(),
        num_items=dataset.get_num_items(),
        num_genres=dataset.get_num_genres(),
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
    )
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    print(f"Training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        avg_loss = train_epoch(model, train_loader, genre_features, optimizer, device)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

    # Save checkpoint
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "embedding_dim": embedding_dim,
            "hidden_dim": hidden_dim,
            "num_users": dataset.get_num_users(),
            "num_items": dataset.get_num_items(),
            "num_genres": dataset.get_num_genres(),
        },
        checkpoint_path / "model.pth",
    )
    print(f"Model saved to {checkpoint_path / 'model.pth'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train two-tower recommendation model")
    parser.add_argument("--data-dir", default="data/ml-latest-small", help="Data directory")
    parser.add_argument("--embedding-dim", type=int, default=64, help="Embedding dimension")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden dimension")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--num-epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--checkpoint-dir", default="checkpoints", help="Checkpoint directory")

    args = parser.parse_args()
    train(**vars(args))
