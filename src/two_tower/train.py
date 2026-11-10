"""Training script for two-tower model with improvements."""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm

from two_tower.data import InteractionDataset, MovieLensDataset
from two_tower.model import TwoTowerModel, in_batch_negative_loss


def collate_fn(batch):
    """Collate function to handle variable-length user histories."""
    users, items, histories = zip(*batch)
    users = torch.stack(users)
    items = torch.stack(items)
    # Pad histories to same length
    histories = pad_sequence(histories, batch_first=True, padding_value=0)
    return users, items, histories


def train_epoch(model, dataloader, genre_features, year_features, optimizer, device, temperature, item_freq):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0

    for user_ids, item_ids, user_histories in tqdm(dataloader, desc="Training"):
        user_ids = user_ids.to(device)
        item_ids = item_ids.to(device)
        user_histories = user_histories.to(device)

        # Get features for these items
        batch_genres = genre_features[item_ids].to(device)
        batch_years = year_features[item_ids].to(device)

        # Get item frequencies for popularity correction
        batch_item_freq = item_freq[item_ids].to(device) if item_freq is not None else None

        # Forward pass
        user_emb, item_emb = model(user_ids, item_ids, batch_genres, batch_years, user_histories)

        # Compute loss
        loss = in_batch_negative_loss(user_emb, item_emb, temperature, batch_item_freq)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def evaluate_model(model, dataloader, genre_features, year_features, device):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for user_ids, item_ids, user_histories in dataloader:
            user_ids = user_ids.to(device)
            item_ids = item_ids.to(device)
            user_histories = user_histories.to(device)

            batch_genres = genre_features[item_ids].to(device)
            batch_years = year_features[item_ids].to(device)

            user_emb, item_emb = model(user_ids, item_ids, batch_genres, batch_years, user_histories)
            loss = in_batch_negative_loss(user_emb, item_emb, temperature=0.07)

            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches if num_batches > 0 else float("inf")


def train(
    data_dir: str = "data/ml-latest-small",
    embedding_dim: int = 64,
    hidden_dim: int = 128,
    batch_size: int = 512,
    num_epochs: int = 50,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-5,
    temperature: float = 0.07,
    use_popularity_correction: bool = True,
    checkpoint_dir: str = "checkpoints",
    patience: int = 5,
):
    """Train the two-tower model with early stopping."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    print("Loading data...")
    dataset = MovieLensDataset(data_dir)
    dataset.load()
    dataset.split_leave_last_out(val_ratio=0.1)

    # Build user histories
    print("Building user histories...")
    user_histories = dataset.build_user_histories(max_history=20)

    # Create dataloaders
    train_dataset = InteractionDataset(dataset.train_data, user_histories)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )

    if len(dataset.val_data) > 0:
        val_dataset = InteractionDataset(dataset.val_data, user_histories)
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
        )
    else:
        val_loader = None

    # Get features
    genre_features = torch.FloatTensor(dataset.get_item_features())
    year_features = torch.LongTensor(dataset.get_item_years())

    # Compute item popularity for logQ correction
    item_freq = None
    if use_popularity_correction:
        item_counts = dataset.train_data["item_idx"].value_counts()
        item_freq_array = np.zeros(dataset.get_num_items())
        for item_idx, count in item_counts.items():
            item_freq_array[item_idx] = count
        # Normalize
        item_freq_array = item_freq_array / item_freq_array.sum()
        item_freq = torch.FloatTensor(item_freq_array)

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

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Training loop with early stopping
    print(f"Training for up to {num_epochs} epochs with early stopping (patience={patience})...")
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    for epoch in range(num_epochs):
        train_loss = train_epoch(
            model, train_loader, genre_features, year_features, optimizer, device, temperature, item_freq
        )

        # Validation
        if val_loader is not None:
            val_loss = evaluate_model(model, val_loader, genre_features, year_features, device)
            print(
                f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
                print("  → New best validation loss!")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs")
                    break
        else:
            print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}")

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("Restored model from best validation epoch")

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
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size")
    parser.add_argument("--num-epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--temperature", type=float, default=0.07, help="Temperature for loss")
    parser.add_argument(
        "--no-popularity-correction",
        action="store_true",
        help="Disable popularity correction",
    )
    parser.add_argument("--checkpoint-dir", default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")

    args = parser.parse_args()
    train(
        data_dir=args.data_dir,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        temperature=args.temperature,
        use_popularity_correction=not args.no_popularity_correction,
        checkpoint_dir=args.checkpoint_dir,
        patience=args.patience,
    )
