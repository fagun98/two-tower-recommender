"""FAISS index building for fast retrieval."""

import argparse
from pathlib import Path

import faiss
import numpy as np
import torch

from two_tower.data import MovieLensDataset
from two_tower.model import TwoTowerModel


def build_faiss_index(
    data_dir: str = "data/ml-latest-small",
    checkpoint_path: str = "checkpoints/model.pth",
    index_path: str = "checkpoints/items.faiss",
):
    """Build FAISS index from item embeddings."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    print("Loading data...")
    dataset = MovieLensDataset(data_dir)
    dataset.load()
    dataset.split_leave_last_out()

    # Load model
    print("Loading model...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = TwoTowerModel(
        num_users=checkpoint["num_users"],
        num_items=checkpoint["num_items"],
        num_genres=checkpoint["num_genres"],
        embedding_dim=checkpoint["embedding_dim"],
        hidden_dim=checkpoint["hidden_dim"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Generate embeddings for all items
    print("Generating item embeddings...")
    genre_features = torch.FloatTensor(dataset.get_item_features()).to(device)
    year_features = torch.LongTensor(dataset.get_item_years()).to(device)
    all_item_ids = torch.arange(dataset.get_num_items(), device=device)

    with torch.no_grad():
        item_embeddings = model.encode_item(all_item_ids, genre_features, year_features)
        item_embeddings = item_embeddings.cpu().numpy()

    # Build FAISS index (using inner product for normalized vectors)
    print("Building FAISS index...")
    dimension = item_embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(item_embeddings)

    # Save index and embeddings
    output_path = Path(index_path)
    output_path.parent.mkdir(exist_ok=True)

    faiss.write_index(index, str(output_path))
    np.save(output_path.parent / "embeddings.npy", item_embeddings)

    # Save item IDs mapping
    item_ids = dataset.get_all_item_ids()
    np.save(output_path.parent / "item_ids.npy", item_ids)

    print(f"FAISS index saved to {output_path}")
    print(f"Item embeddings shape: {item_embeddings.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build FAISS index")
    parser.add_argument("--data-dir", default="data/ml-latest-small", help="Data directory")
    parser.add_argument(
        "--checkpoint-path", default="checkpoints/model.pth", help="Model checkpoint path"
    )
    parser.add_argument(
        "--index-path", default="checkpoints/items.faiss", help="Output index path"
    )

    args = parser.parse_args()
    build_faiss_index(**vars(args))
