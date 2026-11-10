"""Evaluation metrics for recommendation system."""

import argparse
from collections import defaultdict

import faiss
import numpy as np
import torch
from tqdm import tqdm

from two_tower.data import MovieLensDataset
from two_tower.model import TwoTowerModel


def compute_recall_at_k(predictions, ground_truth, k=10):
    """Compute Recall@K."""
    hits = 0
    total = 0

    for user_id, true_items in ground_truth.items():
        if user_id not in predictions:
            continue

        pred_items = predictions[user_id][:k]
        hits += len(set(pred_items) & set(true_items))
        total += len(true_items)

    return hits / total if total > 0 else 0.0


def compute_ndcg_at_k(predictions, ground_truth, k=10):
    """Compute NDCG@K."""
    ndcg_sum = 0.0
    num_users = 0

    for user_id, true_items in ground_truth.items():
        if user_id not in predictions:
            continue

        pred_items = predictions[user_id][:k]
        true_items_set = set(true_items)

        dcg = 0.0
        for i, item in enumerate(pred_items):
            if item in true_items_set:
                dcg += 1.0 / np.log2(i + 2)

        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(true_items), k)))

        ndcg_sum += dcg / idcg if idcg > 0 else 0.0
        num_users += 1

    return ndcg_sum / num_users if num_users > 0 else 0.0


def evaluate(
    data_dir: str = "data/ml-latest-small",
    checkpoint_path: str = "checkpoints/model.pth",
    index_path: str = "checkpoints/items.faiss",
    k: int = 10,
):
    """Evaluate the model."""
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

    # Load FAISS index
    print("Loading FAISS index...")
    index = faiss.read_index(str(index_path))

    # Build ground truth
    ground_truth = defaultdict(list)
    for _, row in dataset.test_data.iterrows():
        ground_truth[row["user_idx"]].append(row["item_idx"])

    # Build training set for filtering
    train_items_by_user = defaultdict(set)
    for _, row in dataset.train_data.iterrows():
        train_items_by_user[row["user_idx"]].add(row["item_idx"])

    # Generate predictions
    print("Generating predictions...")
    predictions = {}

    test_users = sorted(ground_truth.keys())
    for user_idx in tqdm(test_users, desc="Evaluating"):
        user_tensor = torch.LongTensor([user_idx]).to(device)

        with torch.no_grad():
            user_emb = model.encode_user(user_tensor)
            user_emb = user_emb.cpu().numpy()

        # Search FAISS
        D, I = index.search(user_emb, k + len(train_items_by_user[user_idx]))

        # Filter out training items
        seen_items = train_items_by_user[user_idx]
        pred_items = [item_idx for item_idx in I[0] if item_idx not in seen_items][:k]

        predictions[user_idx] = pred_items

    # Compute metrics
    recall = compute_recall_at_k(predictions, ground_truth, k=k)
    ndcg = compute_ndcg_at_k(predictions, ground_truth, k=k)

    print(f"\n{'='*50}")
    print("Two-Tower Model Results:")
    print(f"Recall@{k}: {recall:.4f}")
    print(f"NDCG@{k}: {ndcg:.4f}")
    print(f"{'='*50}\n")

    # Popularity baseline
    print("Computing popularity baseline...")
    item_counts = dataset.train_data["item_idx"].value_counts()
    popular_items = item_counts.index.tolist()

    baseline_predictions = {}
    for user_idx in test_users:
        seen_items = train_items_by_user[user_idx]
        pred_items = [item for item in popular_items if item not in seen_items][:k]
        baseline_predictions[user_idx] = pred_items

    baseline_recall = compute_recall_at_k(baseline_predictions, ground_truth, k=k)
    baseline_ndcg = compute_ndcg_at_k(baseline_predictions, ground_truth, k=k)

    print(f"{'='*50}")
    print("Popularity Baseline Results:")
    print(f"Recall@{k}: {baseline_recall:.4f}")
    print(f"NDCG@{k}: {baseline_ndcg:.4f}")
    print(f"{'='*50}\n")

    print("Improvement over baseline:")
    print(f"Recall@{k}: {(recall - baseline_recall) / baseline_recall * 100:+.1f}%")
    print(f"NDCG@{k}: {(ndcg - baseline_ndcg) / baseline_ndcg * 100:+.1f}%")

    return {
        "model_recall": recall,
        "model_ndcg": ndcg,
        "baseline_recall": baseline_recall,
        "baseline_ndcg": baseline_ndcg,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate recommendation model")
    parser.add_argument("--data-dir", default="data/ml-latest-small", help="Data directory")
    parser.add_argument(
        "--checkpoint-path", default="checkpoints/model.pth", help="Model checkpoint path"
    )
    parser.add_argument("--index-path", default="checkpoints/items.faiss", help="Index path")
    parser.add_argument("--k", type=int, default=10, help="Top-k for evaluation")

    args = parser.parse_args()
    evaluate(**vars(args))
