"""Two-tower recommendation model."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TwoTowerModel(nn.Module):
    """Two-tower neural recommendation model with user and item towers."""

    def __init__(
        self,
        num_users: int,
        num_items: int,
        num_genres: int,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim

        # User tower
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.user_fc = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

        # Item tower
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.genre_projection = nn.Linear(num_genres, embedding_dim // 2)
        self.item_fc = nn.Sequential(
            nn.Linear(embedding_dim + embedding_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def encode_user(self, user_ids: torch.Tensor) -> torch.Tensor:
        """Encode user IDs to L2-normalized embeddings.

        Args:
            user_ids: User IDs tensor of shape (batch_size,)

        Returns:
            L2-normalized user embeddings of shape (batch_size, embedding_dim)
        """
        user_emb = self.user_embedding(user_ids)
        user_out = self.user_fc(user_emb)
        return F.normalize(user_out, p=2, dim=1)

    def encode_item(self, item_ids: torch.Tensor, genre_features: torch.Tensor) -> torch.Tensor:
        """Encode item IDs and genre features to L2-normalized embeddings.

        Args:
            item_ids: Item IDs tensor of shape (batch_size,)
            genre_features: Genre multi-hot features of shape (batch_size, num_genres)

        Returns:
            L2-normalized item embeddings of shape (batch_size, embedding_dim)
        """
        item_emb = self.item_embedding(item_ids)
        genre_emb = self.genre_projection(genre_features)
        combined = torch.cat([item_emb, genre_emb], dim=1)
        item_out = self.item_fc(combined)
        return F.normalize(item_out, p=2, dim=1)

    def forward(
        self, user_ids: torch.Tensor, item_ids: torch.Tensor, genre_features: torch.Tensor
    ) -> tuple:
        """Forward pass returning user and item embeddings.

        Args:
            user_ids: User IDs tensor
            item_ids: Item IDs tensor
            genre_features: Genre features tensor

        Returns:
            Tuple of (user_embeddings, item_embeddings)
        """
        user_emb = self.encode_user(user_ids)
        item_emb = self.encode_item(item_ids, genre_features)
        return user_emb, item_emb


def in_batch_negative_loss(user_emb: torch.Tensor, item_emb: torch.Tensor) -> torch.Tensor:
    """Compute in-batch negative (sampled softmax) loss.

    This treats all other items in the batch as negative samples for each user.
    The loss encourages high similarity between positive user-item pairs and low
    similarity between users and non-interacted items in the batch.

    Args:
        user_emb: L2-normalized user embeddings (batch_size, embedding_dim)
        item_emb: L2-normalized item embeddings (batch_size, embedding_dim)

    Returns:
        Scalar loss value
    """
    # Compute all pairwise dot products (logits)
    # Shape: (batch_size, batch_size)
    logits = torch.matmul(user_emb, item_emb.t())

    # The diagonal contains positive pairs
    # We want to maximize these while minimizing off-diagonal
    labels = torch.arange(logits.size(0), device=logits.device)

    # Cross-entropy loss with in-batch negatives
    loss = F.cross_entropy(logits, labels)
    return loss
