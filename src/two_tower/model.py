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
        max_year: int = 2020,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim

        # User tower - now takes item embeddings instead of user ID
        self.item_embedding_for_user = nn.Embedding(num_items, embedding_dim)
        self.user_id_embedding = nn.Embedding(num_users, embedding_dim // 2)
        self.user_fc = nn.Sequential(
            nn.Linear(embedding_dim + embedding_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embedding_dim),
        )

        # Item tower with genre and year
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.genre_projection = nn.Linear(num_genres, embedding_dim // 2)
        self.year_embedding = nn.Embedding(max_year - 1900 + 10, embedding_dim // 4)
        self.item_fc = nn.Sequential(
            nn.Linear(embedding_dim + embedding_dim // 2 + embedding_dim // 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
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

    def encode_user(
        self, user_ids: torch.Tensor, user_item_histories: torch.Tensor = None
    ) -> torch.Tensor:
        """Encode user with mean-pooled item history.

        Args:
            user_ids: User IDs tensor of shape (batch_size,)
            user_item_histories: User's item history (batch_size, seq_len) or None

        Returns:
            L2-normalized user embeddings of shape (batch_size, embedding_dim)
        """
        user_id_emb = self.user_id_embedding(user_ids)

        if user_item_histories is not None and user_item_histories.size(1) > 0:
            # Mean pool the item embeddings
            item_embs = self.item_embedding_for_user(user_item_histories)
            user_history_emb = item_embs.mean(dim=1)
        else:
            # Fallback: use zeros
            user_history_emb = torch.zeros(
                user_ids.size(0), self.embedding_dim, device=user_ids.device
            )

        combined = torch.cat([user_history_emb, user_id_emb], dim=1)
        user_out = self.user_fc(combined)
        return F.normalize(user_out, p=2, dim=1)

    def encode_item(
        self, item_ids: torch.Tensor, genre_features: torch.Tensor, years: torch.Tensor
    ) -> torch.Tensor:
        """Encode item IDs, genre features, and years to L2-normalized embeddings.

        Args:
            item_ids: Item IDs tensor of shape (batch_size,)
            genre_features: Genre multi-hot features of shape (batch_size, num_genres)
            years: Release years of shape (batch_size,)

        Returns:
            L2-normalized item embeddings of shape (batch_size, embedding_dim)
        """
        item_emb = self.item_embedding(item_ids)
        genre_emb = self.genre_projection(genre_features)
        year_emb = self.year_embedding(years)
        combined = torch.cat([item_emb, genre_emb, year_emb], dim=1)
        item_out = self.item_fc(combined)
        return F.normalize(item_out, p=2, dim=1)

    def forward(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        genre_features: torch.Tensor,
        years: torch.Tensor,
        user_item_histories: torch.Tensor = None,
    ) -> tuple:
        """Forward pass returning user and item embeddings.

        Args:
            user_ids: User IDs tensor
            item_ids: Item IDs tensor
            genre_features: Genre features tensor
            years: Release years tensor
            user_item_histories: User item history tensor (optional)

        Returns:
            Tuple of (user_embeddings, item_embeddings)
        """
        user_emb = self.encode_user(user_ids, user_item_histories)
        item_emb = self.encode_item(item_ids, genre_features, years)
        return user_emb, item_emb


def in_batch_negative_loss(
    user_emb: torch.Tensor,
    item_emb: torch.Tensor,
    temperature: float = 0.07,
    item_frequencies: torch.Tensor = None,
) -> torch.Tensor:
    """Compute in-batch negative loss with temperature scaling and optional popularity correction.

    Args:
        user_emb: L2-normalized user embeddings (batch_size, embedding_dim)
        item_emb: L2-normalized item embeddings (batch_size, embedding_dim)
        temperature: Temperature for scaling logits (lower = sharper)
        item_frequencies: Item popularity frequencies for logQ correction (optional)

    Returns:
        Scalar loss value
    """
    # Compute all pairwise dot products and scale by temperature
    logits = torch.matmul(user_emb, item_emb.t()) / temperature

    # Apply popularity correction if provided (logQ correction)
    if item_frequencies is not None:
        # Add log of item frequency to negatives to down-weight popular items
        log_freq = torch.log(item_frequencies + 1e-8)
        logits = logits - log_freq.unsqueeze(0)

    # The diagonal contains positive pairs
    labels = torch.arange(logits.size(0), device=logits.device)

    # Cross-entropy loss
    loss = F.cross_entropy(logits, labels)
    return loss
