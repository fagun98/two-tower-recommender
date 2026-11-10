"""Tests for two-tower model."""

import torch

from two_tower.model import TwoTowerModel, in_batch_negative_loss


def test_model_output_shape():
    """Test that model outputs correct shapes."""
    num_users = 100
    num_items = 50
    num_genres = 20
    embedding_dim = 64
    batch_size = 16

    model = TwoTowerModel(num_users, num_items, num_genres, embedding_dim)

    user_ids = torch.randint(0, num_users, (batch_size,))
    item_ids = torch.randint(0, num_items, (batch_size,))
    genre_features = torch.rand(batch_size, num_genres)
    years = torch.randint(0, 130, (batch_size,))
    user_histories = torch.randint(0, num_items, (batch_size, 5))

    user_emb, item_emb = model(user_ids, item_ids, genre_features, years, user_histories)

    assert user_emb.shape == (batch_size, embedding_dim)
    assert item_emb.shape == (batch_size, embedding_dim)


def test_embedding_normalization():
    """Test that embeddings are L2-normalized."""
    num_users = 100
    num_items = 50
    num_genres = 20
    embedding_dim = 64
    batch_size = 16

    model = TwoTowerModel(num_users, num_items, num_genres, embedding_dim)

    user_ids = torch.randint(0, num_users, (batch_size,))
    item_ids = torch.randint(0, num_items, (batch_size,))
    genre_features = torch.rand(batch_size, num_genres)
    years = torch.randint(0, 130, (batch_size,))
    user_histories = torch.randint(0, num_items, (batch_size, 5))

    user_emb, item_emb = model(user_ids, item_ids, genre_features, years, user_histories)

    # Check L2 norms are close to 1
    user_norms = torch.norm(user_emb, p=2, dim=1)
    item_norms = torch.norm(item_emb, p=2, dim=1)

    assert torch.allclose(user_norms, torch.ones_like(user_norms), atol=1e-5)
    assert torch.allclose(item_norms, torch.ones_like(item_norms), atol=1e-5)


def test_loss_computation():
    """Test that loss is computed correctly."""
    batch_size = 32
    embedding_dim = 64

    user_emb = torch.randn(batch_size, embedding_dim)
    item_emb = torch.randn(batch_size, embedding_dim)

    # Normalize
    user_emb = torch.nn.functional.normalize(user_emb, p=2, dim=1)
    item_emb = torch.nn.functional.normalize(item_emb, p=2, dim=1)

    loss = in_batch_negative_loss(user_emb, item_emb)

    assert loss.item() > 0
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)


def test_training_reduces_loss():
    """Test that a training step reduces loss."""
    num_users = 100
    num_items = 50
    num_genres = 20
    embedding_dim = 32
    batch_size = 16

    model = TwoTowerModel(num_users, num_items, num_genres, embedding_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    user_ids = torch.randint(0, num_users, (batch_size,))
    item_ids = torch.randint(0, num_items, (batch_size,))
    genre_features = torch.rand(batch_size, num_genres)
    years = torch.randint(0, 130, (batch_size,))
    user_histories = torch.randint(0, num_items, (batch_size, 5))

    # Compute initial loss
    model.eval()
    with torch.no_grad():
        user_emb, item_emb = model(user_ids, item_ids, genre_features, years, user_histories)
        initial_loss = in_batch_negative_loss(user_emb, item_emb).item()

    # Train for a few steps
    model.train()
    for _ in range(10):
        optimizer.zero_grad()
        user_emb, item_emb = model(user_ids, item_ids, genre_features, years, user_histories)
        loss = in_batch_negative_loss(user_emb, item_emb)
        loss.backward()
        optimizer.step()

    # Compute final loss
    model.eval()
    with torch.no_grad():
        user_emb, item_emb = model(user_ids, item_ids, genre_features, years, user_histories)
        final_loss = in_batch_negative_loss(user_emb, item_emb).item()

    assert final_loss < initial_loss
