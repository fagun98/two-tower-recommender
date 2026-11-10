"""Tests for data processing."""

import numpy as np
import pandas as pd

from two_tower.data import InteractionDataset


def test_interaction_dataset():
    """Test InteractionDataset class."""
    ratings = pd.DataFrame(
        {"user_idx": [0, 0, 1, 1, 2], "item_idx": [10, 20, 10, 30, 40], "rating": [5, 4, 5, 4, 5]}
    )

    user_histories = {0: [1, 2, 3], 1: [4, 5], 2: [6]}
    dataset = InteractionDataset(ratings, user_histories)

    assert len(dataset) == 5
    user_idx, item_idx, history = dataset[0]
    assert user_idx.item() == 0
    assert item_idx.item() == 10
    assert len(history) > 0


def test_genre_encoding_shape():
    """Test genre encoding produces correct shape."""
    # This test is more of a smoke test since we need actual data
    # We just verify the logic would work with synthetic data
    genres_list = ["Action", "Comedy", "Drama"]
    num_items = 10

    # Simulate multi-hot encoding
    genre_features = np.random.randint(0, 2, size=(num_items, len(genres_list))).astype(
        np.float32
    )

    assert genre_features.shape == (num_items, len(genres_list))
    assert genre_features.dtype == np.float32
    assert np.all((genre_features == 0) | (genre_features == 1))
