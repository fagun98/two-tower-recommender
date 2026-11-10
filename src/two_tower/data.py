"""Data loading and preprocessing for MovieLens dataset."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class MovieLensDataset:
    """MovieLens dataset loader and preprocessor."""

    def __init__(self, data_dir: str = "data/ml-latest-small"):
        self.data_dir = Path(data_dir)
        self.ratings = None
        self.movies = None
        self.train_data = None
        self.test_data = None
        self.user_id_map = {}
        self.item_id_map = {}
        self.genre_columns = []

    def load(self):
        """Load ratings and movies data."""
        self.ratings = pd.read_csv(self.data_dir / "ratings.csv")
        self.movies = pd.read_csv(self.data_dir / "movies.csv")

        # Convert to implicit feedback (ratings >= 4 as positive)
        self.ratings = self.ratings[self.ratings["rating"] >= 4.0].copy()

        # Create ID mappings
        unique_users = sorted(self.ratings["userId"].unique())
        unique_items = sorted(self.ratings["movieId"].unique())

        self.user_id_map = {uid: idx for idx, uid in enumerate(unique_users)}
        self.item_id_map = {mid: idx for idx, mid in enumerate(unique_items)}

        self.ratings["user_idx"] = self.ratings["userId"].map(self.user_id_map)
        self.ratings["item_idx"] = self.ratings["movieId"].map(self.item_id_map)

        # Process genres into multi-hot encoding
        self._process_genres()

        print(f"Loaded {len(self.ratings)} positive interactions")
        print(f"Users: {len(self.user_id_map)}, Items: {len(self.item_id_map)}")

    def _process_genres(self):
        """Process movie genres into multi-hot encoding."""
        # Get all unique genres
        all_genres = set()
        for genres in self.movies["genres"]:
            all_genres.update(genres.split("|"))

        all_genres = sorted(all_genres - {"(no genres listed)"})
        self.genre_columns = all_genres

        # Create multi-hot encoding
        for genre in all_genres:
            self.movies[f"genre_{genre}"] = self.movies["genres"].str.contains(
                genre, regex=False
            ).astype(int)

        # Map movies to our item indices
        self.movies = self.movies[self.movies["movieId"].isin(self.item_id_map.keys())].copy()
        self.movies["item_idx"] = self.movies["movieId"].map(self.item_id_map)
        self.movies = self.movies.sort_values("item_idx").reset_index(drop=True)

    def split_leave_last_out(self):
        """Split data using leave-last-out strategy (temporal)."""
        # Sort by timestamp
        self.ratings = self.ratings.sort_values("timestamp")

        # For each user, leave last interaction for testing
        train_list = []
        test_list = []

        for user_idx, group in self.ratings.groupby("user_idx"):
            if len(group) > 1:
                train_list.append(group.iloc[:-1])
                test_list.append(group.iloc[-1:])
            else:
                # If user has only one interaction, put in train
                train_list.append(group)

        self.train_data = pd.concat(train_list, ignore_index=True)
        self.test_data = pd.concat(test_list, ignore_index=True)

        print(f"Train: {len(self.train_data)} interactions")
        print(f"Test: {len(self.test_data)} interactions")

    def get_num_users(self) -> int:
        return len(self.user_id_map)

    def get_num_items(self) -> int:
        return len(self.item_id_map)

    def get_num_genres(self) -> int:
        return len(self.genre_columns)

    def get_item_features(self) -> np.ndarray:
        """Get item genre features as numpy array."""
        genre_cols = [f"genre_{g}" for g in self.genre_columns]
        features = self.movies[genre_cols].values
        return features.astype(np.float32)

    def get_all_item_ids(self) -> np.ndarray:
        """Get all item IDs in index order."""
        return self.movies["movieId"].values


class InteractionDataset(Dataset):
    """PyTorch dataset for user-item interactions."""

    def __init__(self, ratings_df: pd.DataFrame):
        self.user_indices = torch.LongTensor(ratings_df["user_idx"].values)
        self.item_indices = torch.LongTensor(ratings_df["item_idx"].values)

    def __len__(self):
        return len(self.user_indices)

    def __getitem__(self, idx):
        return self.user_indices[idx], self.item_indices[idx]
