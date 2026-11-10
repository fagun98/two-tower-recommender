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
        """Process movie genres and years."""
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

        # Extract year from title (format: "Title (YYYY)")
        self.movies["year"] = (
            self.movies["title"].str.extract(r"\((\d{4})\)$")[0].astype(float)
        )
        # Fill missing years with median
        median_year = self.movies["year"].median()
        self.movies["year"] = self.movies["year"].fillna(median_year).astype(int)
        # Clamp years to reasonable range
        self.movies["year"] = self.movies["year"].clip(1900, 2030)

        # Map movies to our item indices
        self.movies = self.movies[self.movies["movieId"].isin(self.item_id_map.keys())].copy()
        self.movies["item_idx"] = self.movies["movieId"].map(self.item_id_map)
        self.movies = self.movies.sort_values("item_idx").reset_index(drop=True)

    def split_leave_last_out(self, val_ratio=0.1):
        """Split data using leave-last-out strategy with validation set.

        Args:
            val_ratio: Ratio of training users to use for validation
        """
        # Sort by timestamp
        self.ratings = self.ratings.sort_values("timestamp")

        # For each user, leave last interaction for testing
        train_list = []
        test_list = []
        val_list = []

        users = list(self.ratings["user_idx"].unique())
        np.random.seed(42)
        np.random.shuffle(users)
        val_users = set(users[: int(len(users) * val_ratio)])

        for user_idx, group in self.ratings.groupby("user_idx"):
            if len(group) > 1:
                if user_idx in val_users:
                    # Validation user: last item is validation
                    train_list.append(group.iloc[:-1])
                    val_list.append(group.iloc[-1:])
                else:
                    # Test user: last item is test
                    train_list.append(group.iloc[:-1])
                    test_list.append(group.iloc[-1:])
            else:
                # If user has only one interaction, put in train
                train_list.append(group)

        self.train_data = pd.concat(train_list, ignore_index=True)
        self.test_data = pd.concat(test_list, ignore_index=True)
        self.val_data = pd.concat(val_list, ignore_index=True) if val_list else pd.DataFrame()

        print(f"Train: {len(self.train_data)} interactions")
        print(f"Val: {len(self.val_data)} interactions")
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

    def get_item_years(self) -> np.ndarray:
        """Get item years as numpy array (offset from 1900)."""
        years = (self.movies["year"].values - 1900).astype(np.int64)
        return years

    def get_all_item_ids(self) -> np.ndarray:
        """Get all item IDs in index order."""
        return self.movies["movieId"].values

    def build_user_histories(self, max_history: int = 20) -> dict:
        """Build user interaction histories from training data.

        Args:
            max_history: Maximum number of items to keep per user

        Returns:
            Dictionary mapping user_idx to list of item_idx
        """
        user_histories = {}
        for user_idx, group in self.train_data.groupby("user_idx"):
            # Sort by timestamp and take most recent items
            items = group.sort_values("timestamp")["item_idx"].values[-max_history:]
            user_histories[user_idx] = items.tolist()
        return user_histories


class InteractionDataset(Dataset):
    """PyTorch dataset for user-item interactions with histories."""

    def __init__(self, ratings_df: pd.DataFrame, user_histories: dict = None, max_history: int = 20):
        self.user_indices = torch.LongTensor(ratings_df["user_idx"].values)
        self.item_indices = torch.LongTensor(ratings_df["item_idx"].values)
        self.user_histories = user_histories or {}
        self.max_history = max_history

    def __len__(self):
        return len(self.user_indices)

    def __getitem__(self, idx):
        user_idx = self.user_indices[idx].item()
        item_idx = self.item_indices[idx]

        # Get user history
        history = self.user_histories.get(user_idx, [])
        if len(history) == 0:
            # Padding: use zeros
            history_tensor = torch.zeros(1, dtype=torch.long)
        else:
            history_tensor = torch.LongTensor(history[-self.max_history:])

        return self.user_indices[idx], item_idx, history_tensor
