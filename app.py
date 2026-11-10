"""Gradio demo app for two-tower recommender system."""

import sys
import time
from pathlib import Path

import faiss
import gradio as gr
import torch

sys.path.insert(0, str(Path(__file__).parent / "src"))

from two_tower.data import MovieLensDataset
from two_tower.model import TwoTowerModel


class RecommenderApp:
    """Recommendation system app."""

    def __init__(
        self,
        data_dir: str = "data/ml-latest-small",
        checkpoint_path: str = "checkpoints/model.pth",
        index_path: str = "checkpoints/items.faiss",
    ):
        self.device = torch.device("cpu")
        self.model = None
        self.index = None
        self.dataset = None
        self.item_id_to_idx = None
        self.idx_to_item_id = None
        self.movies_info = None

        self.data_dir = data_dir
        self.checkpoint_path = checkpoint_path
        self.index_path = index_path

        self._load_all()

    def _load_all(self):
        """Load model, index, and data."""
        print("Loading data...")
        self.dataset = MovieLensDataset(self.data_dir)
        self.dataset.load()
        self.dataset.split_leave_last_out(val_ratio=0.1)

        self.item_id_to_idx = self.dataset.item_id_map
        self.idx_to_item_id = {v: k for k, v in self.item_id_to_idx.items()}

        # Build user histories
        self.user_histories = self.dataset.build_user_histories(max_history=20)

        # Load movies info
        self.movies_info = self.dataset.movies.set_index("item_idx")

        print("Loading model...")
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.model = TwoTowerModel(
            num_users=checkpoint["num_users"],
            num_items=checkpoint["num_items"],
            num_genres=checkpoint["num_genres"],
            embedding_dim=checkpoint["embedding_dim"],
            hidden_dim=checkpoint["hidden_dim"],
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        print("Loading FAISS index...")
        self.index = faiss.read_index(str(self.index_path))

    def get_user_history(self, user_id: int, n: int = 10):
        """Get user's recent liked movies."""
        if user_id not in self.dataset.user_id_map:
            return None

        user_idx = self.dataset.user_id_map[user_id]
        user_items = self.dataset.train_data[self.dataset.train_data["user_idx"] == user_idx]
        user_items = user_items.sort_values("timestamp", ascending=False).head(n)

        history = []
        for _, row in user_items.iterrows():
            item_idx = row["item_idx"]
            movie_id = self.idx_to_item_id[item_idx]
            movie_info = self.movies_info.loc[item_idx]
            history.append(
                {
                    "title": movie_info["title"],
                    "genres": movie_info["genres"],
                    "movieId": int(movie_id),
                }
            )

        return history

    def recommend_for_user(self, user_id: int, k: int = 10):
        """Generate recommendations for a user."""
        if user_id not in self.dataset.user_id_map:
            return None, None

        user_idx = self.dataset.user_id_map[user_id]

        # Get seen items
        seen_items = set(
            self.dataset.train_data[self.dataset.train_data["user_idx"] == user_idx][
                "item_idx"
            ].values
        )

        # Get user history
        history = self.user_histories.get(user_idx, [])
        if len(history) == 0:
            history_tensor = torch.zeros(1, 1, dtype=torch.long)
        else:
            history_tensor = torch.LongTensor(history[-20:]).unsqueeze(0)

        # Get user embedding
        start_time = time.time()
        user_tensor = torch.LongTensor([user_idx])
        with torch.no_grad():
            user_emb = self.model.encode_user(user_tensor, history_tensor).numpy()

        # Search FAISS
        D, I = self.index.search(user_emb, k + len(seen_items))
        query_time = time.time() - start_time

        # Filter seen items
        recommendations = []
        for dist, item_idx in zip(D[0], I[0]):
            if item_idx not in seen_items:
                movie_id = self.idx_to_item_id[item_idx]
                movie_info = self.movies_info.loc[item_idx]
                recommendations.append(
                    {
                        "title": movie_info["title"],
                        "genres": movie_info["genres"],
                        "similarity": float(dist),
                        "movieId": int(movie_id),
                    }
                )
                if len(recommendations) >= k:
                    break

        return recommendations, query_time

    def recommend_cold_start(self, movie_ids: list, k: int = 10):
        """Generate recommendations based on selected movies (cold start)."""
        if not movie_ids:
            return None, None

        # Convert to item indices
        item_indices = [self.item_id_to_idx[mid] for mid in movie_ids if mid in self.item_id_to_idx]

        if not item_indices:
            return None, None

        # Get item embeddings and average them
        start_time = time.time()
        item_tensors = torch.LongTensor(item_indices)
        genre_features = torch.FloatTensor(self.dataset.get_item_features()[item_indices])
        year_features = torch.LongTensor(self.dataset.get_item_years()[item_indices])

        with torch.no_grad():
            item_embs = self.model.encode_item(item_tensors, genre_features, year_features)
            avg_emb = item_embs.mean(dim=0, keepdim=True).numpy()

        # Search FAISS
        D, I = self.index.search(avg_emb, k + len(item_indices))
        query_time = time.time() - start_time

        # Filter selected items
        seen_set = set(item_indices)
        recommendations = []
        for dist, item_idx in zip(D[0], I[0]):
            if item_idx not in seen_set:
                movie_id = self.idx_to_item_id[item_idx]
                movie_info = self.movies_info.loc[item_idx]
                recommendations.append(
                    {
                        "title": movie_info["title"],
                        "genres": movie_info["genres"],
                        "similarity": float(dist),
                        "movieId": int(movie_id),
                    }
                )
                if len(recommendations) >= k:
                    break

        return recommendations, query_time

    def find_similar_movies(self, movie_id: int, k: int = 10):
        """Find similar movies using item-item similarity."""
        if movie_id not in self.item_id_to_idx:
            return None, None

        item_idx = self.item_id_to_idx[movie_id]

        # Get item embedding
        start_time = time.time()
        item_tensor = torch.LongTensor([item_idx])
        genre_features = torch.FloatTensor(self.dataset.get_item_features()[[item_idx]])
        year_features = torch.LongTensor(self.dataset.get_item_years()[[item_idx]])

        with torch.no_grad():
            item_emb = self.model.encode_item(item_tensor, genre_features, year_features).numpy()

        # Search FAISS
        D, I = self.index.search(item_emb, k + 1)
        query_time = time.time() - start_time

        # Skip the first result (itself)
        similar_movies = []
        for dist, similar_idx in zip(D[0][1:], I[0][1:]):
            similar_movie_id = self.idx_to_item_id[similar_idx]
            movie_info = self.movies_info.loc[similar_idx]
            similar_movies.append(
                {
                    "title": movie_info["title"],
                    "genres": movie_info["genres"],
                    "similarity": float(dist),
                    "movieId": int(similar_movie_id),
                }
            )

        return similar_movies, query_time


def create_ui(app: RecommenderApp):
    """Create Gradio UI."""

    def format_movies_table(movies, show_similarity=False):
        """Format movies as a markdown table."""
        if not movies:
            return "No results found."

        lines = ["| # | Title | Genres | " + ("Similarity |" if show_similarity else "")]
        lines.append("|---|-------|--------|" + ("-----------|" if show_similarity else ""))

        for i, movie in enumerate(movies, 1):
            title = movie['title']
            genres = movie['genres']
            if show_similarity:
                sim = f"{movie['similarity']:.3f}"
                lines.append(f"| {i} | {title} | {genres} | {sim} |")
            else:
                lines.append(f"| {i} | {title} | {genres} |")

        return "\n".join(lines)

    def recommend_interface(user_id, top_k):
        if not user_id:
            return "Please enter a user ID.", "", ""

        try:
            user_id = int(user_id)
        except ValueError:
            return "Invalid user ID.", "", ""

        history = app.get_user_history(user_id)
        if history is None:
            return f"User ID {user_id} not found in dataset.", "", ""

        recs, query_time = app.recommend_for_user(user_id, top_k)

        history_text = "### What This User Liked\n\n" + format_movies_table(history)
        recs_text = "### Recommended for Them\n\n" + format_movies_table(recs, show_similarity=True)
        stats = f"⚡ FAISS query time: **{query_time*1000:.2f} ms**"

        return history_text, recs_text, stats

    def cold_start_interface(selected_titles, top_k):
        if not selected_titles:
            return "Please select at least one movie.", ""

        # Parse movie IDs from selected titles
        movie_ids = []
        for title in selected_titles:
            try:
                mid = int(title.split("(ID: ")[1].split(")")[0])
                movie_ids.append(mid)
            except:
                continue

        if not movie_ids:
            return "Could not parse movie IDs.", ""

        recs, query_time = app.recommend_cold_start(movie_ids, top_k)

        recs_text = format_movies_table(recs, show_similarity=True)
        stats = f"⚡ FAISS query time: **{query_time*1000:.2f} ms**"

        return recs_text, stats

    def similar_movies_interface(movie_title, top_k):
        if not movie_title:
            return "Please select a movie.", ""

        try:
            movie_id = int(movie_title.split("(ID: ")[1].split(")")[0])
        except:
            return "Could not parse movie ID.", ""

        similar, query_time = app.find_similar_movies(movie_id, top_k)

        similar_text = format_movies_table(similar, show_similarity=True)
        stats = f"⚡ FAISS query time: **{query_time*1000:.2f} ms**"

        return similar_text, stats

    # Get movie list for dropdowns
    all_movies = app.dataset.movies.copy()
    all_movies["display"] = (
        all_movies["title"] + " (ID: " + all_movies["movieId"].astype(str) + ")"
    )
    movie_choices = sorted(all_movies["display"].tolist())[:500]

    with gr.Blocks(title="Two-Tower Recommender", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🎬 Two-Tower Movie Recommender")
        gr.Markdown(
            "A neural recommendation system using two-tower architecture with FAISS for fast retrieval. "
            "**2x better than popularity baseline!**"
        )

        with gr.Tab("👤 User Recommendations"):
            gr.Markdown(
                "**Personalized recommendations based on viewing history.** "
                "The user tower encodes their recent likes into a single embedding, "
                "then FAISS finds the closest item embeddings in milliseconds."
            )
            with gr.Row():
                with gr.Column():
                    user_input = gr.Textbox(
                        label="User ID",
                        placeholder="Enter a user ID (e.g., 1, 10, 100)",
                        value="1",
                    )
                    user_topk = gr.Slider(5, 20, value=10, step=1, label="Top-K")
                    user_btn = gr.Button("Get Recommendations", variant="primary")

            with gr.Row():
                with gr.Column(scale=1):
                    user_history = gr.Markdown()
                with gr.Column(scale=1):
                    user_recs = gr.Markdown()

            user_stats = gr.Markdown()

            user_btn.click(
                recommend_interface,
                inputs=[user_input, user_topk],
                outputs=[user_history, user_recs, user_stats],
            )

        with gr.Tab("🆕 Cold Start"):
            gr.Markdown(
                "**No account? No problem!** "
                "Select a few movies you like, and we'll average their item embeddings to create a pseudo-user vector for instant recommendations."
            )
            with gr.Row():
                with gr.Column():
                    cold_movies = gr.Dropdown(
                        choices=movie_choices,
                        label="Select movies you like",
                        multiselect=True,
                        max_choices=10,
                    )
                    cold_topk = gr.Slider(5, 20, value=10, step=1, label="Top-K")
                    cold_btn = gr.Button("Get Recommendations", variant="primary")

            cold_recs = gr.Markdown()
            cold_stats = gr.Markdown()

            cold_btn.click(
                cold_start_interface,
                inputs=[cold_movies, cold_topk],
                outputs=[cold_recs, cold_stats],
            )

        with gr.Tab("🔍 Similar Movies"):
            gr.Markdown(
                "**Item-to-item similarity search.** "
                "Find movies with similar embeddings using FAISS inner product search—great for 'more like this' features."
            )
            with gr.Row():
                with gr.Column():
                    similar_movie = gr.Dropdown(
                        choices=movie_choices, label="Select a movie", value=movie_choices[0]
                    )
                    similar_topk = gr.Slider(5, 20, value=10, step=1, label="Top-K")
                    similar_btn = gr.Button("Find Similar", variant="primary")

            similar_recs = gr.Markdown()
            similar_stats = gr.Markdown()

            similar_btn.click(
                similar_movies_interface,
                inputs=[similar_movie, similar_topk],
                outputs=[similar_recs, similar_stats],
            )

    return demo


def setup_on_first_start():
    """Download data and train model on first start if needed."""
    import subprocess

    checkpoint_path = Path("checkpoints/model.pth")
    index_path = Path("checkpoints/items.faiss")
    data_path = Path("data/ml-latest-small")

    # Check if we need to set up
    if checkpoint_path.exists() and index_path.exists() and data_path.exists():
        return  # Already set up

    print("=" * 60)
    print("First-time setup detected!")
    print("Downloading data and training model (takes ~2 minutes)...")
    print("=" * 60)

    # Download data
    if not data_path.exists():
        print("\n[1/3] Downloading MovieLens dataset...")
        subprocess.run([sys.executable, "scripts/download_data.py"], check=True)

    # Train model
    if not checkpoint_path.exists():
        print("\n[2/3] Training two-tower model...")
        subprocess.run([
            sys.executable, "-m", "two_tower.train",
            "--num-epochs", "50",
            "--batch-size", "512",
            "--patience", "7"
        ], check=True)

    # Build FAISS index
    if not index_path.exists():
        print("\n[3/3] Building FAISS index...")
        subprocess.run([sys.executable, "-m", "two_tower.index"], check=True)

    print("\n" + "=" * 60)
    print("Setup complete! Starting Gradio app...")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    setup_on_first_start()
    app = RecommenderApp()
    demo = create_ui(app)
    demo.launch()
