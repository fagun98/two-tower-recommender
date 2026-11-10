---
title: Two-Tower Movie Recommender
emoji: 🎬
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
---

# 🎬 Two-Tower Movie Recommender

A neural recommendation system using two-tower architecture with PyTorch and FAISS for fast retrieval. This project demonstrates industry-standard retrieval-based recommendation at scale.

[![CI](https://github.com/yourusername/two-tower-recommender/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/two-tower-recommender/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 What is a Two-Tower Model?

A **two-tower model** (also called dual encoder) is a neural architecture widely used in industry for large-scale retrieval systems (YouTube, Pinterest, TikTok, etc.). It consists of:

- **User Tower**: Encodes user information (ID, features) into a dense embedding
- **Item Tower**: Encodes item information (ID, metadata) into a dense embedding

Both towers output L2-normalized embeddings in the same vector space. Items similar to a user's preferences have embeddings close to the user's embedding (high cosine similarity = high dot product for normalized vectors).

### Why Two-Tower?

1. **Decoupling**: User and item embeddings can be computed independently
2. **Scalability**: Pre-compute all item embeddings offline, only encode user at query time
3. **Fast Retrieval**: Use approximate nearest neighbor search (FAISS) to find top-K items in milliseconds
4. **Production-Ready**: Standard architecture for candidate generation in recommendation pipelines

```mermaid
graph TB
    subgraph "Training"
        U[User ID] --> UT[User Tower<br/>Embedding + MLP]
        I[Item ID] --> IT[Item Tower<br/>Embedding + MLP]
        G[Genre Features] --> IT
        UT --> UE[User Embedding<br/>L2-normalized]
        IT --> IE[Item Embedding<br/>L2-normalized]
        UE --> Loss[In-Batch Negative Loss<br/>Contrastive Learning]
        IE --> Loss
    end
    
    subgraph "Serving"
        QU[Query User] --> QUT[User Tower]
        QUT --> QUE[User Embedding]
        AllItems[All Item Embeddings<br/>Pre-computed] --> FAISS[FAISS Index<br/>Inner Product Search]
        QUE --> FAISS
        FAISS --> TopK[Top-K Recommendations<br/>~1ms latency]
    end
    
    style Loss fill:#ff6b6b
    style TopK fill:#51cf66
    style FAISS fill:#339af0
```

## 🔥 In-Batch Negatives

**In-batch negative sampling** (also called sampled softmax) is an efficient training technique:

- For each positive user-item pair in a batch, all other items in that batch become negative samples
- With batch size 256, each user gets 1 positive and 255 negatives "for free"
- No need to explicitly sample negatives from the full catalog
- Proven effective in Google's "Sampling-Bias-Corrected Neural Modeling" paper

The loss is computed as cross-entropy over the similarity matrix, encouraging the model to rank the true positive item higher than in-batch negatives.

## 📊 Results

Evaluated on MovieLens latest-small dataset with leave-last-out temporal split:

| Model | Recall@10 | NDCG@10 |
|-------|-----------|---------|
| **Two-Tower** | **0.0280** | **0.0132** |
| Popularity Baseline | 0.0428 | 0.0228 |

**Note**: These are actual results from this codebase. The two-tower model underperforms the popularity baseline on this small dataset with leave-last-out evaluation. This is not uncommon for neural models on small datasets where:
- Limited training data makes it hard to learn generalizable patterns
- Leave-last-out evaluation can be particularly challenging for temporal data
- The model shows ~38% lower metrics than the baseline but demonstrates the complete retrieval pipeline

For production systems, two-tower models typically excel on larger datasets (millions of interactions) and benefit from additional features, side information, and more sophisticated training techniques. This project focuses on demonstrating the architecture and implementation rather than achieving state-of-the-art metrics on this small dataset.

The value here is in the **complete, production-ready pipeline**: data processing, model training, FAISS indexing, proper evaluation, and a working demo.

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- ~2GB disk space for data and models

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/two-tower-recommender.git
cd two-tower-recommender

# Install dependencies
pip install -r requirements.txt
```

### Download Data

```bash
python scripts/download_data.py
```

This downloads MovieLens latest-small dataset (~1MB) from GroupLens.

### Train Model

```bash
python -m two_tower.train \
    --embedding-dim 64 \
    --batch-size 256 \
    --num-epochs 10 \
    --learning-rate 0.001
```

Training takes ~3-5 minutes on CPU. The model checkpoint is saved to `checkpoints/model.pth`.

### Build FAISS Index

```bash
python -m two_tower.index
```

This generates embeddings for all items and builds a FAISS index (~5 seconds).

### Evaluate

```bash
python -m two_tower.evaluate
```

Computes Recall@10 and NDCG@10 on test set, compared against a popularity baseline.

### Launch Gradio Demo

```bash
python app.py
```

Open http://localhost:7860 in your browser.

## 🎮 Demo Features

The Gradio app provides three modes:

1. **User Recommendations**: Enter an existing user ID to see their history and personalized recommendations
2. **Cold Start**: Select movies you like to get recommendations without an account
3. **Similar Movies**: Find movies similar to one you select (item-to-item similarity)

All modes show FAISS query latency (typically <5ms).

### Screenshots

**User Recommendations:**
![User Recommendations](assets/demo_user_recs.png)

**Cold Start Mode:**
![Cold Start](assets/demo_cold_start.png)

**Similar Movies:**
![Similar Movies](assets/demo_similar_movies.png)

## 📁 Project Structure

```
.
├── src/two_tower/
│   ├── __init__.py
│   ├── data.py           # MovieLens data loading and preprocessing
│   ├── model.py          # Two-tower PyTorch model
│   ├── train.py          # Training loop with in-batch negatives
│   ├── index.py          # FAISS index building
│   └── evaluate.py       # Evaluation metrics (Recall@K, NDCG@K)
├── scripts/
│   └── download_data.py  # Data download script
├── tests/
│   ├── test_model.py     # Model architecture tests
│   ├── test_index.py     # FAISS retrieval tests
│   └── test_data.py      # Data processing tests
├── .github/
│   └── workflows/
│       └── ci.yml        # GitHub Actions CI
├── app.py                # Gradio demo application
├── requirements.txt      # Python dependencies
├── pyproject.toml        # Project configuration
├── LICENSE               # MIT License
└── README.md             # This file
```

## 🧪 Testing

Run tests with pytest:

```bash
pytest tests/ -v
```

Tests cover:
- Model output shapes and L2 normalization
- Training reduces loss on toy data
- FAISS returns correct top-K results
- Seen items are properly excluded from recommendations

## 🤗 Deploy to Hugging Face Spaces

This repo is ready to deploy as a Hugging Face Space:

1. **Create a Space**: Go to https://huggingface.co/new-space
2. **Choose Gradio SDK**: Select "Gradio" as the SDK
3. **Push this repo**:
   ```bash
   git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/SPACE_NAME
   git push hf main
   ```

The Space will automatically detect the Gradio app from the YAML front matter in this README.

**Important**: Make sure to train the model and commit the checkpoint before pushing:
```bash
# Train and generate artifacts
python scripts/download_data.py
python -m two_tower.train
python -m two_tower.index

# Commit model artifacts
git add checkpoints/
git commit -m "Add trained model checkpoint"
git push hf main
```

Alternatively, modify `app.py` to train on first launch (slower startup but no large files in repo).

## 📚 Learn More

- [Two-tower models at Pinterest](https://medium.com/pinterest-engineering/pinnersage-multi-modal-user-embedding-framework-for-recommendations-at-pinterest-bfd116b49475)
- [Sampling-Bias-Corrected Neural Modeling](https://research.google/pubs/pub48840/)
- [FAISS documentation](https://github.com/facebookresearch/faiss)
- [MovieLens dataset](https://grouplens.org/datasets/movielens/)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- MovieLens dataset by GroupLens Research
- PyTorch and FAISS teams for excellent tools
- Gradio for the easy-to-use demo framework

---

**Built with ❤️ using PyTorch, FAISS, and Gradio**
