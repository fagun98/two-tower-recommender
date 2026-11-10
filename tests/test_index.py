"""Tests for FAISS index."""

import faiss
import numpy as np


def test_faiss_returns_correct_topk():
    """Test that FAISS returns correct top-k results."""
    dim = 64
    num_items = 100
    k = 10

    # Create random normalized embeddings
    embeddings = np.random.randn(num_items, dim).astype(np.float32)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    # Build FAISS index
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    # Query with one embedding
    query = embeddings[0:1]
    D, I = index.search(query, k)

    # Check results
    assert I.shape == (1, k)
    assert D.shape == (1, k)
    assert I[0, 0] == 0  # First result should be itself
    assert D[0, 0] > 0.99  # Should have similarity close to 1


def test_seen_items_exclusion():
    """Test that we can correctly filter seen items."""
    dim = 64
    num_items = 100
    k = 10
    seen_items = {0, 5, 10, 15}

    # Create random normalized embeddings
    embeddings = np.random.randn(num_items, dim).astype(np.float32)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    # Build FAISS index
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    # Query
    query = embeddings[0:1]
    D, I = index.search(query, k + len(seen_items))

    # Filter seen items
    recommendations = [idx for idx in I[0] if idx not in seen_items][:k]

    assert len(recommendations) == k
    assert all(idx not in seen_items for idx in recommendations)


def test_faiss_inner_product():
    """Test FAISS inner product search."""
    dim = 64
    num_items = 50

    # Create embeddings with known similarities
    embeddings = np.random.randn(num_items, dim).astype(np.float32)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    # Query with first embedding
    query = embeddings[0:1]
    D, I = index.search(query, 5)

    # Verify distances are in descending order
    assert all(D[0][i] >= D[0][i + 1] for i in range(len(D[0]) - 1))

    # Verify we can reconstruct the distance
    manual_dist = np.dot(query[0], embeddings[I[0, 1]])
    assert np.isclose(D[0, 1], manual_dist, atol=1e-5)
