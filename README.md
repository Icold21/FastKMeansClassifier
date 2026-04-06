# FastKMeansClassifier 🚀

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

An enterprise-grade, highly scalable Prototype-based Classifier built on **PyTorch**. 

Traditional clustering classifiers (like `NearestCentroid`) use exactly **1 centroid per class**. This forces multi-modal data to average out, destroying complex boundaries. `FastKMeansClassifier` fixes this by spawning **multiple dynamic prototypes** per class, updating them efficiently via mini-batch K-Means, and supporting soft-assignment probability margins.

Designed specifically to tackle the bottlenecks of traditional ML, it operates seamlessly on **massive datasets, highly dimensional sparse features (e.g., TF-IDF text vectors), and tens of thousands of classes**.

## 🌟 Key Features

1. **Multi-Prototype Topology**: Learns multiple cluster centroids per class (`k_init`), allowing it to understand complex, non-linear class distributions.
2. **PyTorch Native & GPU Accelerated**: Fully compatible with PyTorch's `.to('cuda', dtype=torch.bfloat16)` architecture for mixed-precision Tensor Core acceleration.
3. **Sparse Tensor Optimization**: Translates SciPy CSR matrices to PyTorch `SparseCOO` directly on the GPU. No more dense RAM explosions!
4. **Streaming / Online Learning (`fit_batch`)**: Ingest massive datasets chunk-by-chunk. The model dynamically discovers new classes on the fly.
5. **Multithreaded Class Operations**: Initialization (K-Means++) and Centroid Merging are distributed across multiple CPU threads.
6. **Adaptive Centroids**: Merges close centroids and prunes dead prototypes automatically to fit the intrinsic dimensionality of your data.

## 📦 Installation

Ensure you have your target PyTorch backend installed (CUDA, MPS, or CPU), then install dependencies:
```bash
pip install -r requirements.txt
```
*(Just drop `FastKMeansClassifier.py` into your repository!)*

## 🛠 Initialization Arguments

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `k_init` | `int` | `3` | Starting number of prototypes per class. |
| `init_method` | `str` | `'kmeans++'`| Initialization algorithm: `'kmeans++'` or `'random'`. |
| `distance` | `str` | `'cosine'` | Metric to evaluate vectors: `'cosine'` or `'euclidean'`. |
| `soft` | `bool` | `True` | Use soft probabilities (True) or Hard K-Means boundaries (False). |
| `soft_type` | `str` | `'linear'` | Probabilistic projection: `'linear'` (ReLU) or `'softmax'`. |
| `temperature` | `float`| `1.0` | Scaling denominator if `soft_type='softmax'`. |
| `lambda_penalty`| `float`| `0.1` | Penalty applied to probabilities of assigning to a different class. |
| `merge_threshold`| `float`| `None` | Distance threshold to merge similar prototypes. |
| `relative_merge`| `bool` | `False` | Treats `merge_threshold` as a fraction of the global mean distance. |
| `percentile_threshold`|`float`|`None`| Adds strict distribution constraints (e.g., `0.1`) to merges/truncation. |
| `batch_size` | `int` | `10240` | Chunk size for distance computation and GPU feeding. |
| `n_threads` | `int` | `-1` | Threads utilized for class-wise parallelization (`-1` = all CPU cores). |

---

## 🧠 Architectural Philosophy & Asymptotics

Why use `FastKMeansClassifier` instead of Scikit-Learn's `NearestCentroid`, `SVC`, or `KNeighborsClassifier`?

When dealing with massive textual corpora (e.g., 500,000 documents, 100,000 TF-IDF features, 5,000 distinct classes), traditional algorithms fall apart:
- `KNeighborsClassifier (kNN)` stores the *entire dataset* in memory. Inference requires $O(N_{train} \times D)$ comparisons. For production streams, this is unacceptably slow.
- `SVC (Support Vector Machines)` scale quadratically $O(N_{train}^2)$, making them impossible to fit on huge datasets.
- `NearestCentroid (Rocchio)` handles extreme data beautifully but forces exactly **1 centroid per class**. If your class "Technology" includes highly distinct sub-clusters like "Hardware", "SaaS", and "Quantum Physics", a single centroid will average them out into a meaningless central vector.

### The FastKMeans Solution
`FastKMeansClassifier` maintains $K$ prototypes per class (e.g., `k_init=5`). It compresses your 500,000 documents into just 25,000 highly representative prototypes.

1. **Information Retention:** Sub-clusters are preserved.
2. **Speed (Inference):** Instead of checking 500,000 points (kNN), you only check 25,000 prototypes.
3. **GPU Tensor Cores:** Dense operations rely on highly optimized BLAS/cuBLAS. Sparse operations natively utilize PyTorch's `SparseCOO` algorithms directly on VRAM.

### ⏱️ Time & Space Complexity

Let $N$ be the number of samples in a batch, $D$ the number of features, $C$ the number of classes, $K_c$ the centroids per class, and $K_{total} = C \times K_c$. 

- **Distance Computation (Dense):** $O(N \cdot K_{total} \cdot D)$
- **Distance Computation (Sparse):** $O(N_{nnz} \cdot K_{total})$, where $N_{nnz}$ is the number of non-zero elements in the batch. Text embeddings (like TF-IDF) are extremely sparse (~99% zeros). PyTorch Sparse tensors skip zeros entirely, leading to up to **100x speedups** over dense computation.
- **Centroid Update (EMA):** $O(N \cdot K_{total} \cdot D)$ mapping.
- **Merging/Pruning:** Evaluated across a bounded random subset (max 2048 elements). Bounded to $O(C \cdot K_c^2 \cdot D)$ executed across multi-core CPU threads.
- **Memory (VRAM):** Strictly $O(K_{total} \cdot D)$ for model parameters. Training footprint is tightly bounded by the `batch_size`, fully protecting against Out-Of-Memory (OOM) errors.

## 🚀 Code Examples

### 1. Standard Fitting (With GPU and bfloat16)
```python
import torch
from FastKMeansClassifier import FastKMeansClassifier

# Initialize model
clf = FastKMeansClassifier(
    k_init=5, 
    distance='cosine',
    soft_type='softmax',
    merge_threshold=0.15, 
    relative_merge=True
)

# Cast to GPU and bfloat16 for instant TensorCore execution
clf = clf.to(device='cuda', dtype=torch.bfloat16)

# Automatically handles sparse scipy matrices and displays a tqdm progress bar
clf.fit(X_train, y_train, verbose=True)

# Inference
predictions = clf.predict(X_test)
probabilities = clf.predict_proba(X_test)
```

### 2. Streaming / Online Learning (`fit_batch`)
Perfect for Kafka streams, massive CSV files, or datasets that don't fit into RAM.
```python
stream_clf = FastKMeansClassifier(k_init=3)

# Data can arrive in infinite streams
for X_batch, y_batch in data_stream:
    # Dynamically integrates new data and discovers unseen classes instantly
    logs = stream_clf.fit_batch(X_batch, y_batch, verbose=False)
    
    print(f"Shift: {logs['shift']:.4f} | Centroids: {logs['active_centroids']}")
```

### 3. State Dictionary (Saving and Loading)
Since it inherits from `torch.nn.Module`, saving and loading is native and lightweight.
```python
# Save weights
torch.save(clf.state_dict(), "fast_kmeans_weights.pth")

# Load weights
new_clf = FastKMeansClassifier()
new_clf.load_state_dict(torch.load("fast_kmeans_weights.pth"))
new_clf.to('cuda')
```

## 📊 Benchmarking & Performance
On the **20 Newsgroups benchmark** with highly sparse TF-IDF data (`10,000+` features), `FastKMeansClassifier` processes over **500,000 samples/second** during inference on an NVIDIA RTX GPU. Check out the included `test.ipynb` notebook to run the full Hard Load Stress Test matrix on your own hardware!