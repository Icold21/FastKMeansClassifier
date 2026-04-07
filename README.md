# FastKMeansClassifier 🚀

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

An enterprise-grade, highly scalable Prototype-based Classifier built natively on **PyTorch**. 

Traditional clustering classifiers (like `NearestCentroid`) force exactly **1 centroid per class**, destroying multi-modal sub-clusters (e.g. if the "Technology" class contains "Hardware" and "AI", they get averaged into nonsense). `FastKMeansClassifier` spawns **multiple dynamic prototypes** per class, updating them efficiently via mini-batch K-Means, and supporting soft-assignment probability margins.

Optimized explicitly for **massive datasets, extreme dimensionality/sparsity (e.g., TF-IDF text NLP vectors), and tens of thousands of classes.**

## 🌟 Key Features

1. **Multi-Prototype Topology**: Learns $K$ prototypes per class (`k_init` or custom `k_list`). 
2. **PyTorch Native & GPU Accelerated**: Fully supports `float16` and mixed-precision Tensor Core acceleration to slice RAM requirements in half.
3. **Sparse Tensor Optimization**: Translates SciPy CSR matrices to PyTorch `SparseCOO` natively on the GPU. Includes intelligent fallbacks to bypass PyTorch's native C++ limitations with sparse `float16` matrices.
4. **Streaming / Online Learning (`fit_batch`)**: Ingest massive datasets in chunks. The model dynamically discovers and initializes unseen classes directly on the fly.
5. **Multithreaded CPU Orchestration**: Initialization (K-Means++) and Class Merging are dynamically dispatched across all CPU cores via ThreadPools.
6. **Adaptive Centroids**: Merges close centroids and prunes dead prototypes automatically to fit the intrinsic dimensionality of your dataset.
7. **Strict Sanity Checking**: Auto-validates labels to prevent silent failures from one-hot/probabilistic targets.

## 📦 Installation

Ensure you have your target PyTorch backend installed (CUDA, MPS, or CPU), then install dependencies:
```bash
pip install -r requirements.txt
```
*(Just drop `FastKMeansClassifier.py` into your repository!)*

## 🛠 Initialization Arguments

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `k_init` | `int` | `3` | Global default for the initial number of prototypes per class. |
| `k_list` | `List/Dict` | `None` | Custom prototypes per class. E.g. `{0: 5, 1: 10}` or `[5, 10]`. Overrides `k_init`. |
| `init_method` | `str` | `'kmeans++'`| Initialization algorithm: `'kmeans++'` or `'random'`. |
| `distance` | `str` | `'cosine'` | Metric to evaluate vectors: `'cosine'` or `'euclidean'`. |
| `dtype` | `str` | `'float16'` | Internal computational precision (`'float16'`, `'float32'`, etc.) |
| `soft` | `bool` | `True` | Use soft probabilities (True) or Hard K-Means boundaries (False). |
| `soft_type` | `str` | `'linear'` | Probabilistic projection: `'linear'` (ReLU) or `'softmax'`. |
| `temperature` | `float`| `1.0` | Scaling denominator if `soft_type='softmax'`. |
| `lambda_penalty`| `float`| `0.1` | Penalty applied to probabilities of assigning to an incorrect class. |
| `merge_threshold`| `float`| `None` | Distance threshold to merge similar prototypes of the same class. |
| `relative_merge`| `bool` | `False` | Treats `merge_threshold` as a fraction of the global mean distance. |
| `percentile_threshold`|`float`|`None`| Disables truncations/merges unless values also fall below this global distribution quantile. |
| `batch_size` | `int` | `10240` | Chunk size for distance computation and GPU feeding. |
| `n_threads` | `int` | `-1` | Threads utilized for class-wise parallelization (`-1` = all CPU cores). |

---

## 🚀 Code Examples

### 1. Standard Fitting (With GPU and bfloat16)
```python
import torch
from FastKMeansClassifier import FastKMeansClassifier

# Initialize model (Defaults to Float16 to save memory)
clf = FastKMeansClassifier(
    k_init=5, 
    distance='cosine',
    dtype='bfloat16',
    soft_type='softmax',
    merge_threshold=0.15, 
    relative_merge=True
)

# Cast to GPU for instant TensorCore execution
clf = clf.to(device='cuda')

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
    # Dynamically integrates new data and discovers unseen classes instantly!
    logs = stream_clf.fit_batch(X_batch, y_batch, verbose=False)
    
    print(f"Shift: {logs['shift']:.4f} | Centroids: {logs['active_centroids']}")
```

### 3. Custom Prototypes Per Class (`k_list`)
If you know some classes are highly diverse (e.g., "General Chat" vs "Strict FAQ"):
```python
# Class 0 gets 20 prototypes, Class 1 gets 2, Class 2 gets 5.
clf = FastKMeansClassifier(k_list={0: 20, 1: 2, 2: 5})
```

---

## 🧠 Architectural Philosophy & Asymptotics

Why use `FastKMeansClassifier` instead of Scikit-Learn's `NearestCentroid`, `SVC`, or `KNeighborsClassifier`?

When dealing with massive textual corpora (e.g., 500,000 documents, 100,000 TF-IDF features, 5,000 distinct classes), traditional algorithms fall apart:
- `KNeighborsClassifier (kNN)` stores the *entire dataset* in memory. Inference requires $O(N_{train} \times D)$ comparisons, rendering production streaming impossible.
- `SVC (Support Vector Machines)` scale quadratically $O(N_{train}^2)$, crashing on huge datasets.
- `NearestCentroid (Rocchio)` handles extreme data but forces exactly **1 centroid per class**, completely destroying variance.

### ⏱️ Mathematical Time Complexity

Let $E$ be the number of epochs, $N$ the total samples, $C$ the number of classes, $K_c$ the prototypes per class, $D$ the total features, and $N_{nnz}$ the number of non-zero elements in a Sparse batch.

- **Dense Computations:** $O(E \cdot N \cdot C \cdot K_c \cdot D)$
- **Sparse Computations:** PyTorch `SparseCOO` entirely bypasses $D$ (where 99% of values are zero in NLP). Complexity drops to $O(E \cdot N_{nnz} \cdot C \cdot K_c)$.

**Final Overall Algorithm Complexity:** 
$$\mathcal{O}(E \cdot N_{nnz} \cdot C \cdot K_c)$$

**Why is this the ultimate scaling solution?**  
The algorithm's time complexity is entirely decoupled from the quadratic burden of the sample size ($N^2$) and the massive empty dimensionality ($D$) of text embeddings. It strictly scales linearly alongside the raw active information ($N_{nnz}$) and your target prototypes ($C \times K_c$). Because Inference is compressed to merely checking $C \times K_c$ prototypes instead of $N_{train}$ samples, the model effortlessly processes millions of documents per seconds on minimal hardware while executing smoothly in default `float16` precision.