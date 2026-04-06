"""FastKMeansClassifier: A scalable, multi-prototype classifier for sparse and dense data.

This module provides a PyTorch-backed classifier that models each class using multiple
centroids via Soft K-Means. It is optimized for extremely large datasets, supporting
streaming capabilities, multithreaded initialization, and GPU-accelerated sparse matrix operations.
"""

import os
import logging
import concurrent.futures
from typing import Optional, Union, Any, Dict, Tuple

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import normalize

logger = logging.getLogger(__name__)


class FastKMeansClassifier(nn.Module, BaseEstimator, ClassifierMixin):
    """A highly scalable, GPU-accelerated Prototype-based Classifier using Soft K-Means.

    Designed for massive datasets, millions of samples, tens of thousands of classes,
    and highly sparse text embeddings (e.g., TF-IDF). It dynamically allocates multiple
    prototypes (centroids) per class to capture complex, multi-modal distributions.

    Attributes:
        centroids (torch.Tensor): The coordinates of all active centroids.
        centroid_labels (torch.Tensor): The class label associated with each centroid.
        centroid_weights (torch.Tensor): The accumulated mass (number of assigned samples) per centroid.
        classes_ (np.ndarray): Array of unique classes discovered during training.
    """

    def __init__(
        self,
        k_init: int = 3,
        init_method: str = 'kmeans++',
        distance: str = 'cosine',
        soft: bool = True,
        soft_type: str = 'linear',
        temperature: float = 1.0,
        lambda_penalty: float = 0.1,
        merge_threshold: Optional[float] = None,
        relative_merge: bool = False,
        min_weight: float = 1e-3,
        truncation_threshold: float = 1e-4,
        percentile_threshold: Optional[float] = None,
        max_iters: int = 50,
        tol: float = 1e-4,
        batch_size: int = 10240,
        n_threads: int = -1,
        random_state: int = 42
    ) -> None:
        """Initializes the FastKMeansClassifier with the specified hyperparameters.

        Args:
            k_init: Initial number of prototypes (centroids) per class.
            init_method: Strategy for initial centroid selection ('kmeans++' or 'random').
            distance: Distance metric to use ('cosine' or 'euclidean').
            soft: If True, uses soft probabilistic assignment; otherwise, hard assignment.
            soft_type: Method for probabilistic projection ('linear' via ReLU or 'softmax').
            temperature: Scaling factor for 'softmax' assignments (lower means harder assignments).
            lambda_penalty: Inter-class assignment penalty, applied when `soft=True`.
            merge_threshold: Distance threshold below which centroids of the same class are merged.
            relative_merge: If True, `merge_threshold` is evaluated as a fraction of the global mean distance.
            min_weight: Minimum accumulated mass required to keep a centroid alive (dead centroid pruning).
            truncation_threshold: Absolute values below this are zeroed out to prevent sparse array densification.
            percentile_threshold: If provided (e.g., 0.1 for 10%), truncation and merging require
                the target values to also fall below this global distribution quantile.
            max_iters: Maximum number of training epochs for standard batch fitting.
            tol: Convergence tolerance; stops training if maximum centroid shift is below this value.
            batch_size: Number of samples processed simultaneously. If None, processes the full dataset at once.
            n_threads: Number of CPU threads for class-parallel operations (-1 utilizes all available cores).
            random_state: Seed for random number generators to ensure reproducibility.

        Raises:
            ValueError: If `soft_type` or `init_method` contains unrecognized values.
        """
        super().__init__()
        self.k_init = k_init
        self.init_method = init_method.lower()
        self.distance = distance.lower()
        self.soft = soft
        self.soft_type = soft_type.lower()
        self.temperature = temperature
        self.lambda_penalty = lambda_penalty
        self.merge_threshold = merge_threshold
        self.relative_merge = relative_merge
        self.min_weight = min_weight
        self.truncation_threshold = truncation_threshold
        self.percentile_threshold = percentile_threshold
        self.max_iters = max_iters
        self.tol = tol
        self.batch_size = batch_size
        self.n_threads = os.cpu_count() if n_threads == -1 else max(1, n_threads)
        self.random_state = random_state

        if self.soft_type not in['linear', 'softmax']:
            raise ValueError("soft_type must be either 'linear' or 'softmax'.")
        if self.init_method not in ['kmeans++', 'random']:
            raise ValueError("init_method must be either 'kmeans++' or 'random'.")

        # Register PyTorch buffers to ensure parameters map to the correct device upon calling `.to(device)`
        self.register_buffer('centroids', torch.empty(0))
        self.register_buffer('centroid_labels', torch.empty(0, dtype=torch.long))
        self.register_buffer('centroid_weights', torch.empty(0))
        self.classes_ = np.array([])
        self._is_initialized = False

    def _scipy_to_torch_sparse(self, sp_mat: sp.spmatrix) -> torch.Tensor:
        """Safely converts a SciPy sparse matrix to a PyTorch SparseCOO tensor.

        Args:
            sp_mat: The input SciPy sparse matrix.

        Returns:
            A PyTorch SparseCOO tensor residing on the same device as the model's centroids.
        """
        coo_mat = sp_mat.tocoo()
        indices = torch.from_numpy(np.vstack((coo_mat.row, coo_mat.col))).to(
            dtype=torch.long, device=self.centroids.device
        )
        values = torch.from_numpy(coo_mat.data).to(
            dtype=self.centroids.dtype, device=self.centroids.device
        )
        return torch.sparse_coo_tensor(indices, values, size=coo_mat.shape).coalesce()

    def _cdist(self, X_batch: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        """Computes the similarity matrix between a batch of samples and the centroids.

        Args:
            X_batch: A dense or sparse PyTorch tensor of shape (N, D).
            C: The dense centroid tensor of shape (K, D).

        Returns:
            A similarity tensor of shape (N, K). Higher values indicate closer proximity.
        """
        sim = torch.sparse.mm(X_batch, C.t()) if X_batch.is_sparse else torch.mm(X_batch, C.t())
        if self.distance == 'cosine':
            return sim

        # Resolve Euclidean distance using the geometric expansion: (X - C)^2 = X^2 + C^2 - 2XC
        if X_batch.is_sparse:
            sq_values = X_batch.values() ** 2
            X_batch_sq = torch.sparse_coo_tensor(X_batch.indices(), sq_values, X_batch.shape)
            ones = torch.ones((X_batch.shape[1], 1), dtype=X_batch.dtype, device=X_batch.device)
            x2 = torch.sparse.mm(X_batch_sq, ones)
        else:
            x2 = torch.sum(X_batch ** 2, dim=1, keepdim=True)

        c2 = torch.sum(C ** 2, dim=1)
        dist = torch.clamp(x2 + c2 - 2 * sim, min=0.0)
        
        # Invert distance to represent similarity in range (0, 1]
        return 1.0 / (1.0 + dist)

    def _init_single_class(self, X_c_raw: Any, is_sp: bool, class_label: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Core initialization routine for isolating starting centroids of a single class.

        Args:
            X_c_raw: The raw input data subset belonging to the target class.
            is_sp: Boolean flag indicating whether the data is a SciPy sparse matrix.
            class_label: The integer label identifying the class.

        Returns:
            A tuple containing the initialized centroid coordinates and their corresponding labels.
        """
        n_samples = X_c_raw.shape[0]
        k = min(self.k_init, n_samples)

        # Random Initialization
        if self.init_method == 'random':
            indices = np.random.choice(n_samples, k, replace=False)
            centers_raw = X_c_raw[indices]
            if is_sp:
                centers_raw = centers_raw.toarray()
            centers = torch.as_tensor(centers_raw, dtype=self.centroids.dtype, device=self.centroids.device)
            labels = torch.full((k,), class_label, dtype=torch.long, device=self.centroids.device)
            return centers, labels

        # K-Means++ Initialization
        X_c_torch = self._scipy_to_torch_sparse(X_c_raw) if is_sp else torch.as_tensor(X_c_raw, dtype=self.centroids.dtype, device=self.centroids.device)
        first_idx = np.random.randint(0, n_samples)
        
        # Slicing [i:i+1] preserves the 2D dimensional shape (1, Features) required by PyTorch
        center_raw = X_c_raw[first_idx:first_idx + 1]
        if is_sp:
            center_raw = center_raw.toarray()

        centers = torch.as_tensor(center_raw, dtype=self.centroids.dtype, device=self.centroids.device)

        for _ in range(1, k):
            sim = self._cdist(X_c_torch, centers)
            dists = 1.0 - sim if self.distance == 'cosine' else 1.0 / sim - 1.0
            min_dists = torch.min(dists, dim=1)[0].clamp(min=0.0)

            probs = (min_dists ** 2).cpu().numpy()
            sum_probs = probs.sum()

            if sum_probs > 0:
                next_idx = np.random.choice(n_samples, p=probs / sum_probs)
            else:
                next_idx = np.random.choice(n_samples)

            new_center_raw = X_c_raw[next_idx:next_idx + 1]
            if is_sp:
                new_center_raw = new_center_raw.toarray()

            new_center = torch.as_tensor(new_center_raw, dtype=self.centroids.dtype, device=self.centroids.device)
            centers = torch.cat([centers, new_center], dim=0)

        labels = torch.full((k,), class_label, dtype=torch.long, device=self.centroids.device)
        return centers, labels

    def _initialize_new_classes(self, X: Any, y: torch.Tensor, is_sp: bool) -> None:
        """Identifies unseen classes in the input data and initializes prototypes for them via ThreadPools.

        Args:
            X: Input feature matrix.
            y: Target label tensor.
            is_sp: Flag indicating if X is a SciPy sparse matrix.
        """
        np.random.seed(self.random_state)
        y_np = y.cpu().numpy()
        unique_classes_in_batch = np.unique(y_np)
        new_classes = np.setdiff1d(unique_classes_in_batch, self.classes_)

        if len(new_classes) == 0:
            return

        def process_class(c: int) -> Tuple[torch.Tensor, torch.Tensor]:
            idx = np.where(y_np == c)[0]
            X_c_raw = X[idx]
            return self._init_single_class(X_c_raw, is_sp, int(c))

        new_centroids, new_labels = [],[]

        # Dispatch initialization across CPU threads for rapid processing of many classes
        if self.n_threads > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.n_threads) as executor:
                results = list(executor.map(process_class, new_classes))
        else:
            results =[process_class(c) for c in new_classes]

        for centers, labels in results:
            new_centroids.append(centers)
            new_labels.append(labels)

        new_centroids_tensor = torch.cat(new_centroids, dim=0)
        new_labels_tensor = torch.cat(new_labels, dim=0)
        new_weights = torch.ones(len(new_centroids_tensor), dtype=self.centroids.dtype, device=self.centroids.device)

        if len(self.centroids) == 0:
            self.centroids = new_centroids_tensor
            self.centroid_labels = new_labels_tensor
            self.centroid_weights = new_weights
        else:
            self.centroids = torch.cat([self.centroids, new_centroids_tensor])
            self.centroid_labels = torch.cat([self.centroid_labels, new_labels_tensor])
            self.centroid_weights = torch.cat([self.centroid_weights, new_weights])

        if self.distance == 'cosine':
            self.centroids = F.normalize(self.centroids, p=2, dim=1)

        self.classes_ = np.concatenate([self.classes_, new_classes])
        self._is_initialized = True

    def _format_input(self, X: Any, is_sp: bool) -> Any:
        """Validates and formats the input data structure. Normalizes features if Cosine distance is requested.

        Args:
            X: The input data matrix.
            is_sp: Flag indicating if the input is a SciPy sparse format.

        Returns:
            The correctly formatted (and potentially normalized) data matrix.
        """
        if is_sp and not sp.isspmatrix_csr(X):
            X = X.tocsr()
        if self.distance == 'cosine':
            if is_sp or isinstance(X, np.ndarray):
                X = normalize(X, norm='l2', axis=1)
            elif isinstance(X, torch.Tensor):
                X = F.normalize(X, p=2, dim=1)
        return X

    def fit_batch(self, X: Any, y: Any, verbose: bool = False) -> Dict[str, float]:
        """Processes a single batch of data, updating the prototypes (Streaming K-Means step).

        Dynamically discovers and initializes new classes if they are present in the batch.
        Updates centroids via Exponential Moving Average (EMA) and performs pruning and truncation.

        Args:
            X: Input feature batch (SciPy sparse, NumPy array, or PyTorch tensor).
            y: Target label batch.
            verbose: If True, logs the processing metrics for the batch.

        Returns:
            A dictionary containing internal tracking metrics:
            - 'shift': The maximum distance any centroid moved during the update.
            - 'num_merged': The count of redundant centroids merged in this step.
            - 'active_centroids': Total active centroids retained in memory.
        """
        is_sp = sp.issparse(X)
        X = self._format_input(X, is_sp)
        y_tensor = torch.as_tensor(y, dtype=torch.long, device=self.centroids.device)

        self._initialize_new_classes(X, y_tensor, is_sp)

        X_batch = self._scipy_to_torch_sparse(X) if is_sp else torch.as_tensor(X, dtype=self.centroids.dtype, device=self.centroids.device)
        K_total = self.centroids.shape[0]

        sim = self._cdist(X_batch, self.centroids)

        # Assignment Probability Calculation
        if self.soft:
            mask_diff_class = (y_tensor.unsqueeze(1) != self.centroid_labels.unsqueeze(0)).to(self.centroids.dtype)
            if self.soft_type == 'linear':
                scores = F.relu(sim - self.lambda_penalty * mask_diff_class)
                sum_scores = scores.sum(dim=1, keepdim=True)
                
                # Fallback handler for completely unassigned (zero-mass) samples
                zero_mask = (sum_scores == 0).squeeze(1)
                if zero_mask.any():
                    max_idx = sim[zero_mask].argmax(dim=1)
                    scores[zero_mask] = F.one_hot(max_idx, num_classes=K_total).to(self.centroids.dtype)
                    sum_scores[zero_mask] = 1.0
                probs = scores / sum_scores
            elif self.soft_type == 'softmax':
                logits = (sim - self.lambda_penalty * mask_diff_class) / self.temperature
                probs = F.softmax(logits, dim=1)
        else:
            # Hard K-Means Assignment
            mask_same_class = (y_tensor.unsqueeze(1) == self.centroid_labels.unsqueeze(0))
            sim_masked = torch.where(mask_same_class, sim, torch.tensor(-float('inf'), dtype=self.centroids.dtype, device=self.centroids.device))
            probs = F.one_hot(torch.argmax(sim_masked, dim=1), num_classes=K_total).to(self.centroids.dtype)

        # EMA Centroid Coordinate Updating
        C_num_update = torch.sparse.mm(X_batch.t(), probs).t() if X_batch.is_sparse else torch.mm(X_batch.t(), probs).t()
        W_update = probs.sum(dim=0)

        valid_update = W_update > 0
        self.centroid_weights[valid_update] += W_update[valid_update]
        
        lr = W_update[valid_update] / self.centroid_weights[valid_update]
        
        old_centroids = self.centroids.clone()
        self.centroids[valid_update] = (
            (1 - lr.unsqueeze(1)) * self.centroids[valid_update] + 
            lr.unsqueeze(1) * (C_num_update[valid_update] / W_update[valid_update].unsqueeze(1))
        )

        # Dimensional Sparsification (Truncation)
        if self.truncation_threshold > 0:
            if self.percentile_threshold is not None:
                active_weights = torch.abs(self.centroids)[torch.abs(self.centroids) > 1e-6]
                q_val = torch.quantile(active_weights, self.percentile_threshold).item() if len(active_weights) > 0 else 0.0
                trunc_mask = (torch.abs(self.centroids) < self.truncation_threshold) & (torch.abs(self.centroids) < q_val)
            else:
                trunc_mask = torch.abs(self.centroids) < self.truncation_threshold
            self.centroids = torch.where(trunc_mask, torch.zeros_like(self.centroids), self.centroids)

        if self.distance == 'cosine':
            self.centroids = F.normalize(self.centroids, p=2, dim=1)

        shift = torch.norm(self.centroids - old_centroids, dim=1).max().item()

        # Dead Centroid Pruning
        valid_mask = (self.centroid_weights > self.min_weight)
        self.centroids = self.centroids[valid_mask]
        self.centroid_labels = self.centroid_labels[valid_mask]
        self.centroid_weights = self.centroid_weights[valid_mask]

        num_merged = self._merge() if self.merge_threshold is not None else 0

        logs = {
            'shift': shift,
            'num_merged': num_merged,
            'active_centroids': len(self.centroids)
        }

        if verbose:
            logger.info(f"Batch processed. Shift: {shift:.5f} | Merged: {num_merged} | Active: {len(self.centroids)}")
        
        return logs

    def fit(self, X: Any, y: Any, verbose: bool = False) -> 'FastKMeansClassifier':
        """Trains the classifier across multiple epochs on the provided dataset.

        Args:
            X: Training data matrix.
            y: Target label array.
            verbose: If True, renders a TQDM progress bar logging iterations and centroid shifts.

        Returns:
            The fitted instance of FastKMeansClassifier.
        """
        is_sp = sp.issparse(X)
        X = self._format_input(X, is_sp)
        y_tensor = torch.as_tensor(y, dtype=torch.long, device=self.centroids.device)

        self._initialize_new_classes(X, y_tensor, is_sp)
        
        N = X.shape[0]
        bs = self.batch_size if self.batch_size is not None else N

        epoch_iterator = range(self.max_iters)
        if verbose:
            epoch_iterator = tqdm(epoch_iterator, desc="Training Epochs")

        for it in epoch_iterator:
            max_shift_epoch = 0.0
            total_merged_epoch = 0
            
            for i in range(0, N, bs):
                X_batch_raw = X[i:i + bs]
                y_batch = y[i:i + bs]
                
                logs = self.fit_batch(X_batch_raw, y_batch, verbose=False)
                max_shift_epoch = max(max_shift_epoch, logs['shift'])
                total_merged_epoch += logs['num_merged']

            if verbose:
                epoch_iterator.set_postfix({
                    "Shift": f"{max_shift_epoch:.5f}", 
                    "Centroids": len(self.centroids)
                })

            if max_shift_epoch < self.tol and total_merged_epoch == 0:
                if verbose:
                    logger.info("Convergence reached.")
                break

        return self

    def _merge_single_class(self, c: int, actual_threshold: float, perc_dist: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Core execution logic for merging close centroids within a specific class."""
        idx = torch.nonzero(self.centroid_labels == c).squeeze(1)
        c_c = self.centroids[idx]
        w_c = self.centroid_weights[idx]
        
        merged = set()
        new_c_list, new_w_list = [],[]
        total_merged = 0

        for i in range(len(c_c)):
            if i in merged: continue
            sim = self._cdist(c_c[i].unsqueeze(0), c_c)[0]
            dist = 1.0 - sim if self.distance == 'cosine' else 1.0 / sim - 1.0
            
            candidates = ((dist < actual_threshold) & (dist < perc_dist)).nonzero(as_tuple=True)[0]
            candidates =[cand.item() for cand in candidates if cand.item() not in merged]
            
            if len(candidates) > 1:
                merged.update(candidates)
                total_merged += (len(candidates) - 1)
                
                weights = w_c[candidates].unsqueeze(1)
                merged_vec = torch.sum(c_c[candidates] * weights, dim=0) / weights.sum()
                if self.distance == 'cosine':
                    merged_vec = F.normalize(merged_vec.unsqueeze(0), p=2, dim=1).squeeze(0)
                
                new_c_list.append(merged_vec)
                new_w_list.append(weights.sum())
            else:
                new_c_list.append(c_c[i])
                new_w_list.append(w_c[i])
                merged.add(i)
                
        return (torch.stack(new_c_list) if new_c_list else torch.empty(0),
                torch.full((len(new_c_list),), c, dtype=torch.long),
                torch.tensor(new_w_list, dtype=self.centroids.dtype),
                total_merged)

    def _merge(self) -> int:
        """Parallelized framework for merging highly overlapping class centroids.

        Returns:
            The total number of centroids that were merged across all classes during this call.
        """
        if self.relative_merge or self.percentile_threshold is not None:
            subset_size = min(2048, len(self.centroids))
            idx_sub = torch.randperm(len(self.centroids))[:subset_size]
            sub_C = self.centroids[idx_sub]
            
            sim_matrix = self._cdist(sub_C, sub_C)
            dist_matrix = 1.0 - sim_matrix if self.distance == 'cosine' else 1.0 / sim_matrix - 1.0
            mask_off_diag = ~torch.eye(subset_size, dtype=torch.bool, device=self.centroids.device)
            
            if self.relative_merge:
                mean_global_dist = dist_matrix[mask_off_diag].mean().item()
                actual_threshold = self.merge_threshold * mean_global_dist
            else:
                actual_threshold = self.merge_threshold

            if self.percentile_threshold is not None:
                perc_dist = torch.quantile(dist_matrix[mask_off_diag], self.percentile_threshold).item()
            else:
                perc_dist = float('inf')
        else:
            actual_threshold = self.merge_threshold
            perc_dist = float('inf')

        unique_classes = torch.unique(self.centroid_labels).cpu().numpy()
        all_new_c, all_new_l, all_new_w = [], [],[]
        global_merged = 0

        def merge_task(c: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
            return self._merge_single_class(int(c), actual_threshold, perc_dist)

        if self.n_threads > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.n_threads) as executor:
                results = list(executor.map(merge_task, unique_classes))
        else:
            results =[merge_task(c) for c in unique_classes]

        for c_t, l_t, w_t, m_cnt in results:
            if len(c_t) > 0:
                all_new_c.append(c_t.to(self.centroids.device))
                all_new_l.append(l_t.to(self.centroids.device))
                all_new_w.append(w_t.to(self.centroids.device))
            global_merged += m_cnt

        if all_new_c:
            self.centroids = torch.cat(all_new_c)
            self.centroid_labels = torch.cat(all_new_l)
            self.centroid_weights = torch.cat(all_new_w)

        return global_merged

    def predict_proba(self, X: Any, batch_size: Union[str, int, None] = 'auto') -> np.ndarray:
        """Returns normalized class probabilities for the input data.

        Args:
            X: Evaluation feature matrix.
            batch_size: Processing batch size to avoid OOM issues.

        Returns:
            A NumPy array of shape (N, Num_Classes) containing probability distributions.
        """
        is_sp = sp.issparse(X)
        X = self._format_input(X, is_sp)
        N = X.shape[0]
        bs = self.batch_size if batch_size == 'auto' else batch_size
        bs = N if bs is None else bs

        num_classes = len(self.classes_)
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        label_map = torch.tensor([class_to_idx[lbl.item()] for lbl in self.centroid_labels], device=self.centroids.device)
        all_probs =[]

        with torch.no_grad():
            for i in range(0, N, bs):
                X_batch_raw = X[i:i + bs]
                X_batch = self._scipy_to_torch_sparse(X_batch_raw) if is_sp else torch.as_tensor(X_batch_raw, dtype=self.centroids.dtype, device=self.centroids.device)
                
                sim = self._cdist(X_batch, self.centroids)
                
                if self.soft_type == 'linear':
                    scores = F.relu(sim)  
                    sum_scores = scores.sum(dim=1, keepdim=True)
                    zero_mask = (sum_scores == 0).squeeze(1)
                    if zero_mask.any():
                        max_idx = sim[zero_mask].argmax(dim=1)
                        scores[zero_mask] = F.one_hot(max_idx, num_classes=self.centroids.shape[0]).to(self.centroids.dtype)
                        sum_scores[zero_mask] = 1.0
                    centroid_probs = scores / sum_scores
                elif self.soft_type == 'softmax':
                    logits = sim / self.temperature
                    centroid_probs = F.softmax(logits, dim=1)
                
                batch_probs = torch.zeros((X_batch.shape[0], num_classes), dtype=self.centroids.dtype, device=self.centroids.device)
                batch_probs.scatter_add_(1, label_map.unsqueeze(0).expand(X_batch.shape[0], -1), centroid_probs)
                all_probs.append(batch_probs)

        return torch.cat(all_probs).cpu().numpy()

    def predict(self, X: Any, batch_size: Union[str, int, None] = 'auto') -> np.ndarray:
        """Predicts target classes for the input vectors using maximum prototype similarity.

        Args:
            X: Evaluation feature matrix.
            batch_size: Processing batch size to avoid OOM issues.

        Returns:
            A NumPy array of predicted class labels.
        """
        is_sp = sp.issparse(X)
        X = self._format_input(X, is_sp)
        N = X.shape[0]
        bs = self.batch_size if batch_size == 'auto' else batch_size
        bs = N if bs is None else bs
        
        preds =[]
        with torch.no_grad():
            for i in range(0, N, bs):
                X_batch_raw = X[i:i + bs]
                X_batch = self._scipy_to_torch_sparse(X_batch_raw) if is_sp else torch.as_tensor(X_batch_raw, dtype=self.centroids.dtype, device=self.centroids.device)
                sim = self._cdist(X_batch, self.centroids)
                min_idx = torch.argmax(sim, dim=1)
                preds.append(self.centroid_labels[min_idx])
            
        return torch.cat(preds).cpu().numpy()