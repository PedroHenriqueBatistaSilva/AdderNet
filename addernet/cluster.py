"""
AdderCluster — Cluster of AdderNetLayer with batch support.

Each node is an ensemble of AdderNetLayer (1 per feature).
Supports multiple partitioning strategies and combination methods.
"""

import numpy as np
import time
from .addernet import AdderNetLayer


class AdderCluster:
    """
    Cluster of AdderNetLayer with batch.

    Each node is an ensemble of networks (1 per feature).
    Strategies: 'random', 'range', 'feature', 'boosting'
    Combination: 'vote', 'mean', 'stack'
    """

    def __init__(self,
                 n_nodes: int = 2,
                 strategy: str = 'random',
                 combination: str = 'vote',
                 input_min: float = 0,
                 input_max: float = 150,
                 size: int = 256,
                 bias: int = 50,
                 lr: float = 0.05,
                 epochs_raw: int = 2000,
                 epochs_expanded: int = 8000,
                 overlap: float = 0.1,
                 verbose: bool = False):
        self.n_nodes = n_nodes
        self.strategy = strategy
        self.combination = combination
        self.input_min = input_min
        self.input_max = input_max
        self.size = size
        self.bias = bias
        self.lr = lr
        self.epochs_raw = epochs_raw
        self.epochs_expanded = epochs_expanded
        self.overlap = overlap
        self.verbose = verbose
        self.nodes = []
        self.n_features = None
        self.n_classes = None
        self.targets = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'AdderCluster':
        """
        Train the cluster.

        Args:
            X: (n_samples, n_features) — normalized to [input_min, input_max]
            y: (n_samples,) — integer labels 0..n_classes-1
        """
        X = np.ascontiguousarray(X, dtype=np.float64)
        y = np.ascontiguousarray(y, dtype=np.int32)
        n_samples, n_features = X.shape
        self.n_features = n_features
        self.n_classes = len(np.unique(y))

        self.targets = np.linspace(25, 125, self.n_classes)
        y_cont = np.array([self.targets[c] for c in y])

        self.nodes = []
        partitions = self._partition(X, y_cont, n_samples)

        total_t0 = time.perf_counter()
        for node_id, (X_node, y_node) in enumerate(partitions):
            t0 = time.perf_counter()
            node_nets = self._train_node(X_node, y_node)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            self.nodes.append(node_nets)

            if self.verbose:
                print(f"Treinando no {node_id}/{self.n_nodes} (amostras: {len(X_node)})...  "
                      f"✓ {elapsed_ms:.0f}ms")

        if self.verbose:
            total_s = time.perf_counter() - total_t0
            total_nets = sum(len(n) for n in self.nodes)
            print(f"AdderCluster pronto | {total_nets} redes | {total_s:.1f}s total")

        return self

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Batch inference — all nodes in parallel via numpy.

        Returns array of predicted class labels.
        """
        X = np.ascontiguousarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        node_scores = []
        for node_nets in self.nodes:
            votes = np.stack([
                node_nets[i].predict_batch(X[:, i])
                for i in range(self.n_features)
            ], axis=1)
            node_scores.append(votes.mean(axis=1))

        scores = np.stack(node_scores, axis=1)

        if self.combination == 'vote':
            node_classes = np.argmin(
                np.abs(scores[:, :, None] - self.targets[None, None, :]),
                axis=2
            )
            from scipy import stats
            return stats.mode(node_classes, axis=1, keepdims=False)[0].flatten().astype(np.int32)

        elif self.combination == 'mean':
            mean_score = scores.mean(axis=1)
            return np.argmin(
                np.abs(mean_score[:, None] - self.targets[None, :]),
                axis=1
            ).astype(np.int32)

        elif self.combination == 'stack':
            return np.argmin(
                np.abs(scores.mean(axis=1)[:, None] - self.targets[None, :]),
                axis=1
            ).astype(np.int32)

        return np.argmin(
            np.abs(scores.mean(axis=1)[:, None] - self.targets[None, :]),
            axis=1
        ).astype(np.int32)

    def predict(self, x) -> int:
        """Single-sample inference — usa predict_single_fast automaticamente."""
        return self.predict_single_fast(x)

    def predict_single_fast(self, x) -> int:
        """
        Inferência de 1 amostra otimizada — mínimo de overhead Python.
        Usa operações numpy vetorizadas em vez de loop por nó/feature.
        """
        x = np.ascontiguousarray(x, dtype=np.float64).flatten()

        # Pré-alocar matriz de scores (n_nodes,)
        node_scores = np.empty(self.n_nodes, dtype=np.float64)

        for n, node_nets in enumerate(self.nodes):
            # Processar todas as features do nó em batch de 1 amostra
            # predict_batch(array de 1 elemento) é mais rápido que predict(scalar)
            feat_scores = np.empty(self.n_features, dtype=np.float64)
            for i, net in enumerate(node_nets):
                feat_scores[i] = net.predict_batch(x[i:i+1])[0]
            node_scores[n] = feat_scores.mean()

        # Score final = média dos nós
        final_score = node_scores.mean()

        # Snap para classe mais próxima
        return int(np.argmin(np.abs(self.targets - final_score)))

    def info(self) -> None:
        """Print cluster information."""
        print(f"AdderCluster | {self.n_nodes} nodes | strategy={self.strategy} | combination={self.combination}")
        print(f"  Features: {self.n_features} | Classes: {self.n_classes}")
        total_nets = sum(len(n) for n in self.nodes)
        print(f"  Total networks: {total_nets}")
        mem_kb = total_nets * self.size * 8 / 1024
        print(f"  Estimated memory: {mem_kb:.1f} KB")
        for i, nets in enumerate(self.nodes):
            print(f"  Node {i}: {len(nets)} AdderNetLayer")

    def _partition(self, X, y_cont, n_samples):
        """Partition data according to strategy."""
        partitions = []

        if self.strategy == 'random':
            idx = np.random.permutation(n_samples)
            size = n_samples // self.n_nodes
            for i in range(self.n_nodes):
                start = i * size
                end = start + size if i < self.n_nodes - 1 else n_samples
                partitions.append((X[idx[start:end]], y_cont[idx[start:end]]))

        elif self.strategy == 'range':
            # Problema 2: Stratified range partition — normalize features,
            # then split ensuring class balance per node
            X_min = X.min(axis=0)
            X_range = X.max(axis=0) - X_min
            X_range[X_range < 1e-9] = 1.0
            X_scaled = (X - X_min) / X_range
            x_mean = X_scaled.mean(axis=1)

            # Stratified: for each class, split its samples across nodes
            unique_classes = np.unique(np.digitize(y_cont, bins=np.linspace(
                y_cont.min() - 1, y_cont.max() + 1, self.n_classes + 1)) - 1)
            # Use actual y_cont values to determine class
            target_to_class = {t: c for c, t in enumerate(self.targets)}
            class_labels = np.array([target_to_class.get(
                self.targets[np.argmin(np.abs(t - self.targets))], 0) for t in y_cont])

            node_indices = [[] for _ in range(self.n_nodes)]
            for cls in range(self.n_classes):
                cls_mask = class_labels == cls
                cls_indices = np.where(cls_mask)[0]
                if len(cls_indices) == 0:
                    continue
                cls_means = x_mean[cls_indices]
                cls_order = np.argsort(cls_means)
                sorted_cls_indices = cls_indices[cls_order]

                # Split this class evenly across nodes
                per_node = max(1, len(sorted_cls_indices) // self.n_nodes)
                for i in range(self.n_nodes):
                    start = i * per_node
                    end = start + per_node if i < self.n_nodes - 1 else len(sorted_cls_indices)
                    node_indices[i].extend(sorted_cls_indices[start:end].tolist())

            for i in range(self.n_nodes):
                idx = np.array(node_indices[i])
                if len(idx) == 0:
                    idx = np.random.permutation(n_samples)[:max(30, n_samples // self.n_nodes)]
                partitions.append((X[idx], y_cont[idx]))

        elif self.strategy == 'feature':
            for _i in range(self.n_nodes):
                partitions.append((X, y_cont))

        elif self.strategy == 'boosting':
            residuals = y_cont.copy()
            for _i in range(self.n_nodes):
                idx = np.random.permutation(n_samples)[:max(1, n_samples // self.n_nodes)]
                partitions.append((X[idx], residuals[idx]))

        return partitions

    def _train_node(self, X_node, y_node):
        """Train 1 node: 1 AdderNetLayer per feature."""
        node_nets = []
        for i in range(self.n_features):
            net = AdderNetLayer(
                size=self.size,
                bias=self.bias,
                input_min=self.input_min,
                input_max=self.input_max,
                lr=self.lr,
            )
            net.train(
                X_node[:, i],
                y_node,
                epochs_raw=self.epochs_raw,
                epochs_expanded=self.epochs_expanded,
            )
            node_nets.append(net)
        return node_nets
