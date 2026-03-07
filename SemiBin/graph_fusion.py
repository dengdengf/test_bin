import numpy as np
from scipy import sparse
from sklearn.neighbors import kneighbors_graph


def _distance_to_similarity(graph):
    graph = graph.tocsr().astype(np.float32)
    if graph.nnz == 0:
        return graph
    sigma = float(np.median(graph.data))
    if sigma <= 0:
        sigma = float(np.mean(graph.data) + 1e-6)
    graph.data = np.exp(-graph.data / (sigma + 1e-6))
    return symmetrize_graph(graph)


def symmetrize_graph(graph):
    return graph.maximum(graph.T).tocsr()


def build_similarity_graph(features, n_neighbors, num_process):
    if features is None or features.shape[1] == 0 or features.shape[0] <= 1:
        return None
    graph = kneighbors_graph(
        features,
        n_neighbors=min(n_neighbors, features.shape[0] - 1),
        mode='distance',
        p=2,
        n_jobs=num_process,
    )
    return _distance_to_similarity(graph)


def normalize_graph(graph):
    if graph is None or graph.nnz == 0:
        return graph
    graph = graph.tocsr().astype(np.float32)
    row_sums = np.asarray(graph.sum(axis=1)).ravel()
    row_sums[row_sums == 0] = 1.0
    inv = sparse.diags(1.0 / row_sums)
    return inv @ graph


def fuse_similarity_graphs(graph_specs):
    valid_specs = [(graph, weight) for graph, weight in graph_specs if graph is not None and weight > 0]
    if not valid_specs:
        raise ValueError('No valid graphs were provided for fusion')

    weight_sum = sum(weight for _, weight in valid_specs)
    fused = None
    for graph, weight in valid_specs:
        normalized = normalize_graph(graph)
        current = normalized.multiply(weight / weight_sum)
        fused = current if fused is None else fused + current
    return symmetrize_graph(fused)


def prune_graph_by_quantile(graph, max_fraction):
    graph = graph.tocsr().astype(np.float32)
    if graph.nnz == 0:
        return graph
    max_axis1 = graph.max(axis=1).toarray().ravel()
    threshold = 0.0
    while threshold < 1:
        threshold += 0.05
        n_above = np.sum(max_axis1 > threshold)
        if round(n_above / graph.shape[0], 2) < max_fraction:
            break
    threshold -= 0.05
    graph.data[graph.data <= threshold] = 0
    graph.eliminate_zeros()
    return graph


def row_normalize_dense(matrix):
    if matrix is None or matrix.shape[1] == 0:
        return matrix
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms
