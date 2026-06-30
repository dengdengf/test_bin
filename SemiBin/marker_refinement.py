import numpy as np
from collections import defaultdict
from scipy import sparse
from sklearn.neighbors import kneighbors_graph


def build_local_affinity(features, num_neighbors=30, node_weights=None):
    n_nodes = len(features)
    if n_nodes <= 1:
        return sparse.csr_matrix((n_nodes, n_nodes), dtype=np.float32)
    # Adapt the neighbourhood size to the bin: with only a few dozen contigs a fixed
    # k=30 connects almost everything and washes out the marker seeds. Keep at least 5.
    k = min(num_neighbors, max(5, n_nodes // 3))
    k = min(k, n_nodes - 1)
    graph = kneighbors_graph(
        features,
        n_neighbors=k,
        mode='distance',
        p=2,
        n_jobs=1,
    ).tocsr()
    if graph.nnz:
        sigma = float(np.median(graph.data))
        if sigma <= 0:
            sigma = float(np.mean(graph.data) + 1e-6)
        graph.data = np.exp(-graph.data / (sigma + 1e-6))
    graph = graph.maximum(graph.T).tocsr()
    if node_weights is not None:
        # Let longer contigs exert proportionally more influence on their neighbours by
        # scaling each column (the source of diffused mass) by the contig length. This
        # restores the length-awareness that upstream gets from KMeans sample_weight.
        w = np.asarray(node_weights, dtype=np.float32)
        w = w / (float(np.mean(w)) + 1e-9)
        graph = (graph @ sparse.diags(w)).tocsr()
    row_sum = np.asarray(graph.sum(axis=1)).ravel()
    row_sum[row_sum == 0] = 1.0
    return sparse.diags(1.0 / row_sum) @ graph


def label_propagation_with_seeds(features, seed_local_indices, node_weights=None,
                                 alpha=None, max_iter=100, tol=1e-5, confidence_margin=0.0):
    n_nodes = len(features)
    n_labels = len(seed_local_indices)
    if n_nodes == 0 or n_labels == 0:
        return np.array([], dtype=int), np.array([], dtype=np.float32)

    if alpha is None:
        # Smaller bins -> trust the seeds more (lower alpha) so diffusion does not smear a
        # handful of contigs evenly across every seed label.
        alpha = 0.85 if n_nodes >= 200 else 0.6 + 0.25 * min(1.0, n_nodes / 200.0)

    affinity = build_local_affinity(features, node_weights=node_weights)
    seed_matrix = np.zeros((n_nodes, n_labels), dtype=np.float32)
    for label, local_idx in enumerate(seed_local_indices):
        seed_matrix[local_idx, label] = 1.0

    scores = seed_matrix.copy()
    for _ in range(max_iter):
        updated = alpha * (affinity @ scores) + (1 - alpha) * seed_matrix
        updated[seed_local_indices] = seed_matrix[seed_local_indices]
        if np.max(np.abs(updated - scores)) < tol:
            scores = updated
            break
        scores = updated

    labels = scores.argmax(axis=1).astype(int)
    # Confidence = margin between the top-2 labels on the row-normalised (probability-like)
    # scores. A near-zero margin means the contig sits on a boundary between sub-bins.
    probs = scores / (scores.sum(axis=1, keepdims=True) + 1e-12)
    if n_labels >= 2:
        part = np.partition(probs, -2, axis=1)
        confidence = part[:, -1] - part[:, -2]
    else:
        confidence = probs.max(axis=1)
    if confidence_margin > 0:
        ambiguous = confidence < confidence_margin
        ambiguous[seed_local_indices] = False  # never drop a marker seed
        labels[ambiguous] = -1
    return labels, confidence.astype(np.float32)


def marker_redundancy(contigs, contig_to_marker):
    markers = []
    for contig in contigs:
        markers.extend(contig_to_marker.get(contig, []))
    return max(0, len(markers) - len(set(markers)))


def evaluate_refinement(bin_groups, contig_dict, contig_to_marker, minfasta):
    stats = []
    for contigs in bin_groups:
        total_bp = sum(len(contig_dict[c]) for c in contigs)
        redundancy = marker_redundancy(contigs, contig_to_marker)
        stats.append({
            'contigs': contigs,
            'bp': total_bp,
            'redundancy': redundancy,
            'valid': total_bp >= minfasta,
        })
    return stats


def refine_contaminated_bin(features, idx_in_bin, seed_names, all_names,
                            contig_dict, contig_to_marker, minfasta,
                            confidence_margin=0.02):
    name_to_local = {all_names[global_idx]: local_idx for local_idx, global_idx in enumerate(idx_in_bin)}
    seed_local = [name_to_local[name] for name in seed_names if name in name_to_local]
    if len(seed_local) <= 1:
        return None

    lengths = np.array([len(contig_dict[all_names[g]]) for g in idx_in_bin], dtype=np.float32)
    labels, confidence = label_propagation_with_seeds(
        features, seed_local,
        node_weights=lengths,
        confidence_margin=confidence_margin)
    if len(labels) == 0:
        return None

    assigned = defaultdict(list)
    for local_idx, label in enumerate(labels):
        if label < 0:
            # Boundary contig left unassigned; it becomes a singleton upstream rather than
            # being forced into (and contaminating) one of the refined sub-bins.
            continue
        global_idx = idx_in_bin[local_idx]
        assigned[label].append(all_names[global_idx])

    refined_groups = [contigs for contigs in assigned.values() if contigs]
    if len(refined_groups) <= 1:
        return None

    refined_stats = evaluate_refinement(refined_groups, contig_dict, contig_to_marker, minfasta)
    original_redundancy = marker_redundancy([all_names[g] for g in idx_in_bin], contig_to_marker)
    valid_groups = [stat for stat in refined_stats if stat['valid']]
    refined_redundancy = sum(stat['redundancy'] for stat in refined_stats)

    if len(valid_groups) < 2:
        return None
    if refined_redundancy >= original_redundancy:
        return None
    return refined_groups
