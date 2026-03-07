import numpy as np
from collections import defaultdict
from scipy import sparse
from sklearn.neighbors import kneighbors_graph


def build_local_affinity(features, num_neighbors=30):
    if len(features) <= 1:
        return sparse.csr_matrix((len(features), len(features)), dtype=np.float32)
    graph = kneighbors_graph(
        features,
        n_neighbors=min(num_neighbors, len(features) - 1),
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
    row_sum = np.asarray(graph.sum(axis=1)).ravel()
    row_sum[row_sum == 0] = 1.0
    return sparse.diags(1.0 / row_sum) @ graph


def label_propagation_with_seeds(features, seed_local_indices, alpha=0.85, max_iter=100, tol=1e-5):
    n_nodes = len(features)
    n_labels = len(seed_local_indices)
    if n_nodes == 0 or n_labels == 0:
        return np.array([], dtype=int), np.array([], dtype=np.float32)

    affinity = build_local_affinity(features)
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

    labels = scores.argmax(axis=1)
    confidence = scores.max(axis=1)
    return labels.astype(int), confidence.astype(np.float32)


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
                            contig_dict, contig_to_marker, minfasta):
    name_to_local = {all_names[global_idx]: local_idx for local_idx, global_idx in enumerate(idx_in_bin)}
    seed_local = [name_to_local[name] for name in seed_names if name in name_to_local]
    if len(seed_local) <= 1:
        return None

    labels, confidence = label_propagation_with_seeds(features, seed_local)
    if len(labels) == 0:
        return None

    assigned = defaultdict(list)
    for local_idx, label in enumerate(labels):
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
