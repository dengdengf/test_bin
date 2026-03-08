import os
from os import path
import math
import shutil
import tempfile
import numpy as np
from .utils import write_bins
from .markers import estimate_seeds

# This is the default in the igraph package
NR_INFOMAP_TRIALS = 10

def run_infomap1(g, edge_weights, vertex_weights, trials):
    return g.community_infomap(edge_weights=edge_weights, vertex_weights=vertex_weights, trials=trials)

def run_infomap(g, edge_weights, vertex_weights, num_process):
    '''Run infomap, using multiple processors (if available)'''
    if num_process == 1:
        return g.community_infomap(edge_weights=edge_weights, vertex_weights=vertex_weights, trials=NR_INFOMAP_TRIALS)
    import multiprocessing as mp
    with mp.Pool(min(num_process, NR_INFOMAP_TRIALS)) as p:
        rs = [p.apply_async(run_infomap1, (g, edge_weights, vertex_weights, 1))
                for _ in range(NR_INFOMAP_TRIALS)]
        p.close()
        p.join()
    rs = [r.get() for r in rs]
    best = rs[0]
    for r in rs[1:]:
        if r.codelength < best.codelength:
            best = r
    return best



def cal_kl(m, v, use_ne='auto'):
    # A naive implementation creates a lot of copies of what can
    # become large matrices
    import numpy as np

    m = np.clip(m, 1e-6, None)
    v = np.clip(v, 1.0, None)

    m1 = m.reshape(1, len(m))
    m2 = m.reshape(len(m), 1)

    v1 = v.reshape(1, len(v))
    v2 = v.reshape(len(v), 1)


    if use_ne != 'no':
        try:
            import numexpr as ne

            res = ne.evaluate(
                    '(log(v1) - log(v2))/2 + ( (m1 - m2)**2 + v2 ) / ( 2 * v1 ) - half',
                    {
                        'v1': v1,
                        'v2': v2,
                        'm1': m1,
                        'm2': m2,
                        # numexpr rules are that mixing with floats causes
                        # conversion to float64
                        # Note that this does not happen for integers
                        'half': np.float32(0.5),
                    })
            np.clip(res, 1e-6, 1-1e-6, out=res)
            return res
        except ImportError:
            if use_ne != 'auto':
                raise
    v_div = np.log(v1) - np.log(v2)
    v_div /= 2.0

    m_dif = m1 - m2
    m_dif **=2
    m_dif += v2
    m_dif /= 2 * v1

    v_div += m_dif
    v_div -= 0.5
    np.clip(v_div, 1e-6, 1-1e-6, out=v_div)
    return v_div


def run_embed_infomap(logger, model, data, * ,
            device, max_edges, max_node, is_combined, n_sample, contig_dict,
            num_process : int, random_seed, dnabert_embedding=None): # <--- 新增参数
    """
    Cluster contigs into bins
    """
    from igraph import Graph
    from sklearn.neighbors import kneighbors_graph
    from scipy import sparse
    import torch
    import numpy as np
    from .utils import norm_abundance
    from sklearn.preprocessing import normalize

    # 1. 准备数据和 SemiBin 原生 Embedding (embedding_B)
    train_data_input = data.values[:, 0:136] if not is_combined else data.values
    if is_combined:
        if norm_abundance(train_data_input):
            norm = np.sum(train_data_input, axis=0)
            train_data_input = train_data_input / norm
            train_data_input = normalize(train_data_input, axis=1, norm='l1')

    depth = data.values[:, 136:len(data.values[0])].astype(np.float32)
    num_contigs = train_data_input.shape[0]

    with torch.no_grad():
        model.eval()
        x = torch.from_numpy(train_data_input).to(device)
        embedding_B = model.embedding(x.float()).detach().cpu().numpy()

    # 2. 构建图矩阵 (embedding_matrix)
    if dnabert_embedding is not None:
        logger.info("Running hybrid binning mode (DNABERT + SemiBin).")
        # 确保类型一致
        embedding_A = dnabert_embedding.astype(np.float32)

        # 构建两个近邻图
        matrix_A = kneighbors_graph(embedding_A, n_neighbors=min(max_edges, data.shape[0]-1),
                                    mode='distance', p=2, n_jobs=num_process)
        matrix_B = kneighbors_graph(embedding_B, n_neighbors=min(max_edges, data.shape[0]-1),
                                    mode='distance', p=2, n_jobs=num_process)

        # 取交集
        matrix_A.eliminate_zeros()
        matrix_A.data.fill(1.0)
        embedding_matrix = matrix_B.multiply(matrix_A)
        embedding_matrix.eliminate_zeros()
        
        # 释放内存
        del matrix_A, matrix_B
    else:
        # 原生逻辑
        embedding_matrix = kneighbors_graph(embedding_B, n_neighbors=min(max_edges, data.shape[0]-1),
                                            mode='distance', p=2, n_jobs=num_process)
        kmer_matrix = kneighbors_graph(train_data_input, n_neighbors=min(max_edges, data.shape[0]-1),
                                     mode='distance', p=2, n_jobs=num_process)
        kmer_matrix.eliminate_zeros()
        kmer_matrix.data.fill(1.)
        embedding_matrix = embedding_matrix.multiply(kmer_matrix)
        embedding_matrix.eliminate_zeros()
        del kmer_matrix

    # 3. 后处理图矩阵 (阈值、Abundance 校正等) - 保持原样
    np.clip(embedding_matrix.data, None, 1., out=embedding_matrix.data)
    embedding_matrix.data = 1 - embedding_matrix.data

    threshold = 0
    max_axis1 = embedding_matrix.max(axis=1).toarray()
    while threshold < 1:
        threshold += 0.05
        n_above = np.sum(max_axis1 > threshold)
        if round(n_above / num_contigs, 2) < max_node:
            break
    threshold -= 0.05
    embedding_matrix.data[embedding_matrix.data <= threshold] = 0
    embedding_matrix.eliminate_zeros()

    if not is_combined:
        logger.debug('Calculating depth matrix.')
        kl_matrix = None
        for k in range(n_sample):
            kl = cal_kl(depth[:,2*k], depth[:, 2*k + 1])
            if kl_matrix is None:
                kl *= -1
                kl_matrix = kl
            else:
                kl_matrix -= kl
        kl_matrix += n_sample
        kl_matrix /= n_sample
        embedding_matrix = embedding_matrix.multiply(kl_matrix)

    embedding_matrix.data[embedding_matrix.data <= 1e-6] = 0
    X, Y, V = sparse.find(embedding_matrix)
    above_diag = Y > X
    X = X[above_diag]
    Y = Y[above_diag]
    edges = [(x,y) for x,y in zip(X, Y)]
    edge_weights = V[above_diag]

    # 4. 运行 Infomap 聚类
    logger.debug(f'Running infomap with {num_process} processes...')
    
    g = Graph()
    g.add_vertices(np.arange(num_contigs))
    g.add_edges(edges)
    length_weight = np.array([len(contig_dict[name]) for name in data.index])
    
    result = run_infomap(g,
                edge_weights=edge_weights,
                vertex_weights=length_weight,
                num_process=num_process)
    
    contig_labels = np.zeros(shape=num_contigs, dtype=int)
    for i, r in enumerate(result):
        contig_labels[r] = i

    # 5. 准备返回值
    # 如果有 DNABERT，我们将两者归一化后拼接，供 Reclustering 使用

# run_embed_infomap 函数的第 5 步：准备返回值

    # 5. 准备返回值
    # 如果有 DNABERT，我们将两者归一化后拼接，供 Reclustering 使用
    if dnabert_embedding is not None:
        embedding_A = dnabert_embedding.astype(np.float32)
        logger.info(f"Returning DNABERT embedding ONLY for reclustering (Dim: {embedding_A.shape[1]})")
        return embedding_A, contig_labels # <--- 关键修改
    else:
        return embedding_B, contig_labels


    # if dnabert_embedding is not None:
    #     combined_emb = np.concatenate([
    #         normalize(embedding_B, axis=1), 
    #         normalize(dnabert_embedding, axis=1)
    #     ], axis=1)
    #     logger.info(f"Returning hybrid embedding for reclustering (Dim: {combined_emb.shape[1]})")
    #     return combined_emb, contig_labels
    # else:
    #     return embedding_B, contig_labels


# def run_embed_infomap(logger, model, data, * ,
#             device, max_edges, max_node, is_combined, n_sample, contig_dict,
#             num_process : int, random_seed):
#     """
#     Cluster contigs into bins
#     max_edges: max edges of one contig considered in binning
#     max_node: max percentage of contigs considered in binning
#     """
#     from igraph import Graph
#     from sklearn.neighbors import kneighbors_graph
#     from scipy import sparse
#     import torch
#     import numpy as np
#     from .utils import norm_abundance

#     train_data_input = data.values[:, 0:136] if not is_combined else data.values

#     if is_combined:
#         if norm_abundance(train_data_input):
#             from sklearn.preprocessing import normalize
#             norm = np.sum(train_data_input, axis=0)
#             train_data_input = train_data_input / norm
#             train_data_input = normalize(train_data_input, axis=1, norm='l1')

#     depth = data.values[:, 136:len(data.values[0])].astype(np.float32)
#     num_contigs = train_data_input.shape[0]
#     with torch.no_grad():
#         model.eval()
#         x = torch.from_numpy(train_data_input).to(device)
#         embedding = model.embedding(x.float()).detach().cpu().numpy()
#         embedding_matrix = kneighbors_graph(
#             embedding,
#             n_neighbors=min(max_edges, data.shape[0] - 1),
#             mode='distance',
#             p=2,
#             n_jobs=num_process)
#         kmer_matrix = kneighbors_graph(
#             train_data_input,
#             n_neighbors=min(max_edges, data.shape[0] - 1),
#             mode='distance',
#             p=2,
#             n_jobs=num_process)

#         # We want to intersect the matrices, so we make kmer_matrix into a
#         # matrix of 1.s and multiply
#         kmer_matrix.eliminate_zeros()
#         kmer_matrix.data.fill(1.)
#         embedding_matrix = embedding_matrix.multiply(kmer_matrix)
#         embedding_matrix.eliminate_zeros()
#         del kmer_matrix

#     np.clip(embedding_matrix.data, None, 1., out=embedding_matrix.data)
#     embedding_matrix.data = 1 - embedding_matrix.data

#     threshold = 0
#     max_axis1 = embedding_matrix.max(axis=1).toarray()
#     while threshold < 1:
#         threshold += 0.05
#         n_above = np.sum(max_axis1 > threshold)
#         if round(n_above / num_contigs, 2) < max_node:
#             break
#     threshold -= 0.05

#     embedding_matrix.data[embedding_matrix.data<= threshold] = 0
#     embedding_matrix.eliminate_zeros()
#     if not is_combined:
#         logger.debug('Calculating depth matrix.')
#         kl_matrix = None
#         for k in range(n_sample):
#             kl = cal_kl(depth[:,2*k], depth[:, 2*k + 1])
#             if kl_matrix is None:
#                 kl *= -1
#                 kl_matrix = kl
#             else:
#                 kl_matrix -= kl
#         kl_matrix += n_sample
#         kl_matrix /= n_sample
#         embedding_matrix = embedding_matrix.multiply(kl_matrix)

#     embedding_matrix.data[embedding_matrix.data <= 1e-6] = 0
#     X, Y, V = sparse.find(embedding_matrix)
#     # Find above diagonal positions:
#     above_diag = Y > X
#     X = X[above_diag]
#     Y = Y[above_diag]

#     edges = [(x,y) for x,y in zip(X, Y)]
#     edge_weights = V[above_diag]

#     logger.debug(f'Number of edges in clustering graph: {len(edges)}')

#     g = Graph()
#     g.add_vertices(np.arange(num_contigs))
#     g.add_edges(edges)
#     length_weight = np.array([len(contig_dict[name]) for name in data.index])
#     logger.debug(f'Running infomap with {num_process} processes...')
#     result = run_infomap(g,
#                 edge_weights=edge_weights,
#                 vertex_weights=length_weight,
#                 num_process=num_process)
#     contig_labels = np.zeros(shape=num_contigs, dtype=int)

#     for i, r in enumerate(result):
#         contig_labels[r] = i
#     return embedding, contig_labels




def recluster_bins(logger, data, *,
                n_sample,
                embedding,
                is_combined: bool,
                contig_labels,
                minfasta,
                contig_dict,
                binned_length,
                orf_finder,
                num_process,
                random_seed):
    from sklearn.cluster import KMeans, MiniBatchKMeans
    import numpy as np
    import math
    from collections import defaultdict
    import tempfile
    import os
    import gc

    np.random.seed(random_seed)
    gc.collect()

    # === 1. 深度预处理 ===
    if not is_combined:
        depth = data.values[:, 136:].astype(np.float32)
        depth_mean = depth[:, ::2]
        scale = np.ceil(depth_mean.mean(axis=0) / 100) * 100
        scale = np.clip(scale, 1e-6, None)
        depth_mean = depth_mean / scale
        scaling = np.mean(np.abs(embedding)) / (np.mean(depth_mean) + 1e-8)
        base = 10

    # === 2. bin 长度统计 ===
    total_size = defaultdict(int)
    for i, label in enumerate(contig_labels):
        total_size[label] += len(contig_dict[data.index[i]])

    # === 3. CheckM 种子检测 ===
    with tempfile.TemporaryDirectory() as tdir:
        cfasta = os.path.join(tdir, 'bins.fna')
        with open(cfasta, 'wt') as f:
            for i, ctg in enumerate(data.index):
                bid = contig_labels[i]
                if total_size[bid] < minfasta:
                    continue
                f.write(f'>bin{bid:06d}.{ctg}\n')
                f.write(contig_dict[ctg] + '\n')

        seeds = estimate_seeds(cfasta, binned_length, num_process,
                               multi_mode=True, orf_finder=orf_finder)

    if not seeds:
        logger.info("No contaminated bins detected. Skipping reclustering.")
        return contig_labels.copy()

    name2ix = {n: i for i, n in enumerate(data.index)}
    new_labels = np.full_like(contig_labels, -1)
    next_label = 0

    # === 4. 主循环 ===
    for bin_id in range(contig_labels.max() + 1):
        idx_in_bin = [i for i, l in enumerate(contig_labels) if l == bin_id]
        if not idx_in_bin:
            continue

        seed_names = seeds.get(f'bin{bin_id:06d}', [])
        n_seeds = len(seed_names)

        # 只处理真正污染的 bin
        if n_seeds <= 1 or total_size[bin_id] < minfasta:
            if total_size[bin_id] >= minfasta:
                for i in idx_in_bin:
                    new_labels[i] = next_label
                next_label += 1
            continue

        logger.info(f"--- FORCE SPLIT contaminated bin {bin_id} | Seeds: {n_seeds} | Contigs: {len(idx_in_bin)} ---")

        # 激进参数
        # penalty = 80 + 50 * (n_seeds - 1)                     # 2 seeds → 130
        # keep_pct = max(72, 88 - 6 * (n_seeds - 1))            # 2 seeds → 88%

        penalty = 2
        keep_pct = 60

        logger.info(f"  [FORCE MODE] Penalty={penalty}, Keep top {keep_pct}%, Drop ~{100-keep_pct}% bp")

        # 统一特征空间
        # 1. 提取 contigs 对应的 DNABERT 嵌入（此时 embedding 已经是纯 DNABERT）
        dnabert_features = embedding[idx_in_bin].astype(np.float32)
        
        # 2. 提取并归一化丰度信息
        # 丰度信息位于原始数据 data.values 的 136 列之后
        # 使用原始代码中 depth_mean 的计算逻辑
        
        # **注意**：此处需确保在主循环外能访问到 `data` 和 `depth_mean`
        # 重新计算 depth/depth_mean 确保变量在当前作用域内，并适应 index
        depth = data.values[:, 136:].astype(np.float32)
        depth_mean = depth[:, ::2]
        
        # 提取当前 bin 的丰度
        current_depth_mean = depth_mean[idx_in_bin]
        
        # 归一化丰度 (l1 或 l2 都可以，这里使用 l2 归一化以确保所有维度权重一致)
        from sklearn.preprocessing import normalize
        
        # 归一化丰度矩阵
        # 为了避免丰度维度被 DNABERT 维度完全压制，可以对丰度进行归一化
        # 另一种简单方法是：先对 mean_depth 取 log(x+1) 减少数量级差异
        # 这里使用简单的 l2 归一化
        normalized_abundance = normalize(current_depth_mean, axis=1, norm='l2')

        # 3. 拼接特征
        features = np.concatenate([
            dnabert_features, 
            normalized_abundance 
        ], axis=1)
        
        logger.debug(f"  [FEATURE SPACE] DNABERT + Normalized Abundance Dim: {features.shape[1]}")

        # === 关键：以种子为锚点强制划分 ===
        seed_coords = []
        seed_global_idx = set()
        for s in seed_names:
            g = name2ix.get(s, -1)
            if g != -1 and g in idx_in_bin:
                local_i = idx_in_bin.index(g)          # ← 之前漏掉的这行！
                seed_coords.append(features[local_i])
                seed_global_idx.add(g)

        if len(seed_coords) < n_seeds:
            logger.warning("  Not all seeds found → keep original bin")
            for i in idx_in_bin:
                new_labels[i] = next_label
            next_label += 1
            continue

        seed_coords = np.stack(seed_coords)  # (n_seeds, dim)

        # 计算每个 contig 到所有种子的距离
        dist_to_seeds = np.linalg.norm(
            features[:, np.newaxis, :] - seed_coords[np.newaxis, :, :], axis=2
        )  # (n_contigs, n_seeds)

        labels = np.argmin(dist_to_seeds, axis=1)
        min_dist = dist_to_seeds[np.arange(len(labels)), labels]

        # 距离截断：只保留最近的 keep_pct%
        thresh = np.percentile(min_dist, keep_pct)
        keep_mask = min_dist <= thresh

        logger.info(f"  [HARD SPLIT] Assigned by nearest seed, dropped {(~keep_mask).sum()} contigs (dist>{thresh:.3f})")

        # === 诚实回滚检查：冗余种子是否真正分离 ===
        redundant_per_subbin = []
        valid_subbins = 0
        subbin_bp = []

        for k in range(n_seeds):
            mask_k = (labels == k) & keep_mask
            if not mask_k.any():
                redundant_per_subbin.append(0)
                continue

            redundant = sum(1 for i in np.where(mask_k)[0] if idx_in_bin[i] in seed_global_idx)
            redundant_per_subbin.append(redundant)

            bp = sum(len(contig_dict[data.index[idx_in_bin[i]]]) for i in np.where(mask_k)[0])
            subbin_bp.append(bp)

            if redundant <= 1 and bp >= 500_000 and mask_k.sum() >= 8:
                valid_subbins += 1

        logger.info(f"  [HONEST CHECK] Redundant seeds: {redundant_per_subbin} | BP: {subbin_bp}")

        if valid_subbins >= 2 and all(r <= 1 for r in redundant_per_subbin):
            logger.info(f"  [PERFECT] Successfully created {valid_subbins} high-quality bins!")
            cur_id = next_label
            for k in range(n_seeds):
                mask_k = (labels == k) & keep_mask
                if mask_k.any():
                    new_labels[np.array(idx_in_bin)[mask_k]] = cur_id
                    cur_id += 1
            next_label += valid_subbins
        else:
            logger.warning("  [ROLLBACK] Redundant markers not separated → keep original bin")
            for i in idx_in_bin:
                new_labels[i] = next_label
            next_label += 1

        logger.info("-" * 90)

    # === 清理未分配 ===
    unassigned = new_labels == -1
    if unassigned.any():
        logger.warning(f"{unassigned.sum()} contigs unassigned → treated as singletons")
        for i in np.where(unassigned)[0]:
            if len(contig_dict[data.index[i]]) >= minfasta:
                new_labels[i] = next_label
                next_label += 1

    return new_labels







def cluster(logger, model, data, device, is_combined,
            n_sample, out, contig_dict, *, args,
            binned_length, minfasta,dnabert_embedding=None):
    """
    Cluster contigs into bins
    max_edges: max edges of one contig considered in binning
    max_node: max percentage of contigs considered in binning
    """
    import pandas as pd
    import os
    import shutil

    embedding, contig_labels = run_embed_infomap(logger, model, data,
            device=device, max_edges=args.max_edges, max_node=args.max_node,
            is_combined=is_combined, n_sample=n_sample,
            contig_dict=contig_dict, num_process=args.num_process,
            random_seed=args.random_seed,
            dnabert_embedding=dnabert_embedding)

    if args.write_pre_reclustering_bins or not args.recluster:
        output_bin_path = os.path.join(out,
                    'output_prerecluster_bins' if args.recluster else 'output_bins')
        if os.path.exists(output_bin_path):
            logger.warning(f'Previous output directory `{output_bin_path}` found. Over-writing it.')
            shutil.rmtree(output_bin_path)
        os.makedirs(output_bin_path, exist_ok=True)


        bin_files = write_bins(data.index.tolist(),
                            contig_labels,
                            output_bin_path,
                            contig_dict,
                            minfasta=minfasta,
                            output_tag=args.output_tag,
                            output_compression=args.output_compression)
        if not len(bin_files):
            logger.warning('No bins were created. Please check your input data.')
            return
        if not args.recluster:
            logger.info(f'Number of bins: {len(bin_files)}')
            bin_files.to_csv(os.path.join(out, 'bins_info.tsv'), index=False, sep='\t')
            pd.DataFrame({'contig': data.index, 'bin': contig_labels}).to_csv(
                os.path.join(out, 'contig_bins.tsv'), index=False, sep='\t')
        n_pre_bins = len(bin_files)
    else:
        from collections import defaultdict
        total_size = defaultdict(int)
        for i, c in enumerate(contig_labels):
            total_size[c] += len(contig_dict[data.index[i]])
        n_pre_bins = sum((total_size[bin_ix] >= minfasta for bin_ix in range(contig_labels.max() + 1)))


    if args.recluster:
        logger.info(f'Number of bins prior to reclustering: {n_pre_bins}')
        logger.debug('Reclustering...')

        if n_pre_bins == 0:
            import numpy as np
            logger.warning('No bins were created. Please check your input data.')
            contig_labels_reclustered = np.full(len(data.index), fill_value=-1, dtype=int)
        else:
            contig_labels_reclustered = recluster_bins(logger,
                                                data,
                                                n_sample=n_sample,
                                                embedding=embedding,
                                                contig_labels=contig_labels,
                                                contig_dict=contig_dict,
                                                minfasta=minfasta,
                                                binned_length=binned_length,
                                                num_process=args.num_process,
                                                orf_finder=args.orf_finder,
                                                random_seed=args.random_seed,
                                                is_combined=is_combined)
        output_recluster_bin_path = path.join(out,
                        ('output_recluster_bins'
                            if args.write_pre_reclustering_bins
                            else 'output_bins'))
        if os.path.exists(output_recluster_bin_path):
            logger.warning(f'Previous output directory `{output_recluster_bin_path}` found. Over-writing it.')
            shutil.rmtree(output_recluster_bin_path)
        os.makedirs(output_recluster_bin_path, exist_ok=True)
        outputs = write_bins(data.index.tolist(),
                            contig_labels_reclustered,
                            output_recluster_bin_path,
                            contig_dict,
                            minfasta=minfasta,
                            output_tag=args.output_tag,
                            output_compression=args.output_compression)
        logger.info(f'Number of bins after reclustering: {len(outputs)}')
        outputs.to_csv(os.path.join(out, 'recluster_bins_info.tsv'), index=False, sep='\t')
        pd.DataFrame({'contig': data.index, 'bin': contig_labels_reclustered}).to_csv(
            os.path.join(out, 'contig_bins.tsv'), index=False, sep='\t')
    logger.info('Binning finished')




# def recluster_bins(logger, data, *,
#                 n_sample,
#                 embedding,
#                 is_combined: bool,
#                 contig_labels,
#                 minfasta,
#                 contig_dict,
#                 binned_length,
#                 orf_finder,
#                 num_process,
#                 random_seed):
#     from sklearn.mixture import GaussianMixture 
#     import numpy as np
#     import math
#     from collections import defaultdict
#     import tempfile
#     import os

#     # --- 1. 基础数据准备 ---
#     if not is_combined:
#         depth = data.values[:, 136:len(data.values[0])].astype(np.float32)
#         mean_index = [2 * temp for temp in range(n_sample)]
#         depth_mean = depth[:, mean_index]
#         abun_scale = np.ceil(depth_mean.mean(axis=0) / 100) * 100
#         abun_scale = np.clip(abun_scale, 1e-6, None)
#         depth_mean = depth_mean / abun_scale
#         scaling = np.mean(np.abs(embedding)) / np.mean(depth_mean)
#         base = 10
    
#     total_size = defaultdict(int)
#     for i, c in enumerate(contig_labels):
#         total_size[c] += len(contig_dict[data.index[i]])

#     # --- 2. CheckM Seeds ---
#     with tempfile.TemporaryDirectory() as tdir:
#         cfasta = os.path.join(tdir, 'concatenated.fna')
#         with open(cfasta, 'wt') as concat_out:
#             for ix, h in enumerate(data.index):
#                 bin_ix = contig_labels[ix]
#                 if total_size[bin_ix] < minfasta:
#                     continue
#                 concat_out.write(f'>bin{bin_ix:06}.{h}\n')
#                 concat_out.write(contig_dict[data.index[ix]] + '\n')

#         seeds = estimate_seeds(
#             cfasta,
#             binned_length,
#             num_process,
#             multi_mode=True,
#             orf_finder=orf_finder)
    
#     if not seeds:
#         logger.warning('No bins found in the concatenated fasta file.')
#         return contig_labels

#     name2ix = {name: ix for ix, name in enumerate(data.index)}
#     contig_labels_reclustered = np.empty_like(contig_labels)
#     contig_labels_reclustered.fill(-1)
#     next_label = 0

#     # --- 3. 循环处理 ---
#     for bin_ix in range(contig_labels.max() + 1):
#         contig_indices = [i for i, ell in enumerate(contig_labels) if ell == bin_ix]
#         if len(contig_indices) == 0:
#             continue

#         seed = seeds.get(f'bin{bin_ix:06}', [])
#         num_bin = len(seed)
        
#         # =======================================================
#         # 策略: Rank-based Filtering (排名截断法)
#         # =======================================================
#         if num_bin > 1 and total_size[bin_ix] >= minfasta:
#             logger.info(f"--- Processing Contaminated Bin {bin_ix} ---")
            
#             # 1. 惩罚系数: 50 (能分出 Bin3 的最佳参数)
#             penalty_val = 25
#             # 2. 正则化: 0.01 (防止过拟合)
#             reg_covar_val = 0.1
#             # 3. 保留比例: 0.7 (Top 70%)
#             # 这意味着我们会无条件丢弃最差的 30% 数据
#             keep_ratio = 0.65
            
#             n_components_final = num_bin 
            
#             logger.info(f"  [STRATEGY] Top {keep_ratio*100}% Percentile Cut. Penalty: {penalty_val}")

#             # A. 特征构建
#             if not is_combined:
#                 local_penalty = penalty_val 
#                 weight = local_penalty * base * math.ceil(scaling / base)
#                 bin_emb = embedding[contig_indices]
#                 bin_depth = depth_mean[contig_indices]
#                 re_bin_features = np.concatenate((bin_emb, bin_depth * weight), axis=1)
#             else:
#                 re_bin_features = embedding[contig_indices]

#             # B. Seed Init
#             means_init = None
#             global_seed_indices = [name2ix[s] for s in seed]
#             local_seed_indices = []
#             for gsi in global_seed_indices:
#                 try:
#                     local_seed_indices.append(contig_indices.index(gsi))
#                 except ValueError:
#                     continue
            
#             if len(local_seed_indices) == num_bin:
#                 means_init = re_bin_features[local_seed_indices]
#                 logger.info(f"  [ANCHOR] Initialized with {num_bin} biological seeds.")
#             else:
#                 logger.warning(f"  [WARNING] Seeds missing. Fallback to random.")

#             # C. 运行 GMM
#             gmm = GaussianMixture(
#                 n_components=n_components_final, 
#                 covariance_type='full',
#                 means_init=means_init,
#                 reg_covar=reg_covar_val, 
#                 n_init=1 if means_init is not None else 20,              
#                 random_state=random_seed
#             )
            
#             try:
#                 gmm.fit(re_bin_features)
#             except Exception as e:
#                 logger.error(f"  [ERROR] GMM failed: {e}. Skipping.")
#                 for idx in contig_indices:
#                     contig_labels_reclustered[idx] = next_label
#                 next_label += 1
#                 continue

#             # D. [核心逻辑修改] 基于排名的动态截断
#             probs = gmm.predict_proba(re_bin_features)
#             predicted_labels = probs.argmax(axis=1)
#             max_probs = probs.max(axis=1)
            
#             # 初始化一个全 False 的 mask
#             keep_mask = np.zeros(len(contig_indices), dtype=bool)
            
#             # 对每个新生成的簇单独处理
#             for k in range(n_components_final):
#                 # 找到属于簇 k 的所有 contig 的局部索引
#                 cluster_member_indices = np.where(predicted_labels == k)[0]
                
#                 if len(cluster_member_indices) == 0:
#                     continue
                
#                 # 获取它们的概率分数
#                 cluster_probs = max_probs[cluster_member_indices]
                
#                 # 获取它们的长度
#                 cluster_lengths = np.array([len(contig_dict[data.index[contig_indices[i]]]) 
#                                             for i in cluster_member_indices])
                
#                 # 1. 按概率从高到低排序
#                 # argsort 返回的是从小到大的索引，所以要 [::-1] 反转
#                 sorted_idx = np.argsort(cluster_probs)[::-1]
                
#                 # 2. 计算累积长度
#                 sorted_lens = cluster_lengths[sorted_idx]
#                 cum_len = np.cumsum(sorted_lens)
#                 total_len = cum_len[-1]
                
#                 # 3. 确定截断点 (总长度的 70%)
#                 cutoff_bp = total_len * keep_ratio
                
#                 # 找到截断位置
#                 # searchsorted 找到第一个大于 cutoff_bp 的位置
#                 cutoff_idx = np.searchsorted(cum_len, cutoff_bp)
                
#                 # 4. 选出 Top Contigs
#                 # 注意：cutoff_idx 是数量，我们要取 sorted_idx 的前 cutoff_idx + 1 个
#                 # 至少保留 1 个 (Seed)
#                 valid_count = max(1, min(len(sorted_idx), cutoff_idx + 1))
                
#                 # 获取保留下来的局部索引
#                 indices_to_keep_local = sorted_idx[:valid_count]
#                 indices_to_keep_global = cluster_member_indices[indices_to_keep_local]
                
#                 # 标记为 True
#                 keep_mask[indices_to_keep_global] = True
                
#                 # 日志记录截断处的概率值（看看门槛到底是多少）
#                 min_prob_kept = cluster_probs[sorted_idx[valid_count-1]]
#                 logger.info(f"    -> Sub-Bin {next_label + k}: Kept Top {keep_ratio*100}% bp "
#                             f"({valid_count}/{len(cluster_member_indices)} contigs). "
#                             f"Dynamic Threshold = {min_prob_kept:.4f}")

#             # 统计丢弃
#             dropped_count = (~keep_mask).sum()
#             original_bp = sum(len(contig_dict[data.index[i]]) for i in contig_indices)
#             dropped_bp = sum(len(contig_dict[data.index[contig_indices[i]]]) 
#                              for i in range(len(contig_indices)) if not keep_mask[i])
            
#             logger.info(f"  [RESULT] Total Dropped: {dropped_count} contigs ({dropped_bp / original_bp:.1%} of total bp)")

#             for i, (label, keep) in enumerate(zip(predicted_labels, keep_mask)):
#                 if keep:
#                     contig_labels_reclustered[contig_indices[i]] = next_label + label
            
#             next_label += n_components_final
#             logger.info("------------------------------------------------")

#         # =======================================================
#         # 正常 Bin 保护
#         # =======================================================
#         else:
#             if total_size[bin_ix] >= minfasta:
#                 for idx in contig_indices:
#                     contig_labels_reclustered[idx] = next_label
#                 next_label += 1
#             else:
#                 pass

#     assert contig_labels_reclustered.min() >= -1
#     return contig_labels_reclustered



# def recluster_bins(logger, data, *,
#                 n_sample,
#                 embedding,
#                 is_combined: bool,
#                 contig_labels,
#                 minfasta,
#                 contig_dict,
#                 binned_length,
#                 orf_finder,
#                 num_process,
#                 random_seed):
#     from sklearn.mixture import GaussianMixture 
#     import numpy as np
#     import math
#     from collections import defaultdict
#     import tempfile
#     import os

#     # --- 1. 基础数据准备 ---
#     if not is_combined:
#         depth = data.values[:, 136:len(data.values[0])].astype(np.float32)
#         mean_index = [2 * temp for temp in range(n_sample)]
#         depth_mean = depth[:, mean_index]
#         abun_scale = np.ceil(depth_mean.mean(axis=0) / 100) * 100
#         abun_scale = np.clip(abun_scale, 1e-6, None)
#         depth_mean = depth_mean / abun_scale
#         scaling = np.mean(np.abs(embedding)) / np.mean(depth_mean)
#         base = 10
    
#     total_size = defaultdict(int)
#     for i, c in enumerate(contig_labels):
#         total_size[c] += len(contig_dict[data.index[i]])

#     # --- 2. CheckM Seeds ---
#     with tempfile.TemporaryDirectory() as tdir:
#         cfasta = os.path.join(tdir, 'concatenated.fna')
#         with open(cfasta, 'wt') as concat_out:
#             for ix, h in enumerate(data.index):
#                 bin_ix = contig_labels[ix]
#                 if total_size[bin_ix] < minfasta:
#                     continue
#                 concat_out.write(f'>bin{bin_ix:06}.{h}\n')
#                 concat_out.write(contig_dict[data.index[ix]] + '\n')

#         seeds = estimate_seeds(
#             cfasta,
#             binned_length,
#             num_process,
#             multi_mode=True,
#             orf_finder=orf_finder)
    
#     if not seeds:
#         logger.warning('No bins found in the concatenated fasta file.')
#         return contig_labels

#     name2ix = {name: ix for ix, name in enumerate(data.index)}
#     contig_labels_reclustered = np.empty_like(contig_labels)
#     contig_labels_reclustered.fill(-1)
#     next_label = 0

#     # --- 3. 循环处理 ---
#     for bin_ix in range(contig_labels.max() + 1):
#         contig_indices = [i for i, ell in enumerate(contig_labels) if ell == bin_ix]
#         if len(contig_indices) == 0:
#             continue

#         seed = seeds.get(f'bin{bin_ix:06}', [])
#         num_bin = len(seed)
        
#         # =======================================================
#         # 策略: Regularized GMM (正则化GMM)
#         # =======================================================
#         if num_bin > 1 and total_size[bin_ix] >= minfasta:
#             logger.info(f"--- Processing Contaminated Bin {bin_ix} ---")
            
#             # 1. 惩罚系数: 150 (保持高压)
#             penalty_val = 50
            
#             # 2. 概率阈值: 提升到 0.95 (配合正则化进行严格切割)
#             prob_threshold = 0.96
            
#             # 3. 正则化因子: [关键新增] 1e-3
#             # 防止过拟合，强制让模型对垃圾数据产生犹豫
#             reg_covar_val = 0.01
            
#             n_components_final = num_bin 
            
#             logger.info(f"  [PARAM] Penalty: {penalty_val}, Threshold: {prob_threshold}, Reg_Covar: {reg_covar_val}")

#             # A. 特征构建
#             if not is_combined:
#                 local_penalty = penalty_val 
#                 weight = local_penalty * base * math.ceil(scaling / base)
#                 bin_emb = embedding[contig_indices]
#                 bin_depth = depth_mean[contig_indices]
#                 re_bin_features = np.concatenate((bin_emb, bin_depth * weight), axis=1)
#             else:
#                 re_bin_features = embedding[contig_indices]

#             # B. Seed Init
#             means_init = None
#             global_seed_indices = [name2ix[s] for s in seed]
#             local_seed_indices = []
#             for gsi in global_seed_indices:
#                 try:
#                     local_seed_indices.append(contig_indices.index(gsi))
#                 except ValueError:
#                     continue
            
#             if len(local_seed_indices) == num_bin:
#                 means_init = re_bin_features[local_seed_indices]
#                 logger.info(f"  [ANCHOR] Initialized with {num_bin} biological seeds.")
#             else:
#                 logger.warning(f"  [WARNING] Seeds missing locally. Fallback to random.")

#             # C. 运行 GMM (加入 reg_covar)
#             gmm = GaussianMixture(
#                 n_components=n_components_final, 
#                 covariance_type='full',
#                 means_init=means_init,
#                 reg_covar=reg_covar_val, # <--- 关键修改
#                 n_init=1 if means_init is not None else 20,              
#                 random_state=random_seed
#             )
            
#             try:
#                 gmm.fit(re_bin_features)
#             except Exception as e:
#                 logger.error(f"  [ERROR] GMM failed: {e}. Skipping.")
#                 for idx in contig_indices:
#                     contig_labels_reclustered[idx] = next_label
#                 next_label += 1
#                 continue

#             # D. 过滤与赋值
#             probs = gmm.predict_proba(re_bin_features)
#             predicted_labels = probs.argmax(axis=1)
#             max_probs = probs.max(axis=1)
            
#             # [调试信息] 打印概率分布，看看模型到底有多自信
#             min_p, mean_p, median_p = np.min(max_probs), np.mean(max_probs), np.median(max_probs)
#             logger.info(f"  [DEBUG] Prob Stats -> Min: {min_p:.4f}, Mean: {mean_p:.4f}, Median: {median_p:.4f}")
            
#             keep_mask = max_probs > prob_threshold
            
#             # 统计
#             dropped_count = (~keep_mask).sum()
#             original_bp = sum(len(contig_dict[data.index[i]]) for i in contig_indices)
#             dropped_bp = sum(len(contig_dict[data.index[contig_indices[i]]]) 
#                              for i in range(len(contig_indices)) if not keep_mask[i])
            
#             logger.info(f"  [RESULT] Dropped {dropped_count} contigs ({dropped_bp / original_bp:.1%} of total bp)")

#             for k in range(n_components_final):
#                 mask_k = (predicted_labels == k) & keep_mask
#                 count_k = mask_k.sum()
#                 if count_k > 0:
#                      logger.info(f"    -> Sub-Bin {next_label + k}: {count_k} contigs")

#             for i, (label, keep) in enumerate(zip(predicted_labels, keep_mask)):
#                 if keep:
#                     contig_labels_reclustered[contig_indices[i]] = next_label + label
            
#             next_label += n_components_final
#             logger.info("------------------------------------------------")

#         # =======================================================
#         # 正常 Bin 保护
#         # =======================================================
#         else:
#             if total_size[bin_ix] >= minfasta:
#                 for idx in contig_indices:
#                     contig_labels_reclustered[idx] = next_label
#                 next_label += 1
#             else:
#                 pass

#     assert contig_labels_reclustered.min() >= -1
#     return contig_labels_reclustered


# def recluster_bins(logger, data, *,
#                 n_sample,
#                 embedding,
#                 is_combined: bool,
#                 contig_labels,
#                 minfasta,
#                 contig_dict,
#                 binned_length,
#                 orf_finder,
#                 num_process,
#                 random_seed):
#     from sklearn.cluster import KMeans
#     from sklearn.metrics import silhouette_samples # [新增] 引入轮廓系数
#     import numpy as np
#     import math
#     from collections import defaultdict
#     import tempfile
#     import os
#     from scipy.spatial.distance import cdist


#     # --- 辅助函数：基于几何中心和轮廓系数的强力清洗 ---
#     def purify_cluster(features, labels, centers):
#         """
#         双重清洗策略：
#         1. Silhouette Filter: 剔除处于交界处的模糊点
#         2. Distance Filter: 剔除虽然在簇内但离中心太远的离群点
#         """
#         # 1. 计算轮廓系数 (范围 -1 到 1)
#         # 值越接近 1，说明聚类越紧密；接近 0 说明在边界；负数说明分错了
#         try:
#             sil_values = silhouette_samples(features, labels)
#         except Exception:
#             # 如果只有一个簇或者数据太少，无法计算轮廓系数，全保留
#             sil_values = np.ones(len(labels))

#         # [严格阈值]：只保留轮廓系数大于 0.1 的点
#         # 这意味着我们丢弃所有边界模糊的点
#         sil_mask = sil_values > 0.1 

#         # 2. 距离过滤 (针对通过了轮廓系数的点再次筛选)
#         final_mask = np.zeros(len(labels), dtype=bool)
        
#         unique_labels = np.unique(labels)
#         for k in unique_labels:
#             # 找到该簇的所有点
#             class_member_mask = (labels == k)
            
#             # 结合轮廓系数筛选
#             combined_mask = class_member_mask & sil_mask
            
#             if np.sum(combined_mask) == 0:
#                 continue

#             # 获取这部分核心数据的特征
#             cluster_data = features[combined_mask]
#             center = centers[k].reshape(1, -1)
            
#             # 计算到中心的距离
#             dists = cdist(cluster_data, center, metric='euclidean').ravel()
            
#             # [距离阈值]：剔除最远的 10% (保留 90%)
#             # 这种分位数过滤非常稳健，专门切除长尾巴
#             threshold = np.percentile(dists, 90) 
            
#             # 最终保留的索引
#             keep_indices = np.where(combined_mask)[0][dists <= threshold]
#             final_mask[keep_indices] = True
            
#         return final_mask

#     # --- 主逻辑开始 ---
    
#     # 1. 基础数据准备
#     if not is_combined:
#         depth = data.values[:, 136:len(data.values[0])].astype(np.float32)
#         mean_index = [2 * temp for temp in range(n_sample)]
#         depth_mean = depth[:, mean_index]
#         abun_scale = np.ceil(depth_mean.mean(axis=0) / 100) * 100
#         abun_scale = np.clip(abun_scale, 1e-6, None)
#         depth_mean = depth_mean / abun_scale
        
#         scaling = np.mean(np.abs(embedding)) / np.mean(depth_mean)
#         base = 10
    
#     total_size = defaultdict(int)
#     for i, c in enumerate(contig_labels):
#         total_size[c] += len(contig_dict[data.index[i]])

#     # 2. 预备 CheckM (Estimate Seeds)
#     with tempfile.TemporaryDirectory() as tdir:
#         cfasta = os.path.join(tdir, 'concatenated.fna')
#         with open(cfasta, 'wt') as concat_out:
#             for ix, h in enumerate(data.index):
#                 bin_ix = contig_labels[ix]
#                 if total_size[bin_ix] < minfasta:
#                     continue
#                 concat_out.write(f'>bin{bin_ix:06}.{h}\n')
#                 concat_out.write(contig_dict[data.index[ix]] + '\n')

#         seeds = estimate_seeds(
#             cfasta,
#             binned_length,
#             num_process,
#             multi_mode=True,
#             orf_finder=orf_finder)
    
#     if not seeds:
#         logger.warning('No bins found in the concatenated fasta file.')
#         return contig_labels

#     name2ix = {name: ix for ix, name in enumerate(data.index)}
#     contig_labels_reclustered = np.empty_like(contig_labels)
#     contig_labels_reclustered.fill(-1)
#     next_label = 0

#     # 3. 遍历处理每个 Bin
#     for bin_ix in range(contig_labels.max() + 1):
#         contig_indices = [i for i, ell in enumerate(contig_labels) if ell == bin_ix]
        
#         if len(contig_indices) == 0:
#             continue

#         seed = seeds.get(f'bin{bin_ix:06}', [])
#         num_bin = len(seed)

#         # =======================================================
#         # 策略 A: 针对顽固污染 Bin 的处理 (num_bin > 1)
#         # =======================================================
#         if num_bin > 1 and total_size[bin_ix] >= minfasta:
#             logger.info(f"Bin {bin_ix} implies contamination (Seeds: {num_bin}). Applying SILHOUETTE PURIFICATION.")
            
#             # --- 构建特征 ---
#             if not is_combined:
#                 # 依然保持高权重，但这回我们不完全依赖它
#                 # 如果之前 20 没用，这里尝试 10，防止 Depth 噪声太大反而误导
#                 local_penalty = 10 
#                 weight = local_penalty * base * math.ceil(scaling / base)
#                 bin_emb = embedding[contig_indices]
#                 bin_depth = depth_mean[contig_indices]
#                 re_bin_features = np.concatenate((bin_emb, bin_depth * weight), axis=1)
#             else:
#                 re_bin_features = embedding[contig_indices]

#             # --- 聚类 ---
#             length_weight = np.array([len(contig_dict[data.index[i]]) for i in contig_indices])
            
#             # 增加 n_init 到 20，让 KMeans 多跑几次找最优解
#             kmeans = KMeans(
#                 n_clusters=num_bin,
#                 n_init=20, 
#                 random_state=random_seed)
            
#             kmeans.fit(re_bin_features, sample_weight=length_weight)
            
#             # --- [核心]：调用净化函数 ---
#             # 这会返回一个 bool 数组，True 表示保留，False 表示丢弃
#             keep_mask = purify_cluster(re_bin_features, kmeans.labels_, kmeans.cluster_centers_)

#             # --- 赋值 ---
#             # 只有 keep_mask 为 True 的 Contig 才有资格获得新标签
#             for i, (label, keep) in enumerate(zip(kmeans.labels_, keep_mask)):
#                 if keep:
#                     contig_labels_reclustered[contig_indices[i]] = next_label + label
#                 # else: 丢弃 (label 保持 -1)
            
#             next_label += num_bin

#         # =======================================================
#         # 策略 B: 正常 Bin 的保护 (num_bin <= 1)
#         # =======================================================
#         else:
#             if total_size[bin_ix] >= minfasta:
#                 for idx in contig_indices:
#                     contig_labels_reclustered[idx] = next_label
#                 next_label += 1
#             else:
#                 pass

#     assert contig_labels_reclustered.min() >= -1
#     return contig_labels_reclustered




# def recluster_bins(logger, data, *,
#                 n_sample,
#                 embedding,
#                 is_combined: bool,
#                 contig_labels,
#                 minfasta,
#                 contig_dict,
#                 binned_length,
#                 orf_finder,
#                 num_process,
#                 random_seed):
#     from sklearn.cluster import KMeans
#     import numpy as np
#     from collections import defaultdict
#     if not is_combined:
#         depth = data.values[:, 136:len(data.values[0])].astype(np.float32)
#         mean_index = [2 * temp for temp in range(n_sample)]
#         depth_mean = depth[:, mean_index]
#         abun_scale = np.ceil(depth_mean.mean(axis = 0) / 100) * 100
#         depth_mean = depth_mean / abun_scale
#         scaling = np.mean(np.abs(embedding)) / np.mean(depth_mean)
#         base = 10
#         # weight = 2 * base * math.ceil(scaling / base)
#         contamination_penalty = 5  # 可以尝试 2.0 到 5.0 之间
#         weight = contamination_penalty * base * math.ceil(scaling / base) 
        
#         embedding_new = np.concatenate(
#             (embedding, depth_mean * weight), axis=1)
#     else:
#         embedding_new = embedding


#     total_size = defaultdict(int)
#     for i, c in enumerate(contig_labels):
#         total_size[c] += len(contig_dict[data.index[i]])
#     with tempfile.TemporaryDirectory() as tdir:
#         cfasta = os.path.join(tdir, 'concatenated.fna')
#         with open(cfasta, 'wt') as concat_out:
#             for ix,h in enumerate(data.index):
#                 bin_ix = contig_labels[ix]
#                 if total_size[bin_ix] < minfasta:
#                     continue
#                 concat_out.write(f'>bin{bin_ix:06}.{h}\n')
#                 concat_out.write(contig_dict[data.index[ix]] + '\n')

#         seeds = estimate_seeds(
#             cfasta,
#             binned_length,
#             num_process,
#             multi_mode=True,
#             orf_finder=orf_finder)
#             # we cannot bypass the orf_finder here, because of the renaming of the contigs
#         if seeds == []:
#             logger.warning('No bins found in the concatenated fasta file.')
#             return contig_labels

#     name2ix = {name:ix for ix,name in enumerate(data.index)}
#     contig_labels_reclustered = np.empty_like(contig_labels)
#     contig_labels_reclustered.fill(-1)
#     next_label = 0
#     for bin_ix in range(contig_labels.max() + 1):
#         seed = seeds.get(f'bin{bin_ix:06}', [])
#         num_bin = len(seed)

#         if num_bin > 1 and total_size[bin_ix] >= minfasta:
#             contig_indices = [i for i,ell in enumerate(contig_labels) if ell == bin_ix]
#             re_bin_features = embedding_new[contig_indices]

#             seed_index = [name2ix[s] for s in seed]
#             length_weight = np.array(
#                 [len(contig_dict[name])
#                     for name,ell in zip(data.index, contig_labels)
#                         if ell == bin_ix])
#             seeds_embedding = embedding_new[seed_index]
#             kmeans = KMeans(
#                 n_clusters=num_bin,
#                 init=seeds_embedding,
#                 n_init=1,
#                 random_state=random_seed)
#             kmeans.fit(re_bin_features, sample_weight=length_weight)
#             for i, label in enumerate(kmeans.labels_):
#                 contig_labels_reclustered[contig_indices[i]] = next_label + label
#             next_label += num_bin
#         else:
#             contig_labels_reclustered[contig_labels == bin_ix] = next_label
#             next_label += 1
#     assert contig_labels_reclustered.min() >= 0
#     return contig_labels_reclustered

