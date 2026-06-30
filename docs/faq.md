# 常见问题（FAQ）

本页收集 MAGFuse 使用中的常见问题。命令行入口为 `MAGFuse`。

相关页面：[index.md](index.md)、[install.md](install.md)、[usage.md](usage.md)、[generate.md](generate.md)、[output.md](output.md)、[training.md](training.md)、[subcommands.md](subcommands.md)。

---

## 多模态相关

### DNABERT-S 模型从哪里获取？如何放置？

DNABERT-S 的预训练权重文件较大，已在 `.gitignore` 中排除，**不随本仓库分发**，需要单独获取。

- 来源：[MAGICS-LAB/DNABERT_S](https://github.com/MAGICS-LAB/DNABERT_S)
- 放置方式（二选一）：
  1. 放到仓库内置目录 `SemiBin/DNABERT-S/`（`single_easy_bin` / `multi_easy_bin` 的 `--dnabert-model` 默认指向此处）；
  2. 放到任意位置，运行时用 `--dnabert-model PATH` 指定。

如果你的 DNABERT 推理需要独立的 Python 环境，可用 `--dnabert-python PATH` 指定解释器（默认读取环境变量 `$SEMIBIN_DNABERT_PYTHON`，否则用当前解释器）。

### 如何生成 DNABERT 嵌入？

`single_easy_bin` / `multi_easy_bin` 会**自动**提取 DNABERT 特征，且**自动处理 split 半段**：同时提取 whole 与 split（split 的序列名形如 `h_1` / `h_2`，由父 contig 自动切半），并共享同一个 PCA basis，**无需手动**操作。

`SemiBin/generate_berts.py` 是离线/单独生成嵌入的等价工具。它做批量推理 + `attention_mask` 掩码均值池化，并需要 **whole + split 两份** 输入（split 的序列名形如 `h_1` / `h_2`，参见 `generate_kmer.py`）。

```bash
python SemiBin/generate_berts.py -md /path/DNABERT-S \
  -fd whole.fasta -nd output/dnabert_contig_names.txt -dd output/dnabert_embedding.npy \
  -sfd split.fasta -snd output/dnabert_split_contig_names.txt -sdd output/dnabert_split_embedding.npy
```

注意事项：

- 输出文件名**必须**是 `dnabert_embedding.npy` 和 `dnabert_split_embedding.npy`，放在 `data.csv` 同目录下。
- fasta 的行序必须与 `data.csv` / `data_split.csv` 一致（`load_multimodal_embeddings` 会逐行校验）。
- `single_easy_bin` / `multi_easy_bin` 内置的自动提取路径会自动生成 whole 与 split（含 `h_1` / `h_2`）两份嵌入并共享 PCA basis；`generate_berts.py` 是离线/单独生成同样两份嵌入的等价工具。

### whole 与 split 为什么要共享同一个 PCA basis？

`generate_berts.py` 在 **whole 上 fit** 出 PCA basis，再用同一个 basis 对 **split 做 transform**。这样 whole 与 split 的 DNABERT 特征位于同一坐标系中，下游融合与对齐才有意义；如果各自独立做降维，两份嵌入的维度方向不可比，会破坏后续的相似度图与对齐损失。

### DNABERT 多模态总是启用吗？

是。DNABERT 多模态是 MAGFuse **短读长**（`short_read`）流程**默认且始终启用**的核心组成部分，DNABERT 特征提取与多模态训练是直通的常规步骤，无需任何开关。

长读长不走多模态（见下文）。

### 融合权重和 `--knn-kernel` 怎么调？

多视图相似度图融合聚类（`graph_fusion.py` + `cluster.py`）对 embedding / 组成 / 丰度 /（可选）DNABERT 各建一张 kNN 相似度图，加权融合后用 **Leiden** 做全局社区检测。相关参数对 `single_easy_bin` / `multi_easy_bin` / `bin` 子命令可用：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--knn-kernel {median,local}` | `median` | kNN 相似度的核函数。`median` 用全局中位距离作带宽，较稳健；`local` 用每点局部近邻距离自适应带宽。 |
| `--fusion-weights EMB COMP ABUND` | `0.60 0.25 0.15` | **非**多模态时三视图（embedding / 组成 / 丰度）的融合权重。 |
| `--fusion-weights-multimodal EMB COMP ABUND DNA` | `0.45 0.15 0.15 0.25` | 多模态时四视图（embedding / 组成 / 丰度 / DNABERT）的融合权重。 |
| `--no-coabundance-kl` | 关闭 | 关闭多样本（非 combined）下的"共丰度 KL 散度"边权调制。 |
| `--cluster-resolution FLOAT` | `1.0` | Leiden 模块度分辨率；越大 bin 越多、越小。 |

调参提示：

- 权重为各视图的相对贡献，可按你对各模态质量的信任度调整；DNABERT 的 `--fusion-weights-multimodal` 始终生效。
- `--no-coabundance-kl` 仅影响多样本（非 combined）场景。该 KL 调制按逐边计算以节省内存；样本数较多时若不需要可关闭。
- 数据点分布很不均匀时可尝试 `--knn-kernel local`。
- `--cluster-resolution` 越大、bin 越多越小，越小、bin 越少越大；可据目标基因组数量微调。

### 长读长会走多模态吗？

**不会。** 长读长（`--sequencing-type=long_read` 或 `bin_long` 子命令）使用 DBSCAN 集成聚类算法，**不启用** DNABERT 多模态嵌入与图融合。多模态仅在短读长路径下生效。

---

## 通用问题

### MAGFuse 支持长读长数据吗？

支持。使用 `--sequencing-type=long_read`（或 `bin_long` 子命令）即可，走的是长读长 DBSCAN 聚类算法。注意此时不会启用多模态特性。

### 我有混合数据（短读长 + 长读长）怎么办？

一般按长读长流程处理，即使用 `--sequencing-type=long_read`。注意此时不会启用多模态特性。

### MAGFuse 能用于真核基因组吗？

技术上可以，并能产出 bin，但工具并非为此优化，基准测试均基于原核数据。

- 长读长算法依赖原核单拷贝基因，尤其不建议用于真核数据。
- 短读长场景下，重聚类同样依赖这套基因，处理真核数据时应关闭（`--no-recluster`）。

可以考虑把它作为多算法组合（配合 dereplication）的一环，但不建议单独用于真核数据。

### 能用其它版本的 GTDB 做注释吗？

> **注意**：这仅与已弃用的半监督注释流程相关。

可以，有两种方式：

1. 下载 mmseqs 格式的 GTDB（`mmseqs databases GTDB GTDB tmp` 会下载最新版本），然后用 `--reference-db-data-dir` 指向该数据库。
2. 用任意版本的 GTDB 预先用 mmseqs 计算 contig 注释，再用 `--taxonomy-annotation-table` 把注释表传入。注意工具期望的是 mmseqs 格式文件，格式不符会产生无意义结果。

第二种方式较复杂，但当 contig 的分类注释本身需要服务于更大的流程（不只为本工具）时较为合理。
