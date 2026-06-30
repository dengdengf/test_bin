# 本项目相对 SemiBin2 的改动

**摘要**：Multimodal SemiBin 是在 [SemiBin2](https://github.com/BigDataBiology/SemiBin)（v2.2.0，© BigDataBiology，MIT 许可）基础上扩展的多模态宏基因组分箱工具。命令行入口、子命令、输入输出与上游完全兼容（`SemiBin2` 入口，向后兼容 `SemiBin`），本页只说明**本项目相对上游 SemiBin2 新增/修改的部分**。沿用 SemiBin2 的基础用法请见 [usage.md](usage.md)、[generate.md](generate.md)、[subcommands.md](subcommands.md)。

本项目派生自 SemiBin/SemiBin2，保留对上游的致谢与引用（见文末 [引用](#引用)）。

---

## 四点核心不同

下面四点是本项目相对 SemiBin2 的全部改动，**仅在短读长（`short_read`）路径生效**；长读长（`--sequencing-type=long_read` / `bin_long`）沿用上游的 DBSCAN 集成算法，本项目未改动。

| 改动 | 上游 SemiBin2 的做法 | 本项目的做法 | 相关文件 |
| --- | --- | --- | --- |
| 1. 多模态嵌入 | 组成 + 丰度特征的自监督嵌入 | 组成 / 丰度 / DNABERT 三分支编码 + 学习式 softmax 门控融合 + 跨模态对齐损失（向 DNABERT 单向对齐，stop-gradient/detach） | `SemiBin/multimodal_model.py` |
| 2. 图融合聚类 | embedding ∩ k-mer 取交集，再乘共丰度 KL 深度矩阵 | 对 embedding/组成/丰度/(可选)DNABERT 各建 kNN 相似度图，加权融合后跑 Infomap 全局聚类；多样本下逐边重新引入共丰度 KL 散度调制 | `SemiBin/graph_fusion.py`、`SemiBin/cluster.py` |
| 3. 去污染重聚类 | 用标记基因种子初始化 KMeans，无条件拆分污染 bin | 污染 bin 内做带种子的标签传播（personalized-PageRank），仅当单拷贝标记基因冗余度下降时才接受拆分 | `SemiBin/marker_refinement.py` |
| 4. DNABERT 特征 | 无 | 批量推理 + attention_mask 掩码均值池化；whole 与 split 共享同一 PCA basis | `SemiBin/generate_berts.py` |

### 1. 多模态嵌入模型

`SemiBin/multimodal_model.py` 把原来单一的组成 + 丰度自监督嵌入，扩展为三个独立分支：

- **组成分支**：136 维 canonical 四核苷酸（k-mer 组成，维度与归一化逻辑同上游）。
- **丰度分支**：BAM/CRAM 推导的覆盖度（归一化逻辑同上游）。
- **DNABERT 分支**：由 `generate_berts.py` 生成的序列语言模型嵌入（见下文）。

三个分支经各自编码后，由一个**学习式 softmax 门控**动态加权融合；同时加入**跨模态对齐损失**，让组成 / 丰度分支单向对齐到 DNABERT 分支（对 DNABERT 一侧使用 stop-gradient / `detach`，避免反向污染语言模型特征）。

> 该路径仅在短读长训练时启用；可用 `--disable-multimodal-training` 关闭，回退到标准自监督。

### 2. 多视图相似度图融合聚类

`SemiBin/graph_fusion.py` 与 `SemiBin/cluster.py`：对每一种视图（embedding、组成、丰度，以及可选的 DNABERT）分别构建 kNN 相似度图，按权重**加权融合**成单一图后，跑 **Infomap** 做全局社区发现聚类。

- 融合权重与核函数可配置（见 [新增命令行参数](#新增命令行参数)）。
- 在**多样本（非 combined）**场景下，重新引入上游的「共丰度 KL 散度」作为**边权调制**，但改为**逐边计算**以省内存。
- 对比上游：SemiBin2 是先对 embedding 图与 k-mer 图取**交集**，再整体乘以共丰度 KL 深度矩阵；本项目是多视图加权融合 + Infomap。

### 3. 标记基因去污染重聚类

`SemiBin/marker_refinement.py`：对被判定为污染（单拷贝标记基因冗余）的 bin 做精细化拆分。

- 用**带种子的标签传播**（personalized-PageRank）代替上游的 KMeans：
  - α（重启概率）随 bin 大小自适应；
  - 扩散时按 contig 长度加权；
  - 用 top-2 置信度的**边际**把边界 contig 留作未分配，而非强行归属。
- **仅当**拆分后单拷贝标记基因冗余度**下降**时才接受拆分；否则保留原 bin。
- 对比上游：SemiBin2 用标记基因种子初始化 KMeans 并**无条件**拆分。

### 4. DNABERT 特征提取

`SemiBin/generate_berts.py`：用 DNABERT-S 序列语言模型为每条 contig 生成嵌入。

- 批量推理，使用 **attention_mask 掩码均值池化**。
- `whole` 与 `split` 两份序列**共享同一个 PCA basis**：在 whole 上 `fit`，对 split 用 `transform`，保证两者落在同一子空间。

DNABERT-S 预训练权重较大，已 gitignore，**不随仓库分发**；需单独获取（来源：[DNABERT_S](https://github.com/MAGICS-LAB/DNABERT_S)）放到 `SemiBin/DNABERT-S/`，或用 `--dnabert-model` 指定。详见下文 [DNABERT 用法要点](#dnabert-用法要点)。

---

## 为什么这样改

- **去污染（改动 3）**：上游对污染 bin 无条件 KMeans 拆分，可能把本就纯净的 bin 拆坏。改为「冗余度下降才接受」的带种子标签传播，只在确有收益时动手，且把没把握的边界 contig 留作未分配，减少误拆。
- **共丰度（改动 2）**：多样本数据里，同一基因组的 contig 在样本间的覆盖度分布应当一致。逐边的共丰度 KL 调制把这种跨样本信号重新注入图边权，且逐边计算避免了上游整块深度矩阵相乘的内存开销。
- **多模态（改动 1、4）**：组成与丰度特征对近缘物种区分力有限。引入 DNABERT 序列语言模型特征，并用门控融合 + 单向对齐让它补充而非覆盖原有信号，从而在保持兼容的前提下提升嵌入判别力。

---

## 新增命令行参数

### DNABERT / 训练（`single_easy_bin`、`multi_easy_bin`）

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--dnabert-model PATH` | 内置 `SemiBin/DNABERT-S` 目录 | DNABERT-S 权重路径 |
| `--dnabert-python PATH` | `$SEMIBIN_DNABERT_PYTHON` 或当前解释器 | 运行 DNABERT 推理的 Python 解释器 |
| `--disable-multimodal-training` | 关闭（即默认启用多模态） | 关闭多模态，回退标准自监督 |

### 图融合 / 聚类（`single_easy_bin`、`multi_easy_bin`、`bin`）

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--knn-kernel {median,local}` | `median` | kNN 相似度图的核函数 |
| `--fusion-weights EMB COMP ABUND` | `0.60 0.25 0.15` | 无 DNABERT 时各视图融合权重 |
| `--fusion-weights-multimodal EMB COMP ABUND DNA` | `0.45 0.15 0.15 0.25` | 含 DNABERT 时各视图融合权重 |
| `--no-coabundance-kl` | 关闭（即默认启用 KL 调制） | 关闭共丰度 KL 边权调制 |

---

## DNABERT 用法要点

- DNABERT-S 预训练权重较大，已 gitignore，不随仓库分发；需单独获取放到 `SemiBin/DNABERT-S/`，或用 `--dnabert-model` 指定。来源：<https://github.com/MAGICS-LAB/DNABERT_S>。
- DNABERT 嵌入用 `generate_berts.py` 生成，需 **whole + split 两份**（split 名称形如 `h_1` / `h_2`，见 `generate_kmer.py`），二者共享 PCA basis。
- 推荐命令：

```bash
python SemiBin/generate_berts.py -md /path/DNABERT-S \
  -fd whole.fasta -nd output/dnabert_contig_names.txt -dd output/dnabert_embedding.npy \
  -sfd split.fasta -snd output/dnabert_split_contig_names.txt -sdd output/dnabert_split_embedding.npy
```

- 输出文件名**必须**是 `dnabert_embedding.npy` / `dnabert_split_embedding.npy`，放在 `data.csv` 同目录；fasta 行序须与 `data.csv` / `data_split.csv` **一致**（`load_multimodal_embeddings` 会逐行校验）。
- `single_easy_bin` / `multi_easy_bin` 内置的 `--dnabert-model` 自动提取路径，对 split 半段有局限（原始 fasta 里没有 `h_1` / `h_2`），推荐用 `generate_berts.py` 显式生成。

---

## 沿用 SemiBin2、未改动的部分

以下与上游一致，可照常使用（详见 [usage.md](usage.md)、[generate.md](generate.md)、[output.md](output.md)、[aemb.md](aemb.md)）：

- **输入**：contigs（组装结果）+ BAM/CRAM（或 strobealign-aemb 丰度）。
- **预训练模型** `--environment`：`human_gut` / `dog_gut` / `ocean` / `soil` / `cat_gut` / `human_oral` / `mouse_gut` / `pig_gut` / `built_environment` / `wastewater` / `chicken_caecum` / `global`。
- **长读长**：`--sequencing-type=long_read` 或 `bin_long`，沿用 DBSCAN 集成算法（本项目未改动）。
- **k-mer 组成** = 136 维 canonical 四核苷酸；丰度归一化逻辑未改。
- **外部依赖**：bedtools、hmmer、samtools（可选 mmseqs2、prodigal）。Python 3.7–3.13。
- 子命令与 SemiBin2 相同：`single_easy_bin` / `multi_easy_bin` / `bin` / `bin_long` / `generate_sequence_features_single` / `generate_sequence_features_multi` / `generate_cannot_links` / `concatenate_fasta` / `train` / `train_self` / `download_GTDB` / `citation` 等（见 [subcommands.md](subcommands.md)）。

---

## 致谢与引用

本项目派生自 SemiBin / SemiBin2（© BigDataBiology，MIT 许可），请保留对上游的引用：

- Pan, Zhu, Zhao, Coelho. *A deep siamese neural network improves metagenome-assembled genomes in microbiome datasets across different environments.* **Nat Commun** 13, 2326 (2022). <https://doi.org/10.1038/s41467-022-29843-y>
- Pan, Zhao, Coelho. *SemiBin2: self-supervised contrastive learning leads to better MAGs for short- and long-read sequencing.* **Bioinformatics** 39(Suppl_1): i21–i29 (2023). <https://doi.org/10.1093/bioinformatics/btad209>

如果使用了 DNABERT 特征，请额外引用 **DNABERT-S**（<https://github.com/MAGICS-LAB/DNABERT_S>）。
