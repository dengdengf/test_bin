# MAGFuse 的输出说明

本页说明 MAGFuse 运行后在输出目录中生成的文件。MAGFuse 是一个独立的多模态宏基因组分箱工具，下面会逐一说明分箱结果、模型与训练数据、以及多模态/图融合/去污染相关的输出文件。

相关文档页：[安装](install.md)、[使用](usage.md)、[核心方法与特性](methods.md)、[generate 子命令](generate.md)、[全部子命令](subcommands.md)、[训练](training.md)、[DNABERT/多模态特征](aemb.md)、[FAQ](faq.md)。

## 单样本 / 共组装分箱 (single_easy_bin)

运行 `MAGFuse single_easy_bin` 后，输出目录中主要文件如下。

### 分箱结果

| 文件 / 目录 | 说明 |
| --- | --- |
| `output_bins/` | 最终重建出的全部 bin（每个 bin 一个 fasta 文件）。默认即为最终结果。 |
| `output_recluster_bins/` | 启用标记基因去污染重聚类后产生的去污染 bin 目录（见下文「去污染重聚类输出」）。 |
| `bins_info.tsv` | 每个 bin 的基本信息表：bin 名称、总碱基数、contig 数、N50、L50 等（未来可能新增列）。 |
| `contig_bins.tsv` | contig 到 bin 的映射表：每行给出一个 contig 及其所属 bin。 |

### 模型与训练数据

| 文件 | 说明 |
| --- | --- |
| `model.pt` | 训练好的深度学习模型。短读长 (`short_read`) 路径下为多模态嵌入模型（组成 / 丰度 / DNABERT 三分支 + 门控融合 + 跨模态对齐），始终默认启用。 |
| `data.csv` | 训练所用的 whole（完整 contig）特征：136 维 canonical 四核苷酸 k-mer 组成 + 归一化丰度。 |
| `data_split.csv` | 把 contig 切成两半（命名形如 `h_1` / `h_2`，见 `generate_kmer.py`）后的 split 特征，用于自监督约束。 |
| `*_data_cov.csv` / `*_data_split_cov.csv` | 从 depth（BAM/CRAM 或 strobealign-aemb）生成的覆盖度/丰度数据。 |
| `cannot/cannot.txt` | 训练使用的 cannot-link 约束文件。 |

### DNABERT 特征文件（多模态路径）

短读长路径默认且始终启用多模态，需要 DNABERT 嵌入，存放在与 `data.csv` 同一目录下：

| 文件 | 说明 |
| --- | --- |
| `dnabert_embedding.npy` | whole contig 的 DNABERT 嵌入（经掩码均值池化 + PCA 降维）。行序须与 `data.csv` 完全一致。 |
| `dnabert_split_embedding.npy` | split 半段 (`h_1`/`h_2`) 的 DNABERT 嵌入。行序须与 `data_split.csv` 完全一致；与 whole 共享同一个 PCA basis（在 whole 上 fit、对 split 用 transform）。 |

> 文件名必须严格为 `dnabert_embedding.npy` / `dnabert_split_embedding.npy`，且 fasta 行序须与 `data.csv` / `data_split.csv` 逐行对齐 —— `load_multimodal_embeddings` 会逐行校验。`single_easy_bin` / `multi_easy_bin` 会自动提取 whole 与 split（`h_1`/`h_2` 由父 contig 自动切半），并共享 PCA basis，无需手动；`generate_berts.py` 是离线/单独生成嵌入的等价工具。生成方法见 [DNABERT/多模态特征](aemb.md)。

这两个嵌入文件也可由 `SemiBin/generate_berts.py` 离线/单独显式生成，命令示例：

```bash
python SemiBin/generate_berts.py -md /path/DNABERT-S \
  -fd whole.fasta -nd output/dnabert_contig_names.txt -dd output/dnabert_embedding.npy \
  -sfd split.fasta -snd output/dnabert_split_contig_names.txt -sdd output/dnabert_split_embedding.npy
```

DNABERT-S 预训练权重较大，已被 gitignore、不随仓库分发，需单独获取（来源：<https://github.com/MAGICS-LAB/DNABERT_S>）后放到 `SemiBin/DNABERT-S/`，或用 `--dnabert-model` 指定路径。

## 去污染重聚类输出 (output_recluster_bins/)

标记基因去污染重聚类 (`SemiBin/marker_refinement.py`) 在被判为污染的 bin 内做「带种子的标签传播」（personalized-PageRank，α 随 bin 大小自适应、按 contig 长度加权扩散，并用 top-2 置信度边际把边界 contig 留作未分配），**仅当单拷贝标记基因冗余度下降时才接受拆分**。

| 目录 / 文件 | 说明 |
| --- | --- |
| `output_recluster_bins/` | 去污染重聚类后的最终 bin 目录。未被判为污染、或拆分未带来冗余度下降的 bin 原样保留。 |

去污染重聚类是否启用，由命令行选项控制（详见 [使用](usage.md)）。

## 多样本分箱 (multi_easy_bin)

| 目录 / 文件 | 说明 |
| --- | --- |
| `bins/` | 来自所有样本的重建 bin 汇总。 |
| `samples/*.fasta` | 由输入的合并 contig（`concatenate_fasta` 产物）拆分出的每个样本的 contig fasta。 |
| `samples/*_data_cov.csv` | 每个样本的覆盖度数据，含义同单样本路径。 |
| `samples/{sample-name}/` | 每个样本各自的输出子目录，结构与单样本/共组装分箱一致（包含该样本的 `model.pt`、`data.csv`、`data_split.csv`、`output_bins/`、`bins_info.tsv`、`contig_bins.tsv`，以及 `dnabert_embedding.npy` / `dnabert_split_embedding.npy`）。 |

> 多样本（非 combined）路径下，图融合聚类会额外引入「共丰度 KL 散度」边权调制（逐边计算，省内存）；这只影响聚类过程，不会新增持久化输出文件。可用 `--no-coabundance-kl` 关闭。融合权重/核函数等聚类选项见 [使用](usage.md)。

## 全局社区检测与聚类参数

图融合后用 **Leiden** 做全局社区检测（模块度目标）。相关命令行参数（作用于 `single_easy_bin` / `multi_easy_bin` / `bin`）：

| 参数 | 说明 |
| --- | --- |
| `--cluster-resolution FLOAT` | Leiden 模块度分辨率，默认 `1.0`；越大 bin 越多、越小 bin 越少。 |
| `--knn-kernel` / `--fusion-weights` / `--fusion-weights-multimodal` | 图融合的核函数与各模态融合权重。 |
| `--no-coabundance-kl` | 关闭多样本路径下的共丰度 KL 散度边权调制。 |
