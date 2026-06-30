## 子命令参考

本页详尽列出 MAGFuse 的各个子命令及其参数。命令行入口为 `MAGFuse`。需要更易读的整体用法说明，请参见 [usage](usage.md)；输入文件的生成方式见 [generate](generate.md)。

> MAGFuse 提供多模态嵌入、多视图图融合聚类、标记基因去污染重聚类与 DNABERT 特征提取等能力。这些能力只影响**短读长 (short_read)** 训练与聚类路径；长读长路径采用 DBSCAN 集成算法。核心方法见 [methods](methods.md)，更新说明见 [whatsnew](whatsnew.md)。

MAGFuse 采用**子命令**式接口。大多数场景只需 `single_easy_bin` 或 `multi_easy_bin`，若需要更精细的控制，可使用其它子命令分步执行。

### single_easy_bin

用一条命令完成单样本或共组装 (co-assembly) 分箱。

`single_easy_bin` 以 contig 文件（reads 的组装结果）和 BAM 文件（reads 比对回 contig）作为输入，将重建出的 bin 输出到 `output_recluster_bins` 目录（输入文件的生成方式见 [generate](generate.md)，整体用法见 [usage](usage.md)）。

#### 必需参数

* `-i/--input-fasta`：输入 contig fasta 文件路径（支持 `gzip` 与 `bzip2` 压缩）。
* `-b/--input-bam`：输入 BAM（`.bam`）或 CRAM（`.cram`）文件路径。可传入多个 BAM 文件，每个样本一个。
* `-o/--output`：输出目录（不存在时会自动创建）。
* `-a/--abundance`：来自 strobealign-aemb 的丰度文件路径。仅当参与分箱的样本数 ≥ 5 时可用。

#### 推荐参数

如果你的数据来自我们已有预训练模型的生境，使用 `--environment` 参数将直接复用该模型，而不必重新训练。

* `--environment`：内置模型对应的环境（`human_gut`/`dog_gut`/`ocean`/`soil`/`cat_gut`/`human_oral`/`mouse_gut`/`pig_gut`/`built_environment`/`wastewater`/`chicken_caecum`/`global`）。

若未指定 `--environment`，则会训练一个新模型，计算开销较大。

* `--self-supervised` 或 `--semi-supervised`：指定训练算法。自监督方式通常在效果与计算资源占用上更优；半监督方式属于较早的（已弃用）路线，参见 [semi-supervised](semi-supervised.md)。
* `--sequencing-type=short_reads`/`--sequencing-type=long_reads`：测序类型。**多模态扩展仅在 `short_reads` 路径生效**；`long_reads` 采用 DBSCAN 集成算法。

#### 多模态 / DNABERT 训练参数

短读长训练路径**默认且始终启用** DNABERT 多模态嵌入，它是 MAGFuse 短读长流程的核心组成部分。下列参数控制多模态嵌入与 DNABERT 特征分支，仅在短读长训练路径有效。多模态嵌入模型的结构见 `SemiBin/multimodal_model.py`（组成 / 丰度 / DNABERT 三分支编码 + 学习式 softmax 门控融合 + 向 DNABERT 单向对齐的跨模态对齐损失，对齐使用 stop-gradient/detach）。

* `--dnabert-model PATH`：DNABERT-S 模型目录（默认：内置 `SemiBin/DNABERT-S` 目录）。预训练权重较大，已 gitignore，不随仓库分发，需单独获取放到该目录或用本参数指定。权重来源见下文“DNABERT 用法要点”。
* `--dnabert-python PATH`：用于运行 DNABERT 推理的 Python 解释器（默认：环境变量 `$SEMIBIN_DNABERT_PYTHON`，否则使用当前解释器）。

> 说明：`single_easy_bin` 内置的 `--dnabert-model` 会自动提取 DNABERT 嵌入，并自动处理 split 半段：`single_easy_bin` / `multi_easy_bin` 会自动提取 whole 与 split（`h_1`/`h_2` 由父 contig 自动切半），并共享 PCA basis，无需手动操作。`SemiBin/generate_berts.py` 是离线/单独生成嵌入的等价工具，见下文。

#### 图融合 / 聚类参数

下列参数控制多视图相似度图融合聚类（`SemiBin/graph_fusion.py` 与 `SemiBin/cluster.py`）。对 embedding / 组成 / 丰度 / DNABERT 各建一张 kNN 相似度图，按权重融合后用 **Leiden** 做全局社区检测。

* `--knn-kernel {median,local}`：kNN 相似度图所用的核函数（默认 `median`）。
* `--cluster-resolution FLOAT`：Leiden 模块度分辨率（默认 `1.0`；越大 bin 越多、越小 bin 越少）。
* `--fusion-weights EMB COMP ABUND`：未启用 DNABERT 时的三路融合权重（默认 `0.60 0.25 0.15`，依次对应 embedding / 组成 / 丰度）。
* `--fusion-weights-multimodal EMB COMP ABUND DNA`：启用 DNABERT 时的四路融合权重（默认 `0.45 0.15 0.15 0.25`，依次对应 embedding / 组成 / 丰度 / DNABERT）。
* `--no-coabundance-kl`：关闭“共丰度 KL 散度”边权调制。默认在多样本（非 combined）场景下会逐边计算共丰度 KL 散度对融合图的边权进行调制（逐边计算以节省内存）。

#### 控制输出的可选参数

* `--compression`：是否压缩输出以节省空间，取值之一：`none` / `gz`（默认）/ `xz` / `bz2`。
* `--tag-output`：若传入（如 `--tag-output=mysample`），输出的 bin 文件名会带上该标签，便于区分多次运行的结果。

#### 控制计算资源占用的可选参数

* `-p/--processes/-t/--threads`：使用的 CPU 数（默认 `0` 表示使用全部 CPU）。
* `--write-pre-reclustering-bins`/`--no-write-pre-reclustering-bins`：是否写出重聚类前的 bin（默认不写出）。
* `--engine`：训练所用设备（`auto`/`gpu`/`cpu`）；`auto`（默认）会尝试检测并使用 GPU，找不到则回退到 CPU。
* `--tmpdir`：设置临时目录。
* `-r/--reference-db-data-dir`：GTDB 参考目录（默认 `$HOME/.cache/SemiBin/mmseqs2-GTDB`）。仅在使用已弃用的半监督模式时有用；此时 MAGFuse 会在该路径找不到 GTDB 时惰性下载（占用较多磁盘空间）。

#### 设置内部参数的可选参数

* `--random-seed`：随机种子，用于复现结果。
* `--orf-finder`：用于估计 bin 数量的基因预测器，取值之一：`prodigal`、`fast-naive`（内置的极快实现，默认）、`fraggenescan`（比 `prodigal` 快，但不能在所有平台安装，且仍不及 `fast-naive`）。

#### 跳过内部步骤的可选参数

若你希望在 MAGFuse 之外自行计算某些内部步骤，可将其跳过。例如 mmseqs2 的 contig 注释耗时较长，若你已独立完成，可在此复用结果以避免重算。

这些属于高级用法，传入格式错误的文件容易导致次优或无意义的结果。

* `--taxonomy-annotation-table`：预先计算好的 mmseqs2 格式 taxonomy TSV 文件，用以跳过 mmseqs2 的 GTDB 注释。多样本分箱时，请确保 taxonomy TSV 与（用于 combined fasta 的）contig 文件顺序一致。
* `--depth-metabat2`：由 metabat2 生成的 depth 文件（仅用于单样本分箱）。
* `--prodigal-output-faa`：从用于分箱的 contig 预测出的蛋白编码基因。关键在于预测基因须按 `{contig}_{index}` 命名，其中 `{contig}` 为 contig 名、`{index}` 为某个 ORF 标识，二者以单个下划线分隔。Prodigal 采用该格式，但并非所有工具都如此。

#### 设置内部参数（高级）

通常应对以下参数使用默认值，此处列出仅供需要调参时使用。

* `--minfasta-kbs`：bin 的最小尺寸，单位 kbp（默认 200）。
* `--no-recluster`：不进行 bin 重聚类。这能节省少量时间，但重聚类前的 bin 始终会输出。
* `--epochs`：训练过程的 epoch 数（默认 15）。
* `--batch-size`：训练过程的 batch 大小（默认 2048）。
* `--max-node`：被纳入分箱的 contig 比例（默认 1）。
* `--max-edges`：单个 contig 可连接的最大边数（默认 200）。
* `--ratio`：若长度在 1000–2500 bp 之间的 contig 碱基总数占比小于该值，则最小长度设为 1000 bp，否则设为 2500 bp。若已设置 `-m` 则无需本参数。分步使用 MAGFuse 时，请在所有子命令中保持一致（默认 0.05）。
* `-m/--min-len`：分箱中 contig 的最小长度。分步使用 MAGFuse 时请在所有子命令中保持一致（默认由 MAGFuse 根据上述 1000–2500 bp 占比在 1000 bp / 2500 bp 间选择）。
* `--ml-threshold`：生成 must-link 约束的长度阈值。默认从 contig 中计算，最小默认值为 4000 bp。
* `--cannot-name`：cannot-link 文件名（默认 `cannot`）。

### multi_easy_bin

用一条命令完成多样本分箱。

`multi_easy_bin` 以多个样本合并后的 contig 文件和 BAM 文件（reads 比对回合并 contig）作为输入，将重建出的 bin 输出到 `samples/[sample]/output_recluster_bins` 目录。

#### 必需参数

* `-b/--input-bam`：输入 BAM（`.bam`）或 CRAM（`.cram`）文件路径。可传入多个 BAM 文件，每个样本一个。
* `--input-fasta` 与 `--output`：含义同 `single_easy_bin`。

#### 可选参数

* `-s/--separator`：多样本分箱时用于分隔样本名与 contig 名的字符（默认 `:`）。
* `--self-supervised` 或 `--semi-supervised`：指定训练算法，同 `single_easy_bin`。
* 多模态 / DNABERT 训练参数 `--dnabert-model`、`--dnabert-python`，含义同 `single_easy_bin`。
* 图融合 / 聚类参数 `--knn-kernel`、`--cluster-resolution`、`--fusion-weights`、`--fusion-weights-multimodal`、`--no-coabundance-kl`，含义同 `single_easy_bin`。注意：共丰度 KL 散度调制在多样本（非 combined）场景下默认启用，`--no-coabundance-kl` 可将其关闭。
* `--reference-db-data-dir`、`--processes`、`--minfasta-kbs`、`--epochs`、`--batch-size`、`--max-node`、`--max-edges`、`--random-seed`、`--ratio`、`--min-len`、`--ml-threshold`、`--no-recluster`、`--orf-finder`、`--engine` 与 `--tmpdir`，含义同 `single_easy_bin`。

### generate_cannot_links

:::{warning}
仅在使用较早的（已弃用）半监督方式时才需要，参见 [semi-supervised](semi-supervised.md)。
:::

使用 mmseqs 对 contig 做 GTDB 注释，并生成半监督深度学习训练所用的 `cannot-link` 文件。

`generate_cannot_links` 以 contig 文件为输入，输出 `cannot-link` 约束。

#### 必需参数

* `--input-fasta`
* `--output`

含义同 `single_easy_bin`。

#### 可选参数

* `--cannot-name`
* `-r/--reference-db-data-dir`
* `--ratio`
* `--min-len`
* `--ml-threshold`
* `--taxonomy-annotation-table`
* `--tmpdir`
* `-a/--abundance`

含义同 `single_easy_bin`。

### generate_sequence_features_single

`generate_sequence_features_single` 以 contig 文件与 BAM 文件为输入，为单样本及共组装分箱生成训练数据（`data.csv`、`data_split.csv`）。

#### 必需参数

* `-i/--input-fasta`
* `-b/--input-bam`
* `-o/--output`
* `-a/--abundance`

含义同 `single_easy_bin`。

#### 可选参数

* `-p/--processes/-t/--threads`
* `--ratio`
* `--min-len`
* `--ml-threshold`
* `--depth-metabat2`
* `--tmpdir`

含义同 `single_easy_bin`。

> 若要使用 DNABERT 多模态特征，需在该步生成的 `data.csv` / `data_split.csv` 同目录下，额外准备 `dnabert_embedding.npy` 与 `dnabert_split_embedding.npy`（用 `SemiBin/generate_berts.py` 生成），详见下文“DNABERT 用法要点”。

### generate_sequence_features_multi

`generate_sequence_features_multi` 以合并后的 contig 文件与 BAM 文件为输入，为多样本分箱生成训练数据（`data.csv` 与 `data_split.csv`）。

#### 必需参数

* `-i/--input-fasta`
* `-o/--output`
* `-b/--input-bam`
* `-a/--abundance`

含义同 `multi_easy_bin`。

#### 可选参数

* `-p/--processes/-t/--threads`、`--ratio`、`--min-len`、`--ml-threshold` 与 `--tmpdir`，含义同 `single_easy_bin`。
* `-s/--separator`，含义同 `multi_easy_bin`。

### train

`train` 以 contig 文件，以及 `generate_sequence_features_single`/`generate_sequence_features_multi` 与 `generate_cannot_links` 的输出（`data.csv`、`data_split.csv`、`cannot.txt`）为输入，输出训练好的模型。

注意：你可以用多个样本训练出一个供单样本分箱使用的模型。

#### 必需参数

* `-i/--input-fasta`（同 `single_easy_bin`）
* `-o/--output`（同 `single_easy_bin`）
* `--data`：输入 `data.csv` 文件路径（通常由先前的 `generate_sequence_features_single` 或 `generate_sequence_features_multi` 生成）。
* `--data_split`：输入 `data_split.csv` 文件路径。
* `-c/--cannot-link`：由其它生物学信息生成的 cannot-link 文件路径，每行一个约束，逗号分隔：`contig_1,contig_2`。
* `--train-from-many`：传入时，从多个样本训练模型（跨样本训练可得到更好的单样本分箱预训练模型）。使用该标志时，须按完全相同的顺序为各样本传入 `data`、`data_split`、`cannot`、`fasta`。*注意：* 仅在单样本分箱时可用，不支持与多样本分箱组合使用（详见 [training](training.md)）。

#### 可选参数

* `--epochs`
* `--batch-size`
* `-p/--processes/-t/--threads`
* `--random-seed`
* `--ratio`
* `--min-len`
* `--orf-finder`
* `--engine`

含义同 `single_easy_bin`。

### train_self

`train_self` 以 contig 文件，以及 `generate_sequence_features_single`/`generate_sequence_features_multi` 的输出（`data.csv`、`data_split.csv`）为输入，以自监督方式输出训练好的模型。

#### 必需参数

* `-o/--output`（同 `single_easy_bin`）
* `--data`：输入 `data.csv` 文件路径（通常由先前的 `generate_sequence_features_single` 或 `generate_sequence_features_multi` 生成）。
* `--data_split`：输入 `data_split.csv` 文件路径。
* `--train-from-many`：传入时，从多个样本训练模型（跨样本训练可得到更好的单样本分箱预训练模型）。使用该标志时，须按相同顺序为各样本传入 `data`、`data_split`、`fasta`。*注意：* 仅在单样本分箱时可用，不支持与多样本分箱组合使用（详见 [training](training.md)）。

#### 可选参数

* `--epochs`
* `--batch-size`
* `-p/--processes/-t/--threads`
* `--random-seed`
* `--engine`

含义同 `single_easy_bin`。

### bin_short

`bin_short`（为向后兼容，`bin` 作为其别名）以 contig 文件，以及 `generate_sequence_features_single`/`generate_sequence_features_multi` 与 `train` 的输出（`data.csv`、`model.pt`）为输入，将最终 bin 输出到 `output_recluster_bins` 目录。

#### 必需参数

* `--data`（同 `train`）
* `-i/--input-fasta`（同 `single_easy_bin`）
* `-o/--output`（同 `single_easy_bin`）

此外，下列两者须至少提供其一：

* `--environment`：使用哪个预训练模型（见 `single_easy_bin`）。
* `--model`：训练好的模型路径。

模型的生成方式见 [training](training.md)。

#### 可选参数

* 图融合 / 聚类参数 `--knn-kernel`、`--cluster-resolution`、`--fusion-weights`、`--fusion-weights-multimodal`、`--no-coabundance-kl`，含义同 `single_easy_bin`（这些参数影响短读长的图融合聚类与重聚类阶段）。
* `--minfasta-kbs`、`--max-node`、`--max-edges`、`-p/--processes/-t/--threads`、`--random-seed`、`--environment`、`--ratio`、`--min-len`、`--no-recluster`、`--orf-finder`、`--engine` 与 `--depth-metabat2`，含义同 `single_easy_bin`。

> 关于短读长的重聚类：MAGFuse 对被判为污染的 bin 采用“带种子的标签传播”（personalized-PageRank，α 随 bin 大小自适应、按 contig 长度加权扩散、用 top-2 置信度边际把边界 contig 留作未分配）进行去污染拆分，且仅当单拷贝标记基因冗余度下降时才接受拆分（见 `SemiBin/marker_refinement.py`）。

### bin_long

`bin_long` 以 contig 文件，以及 `generate_sequence_features_single`/`generate_sequence_features_multi` 与 `train` 的输出（`data.csv`、`model.pt`）为输入，将最终 bin 输出到 `output_bins` 目录。本子命令采用 DBSCAN 集成算法（多模态与图融合扩展不在长读长路径生效）。

#### 必需参数

* `--data`（同 `train`）
* `-i/--input-fasta`（同 `single_easy_bin`）
* `-o/--output`（同 `single_easy_bin`）

此外，下列两者须至少提供其一：

* `--environment`：使用哪个预训练模型（见 `single_easy_bin`）。
* `--model`：训练好的模型路径。

模型的生成方式见 [training](training.md)。

#### 可选参数

* `--minfasta-kbs`、`-p/--processes/-t/--threads`、`--random-seed`、`--environment`、`--ratio`、`--min-len`、`--orf-finder`、`--engine` 与 `--depth-metabat2`，含义同 `single_easy_bin`。

### download_GTDB

下载参考基因组（GTDB）。用于训练新模型时的半监督学习。

* `-r/--reference-db-data-dir`：GTDB 数据的存放位置（默认 `$HOME/.cache/SemiBin/mmseqs2-GTDB`）。
* `-f/--force`：即使该路径已有数据也强制下载（默认不下载）。

若将 GTDB 下载到非默认目录，则需在每个命令中通过 `-r` 传入该路径以确保被找到。

### check_install

检查所需依赖是否可用（排障时很有用）。外部依赖包括 bedtools、hmmer、samtools（可选 mmseqs2、prodigal）。

#### 可选参数

* `--allow-missing-mmseqs2`：使用时，找不到 `mmseqs` 不会报错。

### concatenate_fasta

为多样本分箱拼接 fasta 文件。contig 会被重命名为“样本名 + 分隔符 + 原名”。分隔符不能出现在任何样本的 contig 名中；若某样本含默认分隔符（`:`），须改用其它字符并在每个命令中通过 `--separator`/`-s` 传入。

#### 必需参数

* `-i`/`--input-fasta`（同 `single_easy_bin`）
* `-o`/`--output`（同 `single_easy_bin`）

#### 可选参数

* `-s`/`--separator`，含义同 `multi_easy_bin`（见上文说明）。
* `-m`：丢弃长度低于该值的序列（默认 0）。
* `--compression`：是否压缩输出（默认 `gz`）。

### citation

打印引用信息。

#### 可选参数

* `--bibtex`：使用 BibTeX 格式。
* `--ris`：使用 RIS 格式（适用于 Endnote 等工具）。
* `--chicago`：使用 Chicago 格式（默认）。

---

## DNABERT 用法要点

DNABERT 特征提取见 `SemiBin/generate_berts.py`：批量推理 + 用 attention_mask 做掩码均值池化；`whole` 与 `split` 共享同一份 PCA basis（在 `whole` 上 fit，对 `split` 用 transform）。

* DNABERT-S 预训练权重较大，已 gitignore，不随仓库分发。需单独获取并放到 `SemiBin/DNABERT-S/`，或用 `--dnabert-model` 指定。权重来源：<https://github.com/MAGICS-LAB/DNABERT_S>。
* DNABERT 嵌入需要 `whole` + `split` 两份。`split` 名称形如 `h_1`/`h_2`（见 `SemiBin/generate_kmer.py`），二者共享 PCA basis。
* 推荐命令：

```bash
python SemiBin/generate_berts.py -md /path/DNABERT-S \
  -fd whole.fasta -nd output/dnabert_contig_names.txt -dd output/dnabert_embedding.npy \
  -sfd split.fasta -snd output/dnabert_split_contig_names.txt -sdd output/dnabert_split_embedding.npy
```

* 输出文件名必须为 `dnabert_embedding.npy` / `dnabert_split_embedding.npy`，放在 `data.csv` 同目录；fasta 的行序须与 `data.csv` / `data_split.csv` 一致（`load_multimodal_embeddings` 会逐行校验）。
* `single_easy_bin`/`multi_easy_bin` 内置的 `--dnabert-model` 会自动提取（含 split，共享 PCA basis），无需手动操作；`generate_berts.py` 是离线/单独生成嵌入的等价工具。
