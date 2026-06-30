# 更新记录

## MAGFuse 更新说明

**MAGFuse** 是一个独立的多模态宏基因组分箱工具，融合组成（k-mer）、丰度与 DNABERT 序列嵌入三种模态，并以图融合聚类替代传统的"仅用 k-mer 组成 + 丰度"的分箱思路。命令行入口、子命令与输入输出格式见 [usage](usage)、[subcommands](subcommands)、[output](output)；核心方法与特性见 [methods](methods)。

### 四点核心设计

| # | 模块 | 设计 |
|---|------|------|
| 1 | 多模态嵌入模型<br>(`SemiBin/multimodal_model.py`) | 组成 / 丰度 / DNABERT 三分支编码 + 学习式 softmax 门控融合 + 跨模态对齐损失（向 DNABERT 单向对齐，使用 stop-gradient/detach）。**短读长 (`short_read`) 训练路径默认且始终启用。** |
| 2 | 多视图相似度图融合聚类<br>(`SemiBin/graph_fusion.py` + `cluster.py`) | 对 embedding / 组成 / 丰度 / DNABERT 各建 kNN 相似度图，加权融合后用 **Leiden** 做全局社区检测；融合权重、核函数与聚类算法/分辨率可配置。在多样本（非 combined）下引入"共丰度 KL 散度"边权调制（逐边计算，省内存）。 |
| 3 | 标记基因去污染重聚类<br>(`SemiBin/marker_refinement.py`) | 在被判为污染的 bin 内做"带种子的标签传播"(personalized-PageRank，α 随 bin 大小自适应、按 contig 长度加权扩散、用 top-2 置信度边际把边界 contig 留作未分配)；**仅当单拷贝标记基因冗余度下降时才接受拆分。** |
| 4 | DNABERT 特征提取<br>(`SemiBin/generate_berts.py`) | 批量推理 + `attention_mask` 掩码均值池化；`whole` 与 `split` **共享同一个 PCA basis**（在 `whole` 上 fit、对 `split` 用 transform）。 |

### 命令行参数

DNABERT / 训练（`single_easy_bin`、`multi_easy_bin`）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dnabert-model PATH` | 内置 `SemiBin/DNABERT-S` 目录 | DNABERT-S 模型路径 |
| `--dnabert-python PATH` | `$SEMIBIN_DNABERT_PYTHON` 或当前解释器 | 运行 DNABERT 推理的 Python 解释器 |

图融合 / 聚类（`single_easy_bin`、`multi_easy_bin`、`bin`）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--knn-kernel {median,local}` | `median` | kNN 相似度图的核函数 |
| `--fusion-weights EMB COMP ABUND` | `0.60 0.25 0.15` | 无 DNABERT 时的融合权重 |
| `--fusion-weights-multimodal EMB COMP ABUND DNA` | `0.45 0.15 0.15 0.25` | 有 DNABERT 时的融合权重 |
| `--no-coabundance-kl` | （开启）| 关闭共丰度 KL 调制 |
| `--cluster-algorithm {leiden,infomap}` | `leiden` | 全局社区检测算法（infomap 仅作为可选项保留）|
| `--cluster-resolution FLOAT` | `1.0` | Leiden 模块度分辨率（越大 bin 越多、越小）|

### 关键要点

- **共丰度 KL 信号**：在多样本（非 combined）图融合中把"共丰度 KL 散度"作为边权调制引入，逐边计算以节省内存（不构造全量 KL 深度矩阵）。
- **DNABERT 共享 PCA basis**：`whole` 与 `split` 不各自 fit PCA，而是在 `whole` 上 fit、对 `split` 用 transform，保证两份嵌入位于同一子空间。
- **对齐损失 stop-gradient**：跨模态对齐为向 DNABERT 的单向对齐，DNABERT 分支用 detach 冻结，避免对齐项反过来污染 DNABERT 表征。
- **重聚类长度加权 / 自适应 / 置信度剔除**：标签传播按 contig 长度加权扩散、α 随 bin 大小自适应，并用 top-2 置信度边际把边界 contig 留作未分配；只有在单拷贝标记基因冗余度下降时才接受拆分。

### DNABERT 用法要点

- DNABERT-S 预训练权重较大，已 `gitignore`，**不随仓库分发**；需单独获取后放到 `SemiBin/DNABERT-S/`，或用 `--dnabert-model` 指定。来源：<https://github.com/MAGICS-LAB/DNABERT_S>。
- DNABERT 嵌入需 `whole` + `split` 两份（split 名称形如 `h_1`/`h_2`，见 `generate_kmer.py`），二者共享 PCA basis。`single_easy_bin` / `multi_easy_bin` 会**自动提取**（含 split，共享 PCA basis），`generate_berts.py` 是离线/单独生成嵌入的等价工具。
- `generate_berts.py` 用法：

```bash
python SemiBin/generate_berts.py -md /path/DNABERT-S \
  -fd whole.fasta -nd output/dnabert_contig_names.txt -dd output/dnabert_embedding.npy \
  -sfd split.fasta -snd output/dnabert_split_contig_names.txt -sdd output/dnabert_split_embedding.npy
```

- 输出文件名必须是 `dnabert_embedding.npy` / `dnabert_split_embedding.npy`，放在 `data.csv` 同目录；fasta 行序须与 `data.csv` / `data_split.csv` 一致（`load_multimodal_embeddings` 会逐行校验）。
- `single_easy_bin` / `multi_easy_bin` 会自动提取 whole 与 split（`h_1`/`h_2` 由父 contig 自动切半），并共享 PCA basis，无需手动。

更多端到端用法见 [usage](usage)、[generate](generate)、[aemb](aemb)。

### 未改动的部分

- 输入：contigs（组装结果）+ BAM/CRAM（或 strobealign-aemb 丰度）。
- `--environment` 预训练模型：`human_gut` / `dog_gut` / `ocean` / `soil` / `cat_gut` / `human_oral` / `mouse_gut` / `pig_gut` / `built_environment` / `wastewater` / `chicken_caecum` / `global`。
- 长读长：`--sequencing-type=long_read` 或 `bin_long`，沿用 DBSCAN 集成算法（不使用多模态，属于不同算法）。
- k-mer 组成 = 136 维 canonical 四核苷酸；丰度归一化逻辑未改。
- 外部依赖：bedtools、hmmer、samtools（可选 mmseqs2、prodigal）。支持 Python 3.7–3.13。

---

# 历史版本记录

> 下列为分箱流程的历史版本记录，保留以便追溯。

## Version 2.2.0

*Released Mar 20, 2025*

This is a maintenance release with many small improvement rather than a single big new feature. Upgrading is recommended, but not crucial.

### User-visible changes
- Better logging: Always log to file in DEBUG level and log command-line arguments. Print version number in logs.
- Better error messages in several instances
- check_install: Prints out information on the GPU

### Deprecations
- Deprecate `--prodigal-output-faa` argument
- No longer check for `mmseqs` in `check_install` (it is not a hard requirement)

### Internal improvements and bugfixes
- Respect the number of threads requested better
- Better method to save the model which is more compatible with newer versions of PyTorch. Added a subcommand to update old models to the new format (`update_model`)
- Switch to pixi for testing (and recommend it in the README/[installation](install) instructions)
- Convert to `pyproject.toml` instead of `setup.py`
- Do not fail if no bins are produced

## Version 2.1.0

*Released Mar 6, 2024*

Main new feature is adding support for using output of strobealign-aemb.

### User-visible changes

- Support running with [strobealign-aemb](https://github.com/ksahlin/strobealign/releases/tag/v0.13.0) (`--abundance`/`-a`)
- Add `citation` subcommand

### Internal improvements
- Code simplification and refactor
- deprecation: Deprecate --orf-finder=fraggenescan option
- Update abundance normalization

### Bugfixes
- Do not use more processes than can be taken advantage of

## Version 2.0.2

*Released Oct 31, 2023*

### Bugfix release

Fixes issue with `multi_easy_bin --write-pre-reclustering-bins`

## Version 2.0.1

*Released Oct 21, 2023*

This is a bugfix release for _version 2.0.0_.

## Version 2.0.0

*Released Oct 20, 2023*

### User-visible changes

- A log file is now written in the output directory
- The `concatenate_fasta` subcommand now supports compression
- Adds `bin_short` subcommand as alias for `bin` (by analogy with `bin_long`)


## Version 1.5.1

*Released Mar 7, 2023*

### Bugfixes

- Fix use of `--no-recluster` with multi_easy_bin.

## Version 1.5.0

*Released Jan 17, 2023*

### User-visible improvements

- Added a new option for ORF finding, called `fast-naive` which is an internal very fast implementation.
- Added the possibility of bypassing ORF finding altogether by providing prodigal outputs directly (or any other gene prediction in the right format)
- Command line argument checking is more exhaustive instead of exiting at first error
- Added `--quiet` flag to reduce the amount of output printed
- Better `--help` (group required arguments separately)
- Add `--output-compression` option to compress outputs
- Add `--tag-output` option which allows for control of the output filenames (and also makes the anvi'o compatible).
- Add contig->bin mapping table

## Version 1.4.0: long reads binning!

*Released December 15, 2022*

Big change is the added binning algorithm for assemblies from long-read datasets.

When clustering, it does not use infomap, but another procedure (an iterative version of DBSCAN).

Use the flag `--sequencing-type=long_read` to enable an alternative clustering that works better with long reads.

### Other user-visible improvements

- Better error checking at multiple steps in the pipeline so that processes that will crash are caught as early as possible
- Add `--allow-missing-mmseqs2` flag to `check_install` subcommand (eventually, self-supervision will be the default and mmseqs2 will be an optional dependency)

### Command line parameter deprecations

The previous arguments should continue to work, but going forward, the newer arguments are probably a better API.

- Selecting self-supervised learning is now done with the `--self-supervised` flag (instead of `--training-type=self`)
- Training from multiple samples is now enabled with the `--train-from-many` flag (instead of `--mode=several`)

### Bugfixes

- The output table sometimes had the wrong path in `v1.3`. This has been fixed
- Prodigal is now run in a more robust manner when using multiple threads

## Version 1.3.1

*Release December 9, 2022*

### Bugfixes

- Made `--training-type` argument optional (defaults to `semi` to keep backwards compatibility)


## Version 1.3.0

*Released November 4 2022*

### User visible improvements

- Added _self-supervised learning mode_ (see [Training models](training) for more details)

### Bugfixes

- Fix output table to contain correct paths
- Fix mispelling in argument name `--epochs` (the old variation, `--epoches` is still accepted for backwards compatibility, but should be considered deprecated)

## Version 1.2.0

*Released October 19 2022*

### User visible improvements

- Pretrained model from chicken caecum
- Output table with basic information on bins (including N50 & L50)
- When reclustering is used (default), output the unreclusted bins into a directory called `output_prerecluster_bins`
- Added `--verbose` flag and silenced some of the output when it is not used
- Use coloredlogs (if package is available)

## Version 1.1.1

*Released September 27 2022*

### Bugfixes

- Completely remove use of `atomicwrites` package

## Version 1.1.0

*Released September 21 2022*

### User-visible improvements

- Support .cram format input
- Support using depth file from Metabat2
- More flexible specification of prebuilt models (case insensitive, normalize `-` and `_`)
- Better output message when no bins are produced

### Bugfixes

- Fix bug using `atomicwrite` on certain network filesystems

### Internal improvements

- Remove torch version restriction (and test on Python 3.10)


## Version 1.0.3

*Released August 3 2022*

### Bugfixes

- Fix coverage parsing when value is not an integer
- Fix multi_easy_bin with taxonomy file given on the command line


## Version 1.0.2

*Released July 8 2022*

### Bugfixes

- Fix more thoroughly

## Version 1.0.1

*Released May 9 2022*

### Bugfixes

- Fix edge case when calling prodigal with more threads than contigs

## Version 1.0.0

*Released April 29 2022*

### User-visible improvements

- More balanced file split when calling prodigal in parallel should take better advantage of multiple threads
- Fix bug when long stretches of Ns are present
- Better error messages

### Bugfixes

- Fix bugs in training from multiple samples
- Fix bug in incorporating CAT results

## Version 0.7

*Released March 2 2022*

This release solves issues running on Mac OS X.

### User-visible improvements

- Improved `check_install` command: it now prints out paths and correctly handles optionality of FragGeneScan/prodigal
- Add `concatenate_fasta` command to combine fasta files for multi-sample binning
- Add option `--tmpdir` to set temporary directory
- Substitute FragGeneScan with Prodigal (FragGeneScan can still be used with `--orf-finder` parameter). FragGeneScan caused issues, especially on Mac OSX

### Internal improvements
- Reuse `markers.hmmout` file to make the training from several samples faster

## Version 0.6

*Released February 7 2022*

### User-visible improvements
- Provide pretrained models from soil, cat gut, human oral, pig gut, mouse gut, built environment, wastewater and global (training from all samples).
- Users can now pass in the output of running mmseqs2 directly and it will use that instead of calling mmseqs itself (use option `--taxonomy-annotation-table`).
- The subcommand to generate cannot links is now called `generate_cannot_links`. The old name (`predict_taxonomy`) is kept as a deprecated alias.
- Similarly, sequence features (_k_-mer and abundance) are generated using the commands `generate_sequence_features_single` and `generate_sequence_features_multi` (for single- and multi-sample modes, respectively). The old names (`generate_data_single`/`generate_data_multi`) are kept as deprecated aliases.
- Add `check_install` command and run `check_install` before easy command

### Bugfixes
- Fix bug with non-standard characters in sample names.

## Version 0.5

*Released January 7 2022*

### User-visible improvements
- Reclustering is now the default (use `--no-recluster` to disable it; the option `--recluster` is deprecated and ignored) as the computational costs are much lower
- GTDB lazy downloading is now performed even if a non-standard directory is used
- The [CACHEDIR.TAG](https://bford.info/cachedir/) protocol was implemented (this is supported by several tools that perform tasks such as backups).

### Bugfixes
- Fix bug with `--min-len` (minimal length). Previously, only contigs greater than the given minimal length were used (instead of greater-equal to the minimal length).
- GTDB downloading was inconsistent in a few instances which have been fixed

### Internal improvements
- Much more efficient code (including lower memory usage) for binning, especially if a pretrained model is used.

## Version 0.4.0

*Released 27 October 2021*

### User-visible improvements
- Add support for `.xz` FASTA files as input

### Internal improvements
- Removed BioPython dependency

### Bug fixes
- Fix bug when uncompressing FASTA files
- Fix bug when splitting data

## Version 0.3

*Released 10 August 2021*

### User-visible improvements
- Support training from several samples
- Remove `output_bin_path` if `output_bin_path` exists
- Make several internal parameters configuable: (1) minimum length of contigs to bin (`--min-len` parameter); (2) minimum length of contigs to break up in order to generate _must-link_ constraints (`--ml-threshold` parameter); (3) the ratio of the number of base pairs of contigs between 1000-2500 bp smaller than this value, the minimal length will be set as 1000bp, otherwise 2500bp (`--ratio` parameter).
- Add `-p` argument for `predict_taxonomy` mode

### Internal improvements
- Better code overall
- Fix `np.concatenate` warning
- Remove redundant matrix when clustering
- Better pretrained models
- Faster calculating dapth using Numpy
- Use correct number of threads in `kneighbors_graph()`

### Bugfixes

- Respect number of threads (`-p` argument) when training

## Version 0.2

*Release 27 May 2021*

### User-visible improvements
- Add support for training with several samples
- Test with Python 3.9
- Download mmseqs database with `--remove-tmp-file 1`
- Better output names
- Fix bugs when paths have spaces
- Fix installation issues by listing all the dependencies
- Add `download_GTDB` command
- Add `--recluster` option
- Add `--environment` option
- Add `--mode` option

### Internal improvements
- All around more robust code by including more error checking & testing
- Better built-in models

## Version 0.1.1

*Released 21 March 2021*

**Bugfix release** fixing an issue with `minfasta-kbs`

## Version 0.1

*Released 21 March 2021*

- First release: testing version
