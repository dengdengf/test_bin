# 生成 MAGFuse 的输入文件

本页介绍如何从原始 reads 出发，准备 MAGFuse 所需的全部输入：

1. contigs（组装结果）+ 排序 BAM。
2. 用 `generate_sequence_features_single` / `generate_sequence_features_multi` 生成 `data.csv` 与 `data_split.csv`。
3. 用 `generate_berts.py` 生成 DNABERT 嵌入。

相关页面：[install.md](install.md) · [usage.md](usage.md) · [subcommands.md](subcommands.md) · [output.md](output.md) · [aemb.md](aemb.md) · [methods.md](methods.md)

---

## 1. 从 reads 得到 contigs 与排序 BAM

从一个宏基因组样本出发，你需要先得到一个 contigs 文件（`contig.fa`）和一个排序后的 BAM 文件（`output.sorted.bam`，由 reads 比对回组装好的 contigs 得到）。

**步骤 1**：将 reads 组装为 contigs FASTA。可用任意组装流程，例如用 [NGLess](https://ngless.embl.de/) 把 FastQ 预处理与组装（[MEGAHIT](https://academic.oup.com/bioinformatics/article/31/10/1674/177884)）合并到一个脚本里。

**步骤 2**：把 reads 比对回步骤 1 生成的 FASTA。

下面这个 `NGLess` 脚本一次性完成上述操作：

```python
ngless "1.5"
import "samtools" version "1.0"

input = paired('reads_1.fq.gz', 'reads_2.fq.gz')
input = preprocess(input) using |r|:
    r = substrim(r, min_quality=25)
    if len(r) < 45:
        discard

contigs = assemble(input)
write(contigs, ofile='contig.fa')

mapped = map(input, fafile=contigs)

write(samtools_sort(mapped),
    ofile='output.sorted.bam')
```

### 用 bowtie2 比对

你也可以直接用 `bowtie2`（下例用 4 线程，可按需调整 `-p 4`）：

```bash
bowtie2-build -f contig.fa contig.fa

bowtie2 -q --fr -x contig.fa -1 reads_1.fq.gz -2 reads_2.fq.gz -S contig.sam -p 4

samtools view -h -b -S contig.sam -o contig.bam
samtools view -b -F 4 contig.bam -o contig.mapped.bam
samtools sort contig.mapped.bam -o contig.mapped.sorted.bam

samtools index contig.mapped.sorted.bam
```

> 丰度也可以用 `strobealign-aemb` 提供，详见 [aemb.md](aemb.md)。

---

## 2. 生成序列特征：data.csv 与 data_split.csv

得到 contigs 和排序 BAM 后，用 `generate_sequence_features_*` 子命令计算每条 contig 的组成（k-mer）与丰度特征。输出会写入 `--output` 指定目录：

- `data.csv` —— 整条 contig 的特征（136 维 canonical 四核苷酸组成 + 丰度）。
- `data_split.csv` —— 每条 contig 被切成两半（命名形如 `h_1` / `h_2`，见 `SemiBin/generate_kmer.py`）后的特征，用于自监督学习的成对构造。

> 组成特征是 136 维 canonical 四核苷酸频率，并对丰度做归一化处理。

### 单样本（single）

```bash
MAGFuse generate_sequence_features_single \
    -i contig.fa \
    -b output.sorted.bam \
    -o output
```

### 多样本（multi）

多样本流程需要先把各样本的 contigs 拼接（contig 名加样本前缀），再统一比对，最后生成各样本的特征：

```bash
MAGFuse concatenate_fasta \
    -i sample1.fa sample2.fa sample3.fa \
    -o output

# 把各样本 reads 比对到 output/concatenated.fa.gz，得到每个样本的排序 BAM

MAGFuse generate_sequence_features_multi \
    -i output/concatenated.fa.gz \
    -b sample1.sorted.bam sample2.sorted.bam sample3.sorted.bam \
    -o output
```

`generate_sequence_features_multi` 会按样本前缀拆分，在 `output/samples/<sample>/` 下为每个样本各生成一份 `data.csv` 与 `data_split.csv`。

> 一般情况下直接用 `single_easy_bin` / `multi_easy_bin` 一键流程即可，它们内部会自动调用上述步骤。只有在你想手动控制各阶段（例如插入 DNABERT 嵌入）时，才需要单独运行这些子命令。详见 [usage.md](usage.md) 与 [subcommands.md](subcommands.md)。

---

## 3. 生成 DNABERT 嵌入

MAGFuse 在组成 / 丰度之外引入第三条模态：用 [DNABERT-S](https://github.com/MAGICS-LAB/DNABERT_S) 对 contig 序列做嵌入，再与组成、丰度一起送入多模态融合模型。该模态是短读长（`short_read`）训练路径**默认且始终启用**的组成部分。长读长（`long_read`）路径不使用多模态，而是采用 DBSCAN 集成，属于不同算法。

> DNABERT-S 预训练权重较大，已被 `.gitignore`，**不随仓库分发**。请自行从 <https://github.com/MAGICS-LAB/DNABERT_S> 获取权重，放到 `SemiBin/DNABERT-S/`，或在运行时用 `--dnabert-model` 指定路径。

### 3.1 自动提取（含 split）

`single_easy_bin` / `multi_easy_bin` 会自动提取 whole 与 split 两份嵌入：`h_1` / `h_2` 半段由父 contig 自动切半得到，whole 与 split 共享同一个 PCA basis，**无需手动**处理。`generate_berts.py` 则是离线 / 单独生成嵌入的等价工具，本节其余部分介绍它的用法。

### 3.2 关键约束：whole 与 split 共享同一个 PCA basis

`generate_berts.py` 的核心做法：

- 批量推理 + 用 `attention_mask` 做掩码均值池化（mask mean pooling）得到逐 contig 向量；
- **whole 与 split 共享同一个 PCA basis**：在 whole 上 `fit` PCA，再用同一组件对 split 做 `transform`。这样两份嵌入位于同一坐标系，下游融合才有意义。

因此必须**同时**提供 whole fasta 与 split fasta，两者一次调用里一起处理。

### 3.3 对齐要求（务必满足）

| 要求 | 说明 |
| --- | --- |
| 输出文件名 | 必须是 `dnabert_embedding.npy`（whole）与 `dnabert_split_embedding.npy`（split） |
| 输出位置 | 放在与 `data.csv` **同一目录** |
| 行序对齐 | whole fasta 的行序须与 `data.csv` 一致；split fasta 的行序须与 `data_split.csv` 一致 |
| split 命名 | split fasta 中的序列名须形如 `h_1` / `h_2`（与 `generate_kmer.py` 切分一致） |

> 加载时 `load_multimodal_embeddings` 会**逐行校验** contig 名是否与 `data.csv` / `data_split.csv` 对齐，行序不一致会直接报错。生成 split fasta 时请确保切分逻辑与 `generate_sequence_features_*` 产生 `data_split.csv` 时一致。

### 3.4 推荐命令

```bash
python SemiBin/generate_berts.py -md /path/DNABERT-S \
    -fd whole.fasta \
    -nd output/dnabert_contig_names.txt \
    -dd output/dnabert_embedding.npy \
    -sfd split.fasta \
    -snd output/dnabert_split_contig_names.txt \
    -sdd output/dnabert_split_embedding.npy
```

参数含义：

| 参数 | 含义 |
| --- | --- |
| `-md` | DNABERT-S 模型目录 |
| `-fd` / `-nd` / `-dd` | whole 的输入 fasta / 输出 contig 名列表 / 输出嵌入 `.npy` |
| `-sfd` / `-snd` / `-sdd` | split 的输入 fasta / 输出 contig 名列表 / 输出嵌入 `.npy` |

生成完成后，`output/` 目录下应同时存在 `data.csv`、`data_split.csv`、`dnabert_embedding.npy`、`dnabert_split_embedding.npy`，即可进入训练 / 分箱阶段。

### 3.5 相关命令行参数

在 `single_easy_bin` / `multi_easy_bin` 中控制 DNABERT 嵌入：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--dnabert-model PATH` | 内置 `SemiBin/DNABERT-S` 目录 | DNABERT-S 模型路径 |
| `--dnabert-python PATH` | `$SEMIBIN_DNABERT_PYTHON` 或当前解释器 | 运行 DNABERT 推理用的 Python |

全局社区检测 / 聚类相关参数（`single_easy_bin` / `multi_easy_bin` / `bin`）：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--cluster-algorithm {leiden,infomap}` | `leiden` | 全局社区检测算法；`infomap` 仅作为可选项保留 |
| `--cluster-resolution FLOAT` | `1.0` | Leiden 模块度分辨率；越大 bin 越多，越小 bin 越少 |

图融合 / 聚类相关参数（`single_easy_bin` / `multi_easy_bin` / `bin`）见 [usage.md](usage.md) 与 [subcommands.md](subcommands.md)。MAGFuse 用 **Leiden** 做全局社区检测（模块度目标）。

---

## 生成 cannot-link 约束（进阶）

> **注意**：除非你清楚自己在做什么，否则通常**不需要**这一步。半监督相关说明见 [semi-supervised.md](semi-supervised.md)。

MAGFuse 默认用 mmseqs2，但你也可以用 [CAT](https://github.com/dutilh/CAT) 产生 contig 分类，再据此生成 cannot-link 对。

```bash
CAT contigs \
        -c contig.fa \
        -d CAT_prepare_20200304/2020-03-04_CAT_database \
        --path_to_prodigal $path_to_prodigal \
        --path_to_diamond $path_to_diamond \
        -t CAT_prepare_20200304/2020-03-04_taxonomy \
        -o CAT_output/CAT \
        --force \
        -f 0.5 \
        --top 11 \
        --I_know_what_Im_doing \
        --index_chunks 1

CAT add_names \
    CAT_output/CAT.contig2classification.txt \
    -o CAT_output/CAT.out \
    -t CAT_prepare_20200304/2020-03-04_taxonomy \
    --force \
    --only_official
```

再用 `script/` 目录下的脚本生成 cannot-link 约束：

```bash
python script/generate_cannot_link.py -i CAT.out -c contig.fa -s sample-name -o output --CAT
```
