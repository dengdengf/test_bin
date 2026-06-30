# 用法示例

本页给出 MAGFuse 的主要用法示例。MAGFuse 是一个独立的多模态宏基因组分箱工具，命令行入口为 `MAGFuse`。

> 关于如何准备输入（contigs 组装结果 + BAM/CRAM 或 strobealign-aemb 丰度），见 [generate](generate.md)。
> 输出文件结构见 [output](output.md)；全部子命令与参数见 [subcommands](subcommands.md)；安装见 [install](install.md)。
> MAGFuse 的核心方法与特性见 [methods](methods.md) 与 [whatsnew](whatsnew.md)。

---

## 分箱模式概览

| 模式 | 命令 | 是否可用预训练模型 | 适用场景 |
| --- | --- | --- | --- |
| 单样本分箱 | `single_easy_bin` | 是（`--environment`） | 每个样本独立组装、独立分箱 |
| 共组装分箱 | `single_easy_bin`（多个 BAM） | 否 | 样本相似、可共组装（如同一生境的时间序列） |
| 多样本分箱 | `multi_easy_bin` | 否 | 多样本独立组装，但联合使用共丰度信息 |

- **单样本分箱**：每个样本独立组装并独立分箱，可并行、避免跨样本嵌合，但不使用跨样本共丰度信息。配合预训练模型可在几分钟内出结果。
- **共组装分箱**：先把样本池当作单一样本共组装，再分箱。可利用共丰度信息，但可能产生样本间嵌合 contig，且不保留样本特异性差异。
- **多样本分箱**：各样本独立组装、独立分箱，但**联合使用多样本信息**，既能用共丰度又能保留样本特异性差异，通常在复杂生境（如土壤）中得到最多的 bin，代价是计算量更大且无法使用预训练模型。

---

## 多模态相关特性

MAGFuse 在**短读长**（`short_read`）流程中**默认且始终启用**多模态嵌入，提供以下能力（仅在此处罗列，详见 [methods](methods.md)）：

- **多模态嵌入模型**（组成 / 丰度 / DNABERT 三分支 + 学习式门控融合 + 跨模态对齐损失）：是短读长训练路径的常规组成部分，无需手动开启。
- **多视图相似度图融合聚类**：对 embedding / 组成 / 丰度 / DNABERT 各建 kNN 相似度图，加权融合后用 **Leiden** 做全局社区检测（Leiden 是唯一的全局社区检测算法）。融合权重、核函数可配置，并在多样本（非 combined）下重新引入“共丰度 KL 散度”边权调制。
- **标记基因去污染重聚类**：在被判为污染的 bin 内做带种子的标签传播，仅当单拷贝标记基因冗余度下降时才接受拆分。

相关命令行参数：

| 参数 | 适用子命令 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--knn-kernel {median,local}` | `single_easy_bin` / `multi_easy_bin` / `bin` | `median` | kNN 相似度图核函数 |
| `--fusion-weights EMB COMP ABUND` | 同上 | `0.60 0.25 0.15` | 基础融合权重（不含 DNABERT 分支时） |
| `--fusion-weights-multimodal EMB COMP ABUND DNA` | 同上 | `0.45 0.15 0.15 0.25` | 含 DNABERT 分支时的融合权重 |
| `--no-coabundance-kl` | 同上 | （默认开启 KL） | 关闭共丰度 KL 调制 |
| `--cluster-resolution FLOAT` | 同上 | `1.0` | Leiden 模块度分辨率（越大 bin 越多、越小） |
| `--dnabert-model PATH` | `single_easy_bin` / `multi_easy_bin` | 内置 `SemiBin/DNABERT-S` 目录 | DNABERT-S 权重路径 |
| `--dnabert-python PATH` | 同上 | `$SEMIBIN_DNABERT_PYTHON` 或当前解释器 | 运行 DNABERT 推理的 Python 解释器 |

---

## 单样本分箱

所需输入：`S1.fa`（FASTA 格式 contig）与 `S1.sorted.bam`（短读长比对到 contig 并排序后的 BAM）。

### 1. 使用预训练模型（最快）

如果你的宏基因组属于内置生境之一，使用预训练模型可在几分钟内返回结果（也可使用综合模型 `global`）。

```bash
MAGFuse single_easy_bin \
        --environment human_gut \
        -i S1.fa \
        -b S1.sorted.bam \
        -o output
```

支持的生境（名称即含义，`global` 为通用综合模型）：

`human_gut`、`dog_gut`、`ocean`、`soil`、`cat_gut`、`human_oral`、`mouse_gut`、`pig_gut`、`built_environment`、`wastewater`、`chicken_caecum`、`global`。

### 2. 学习一个新模型（自监督）

不使用内置模型，针对你自己的数据学习一个新模型。会更慢，但可能得到更好的结果。短读长下多模态训练路径默认且始终启用：

```bash
MAGFuse single_easy_bin \
        --self-supervised \
        -i S1.fa \
        -b S1.sorted.bam \
        -o output
```

如果样本数量较多且彼此相似、但不属于任何内置模型，你也可以从其中一部分样本自建模型，见 [training](training.md)。

### 长读长

长读长分箱使用 `--sequencing-type=long_read`（采用 DBSCAN 集成聚类算法）：

```bash
MAGFuse single_easy_bin \
        --environment human_gut \
        --sequencing-type long_read \
        -i S1.fa \
        -b S1.sorted.bam \
        -o output
```

> 注意：多模态嵌入是短读长路径的组成部分；长读长路径采用不同的 DBSCAN 集成算法，不涉及多模态嵌入。

### 调整图融合聚类参数

聚类阶段的图融合行为可通过相关参数调整。例如改用 `local` 核、自定义融合权重、并关闭共丰度 KL 调制：

```bash
MAGFuse single_easy_bin \
        --self-supervised \
        --knn-kernel local \
        --fusion-weights 0.70 0.20 0.10 \
        --no-coabundance-kl \
        -i S1.fa \
        -b S1.sorted.bam \
        -o output
```

含 DNABERT 分支时，使用 `--fusion-weights-multimodal`（四个权重对应 EMB / COMP / ABUND / DNA）：

```bash
MAGFuse single_easy_bin \
        --self-supervised \
        --fusion-weights-multimodal 0.45 0.15 0.15 0.25 \
        -i S1.fa \
        -b S1.sorted.bam \
        -o output
```

全局社区检测用 Leiden（唯一算法），可用 `--cluster-resolution` 调整 bin 粒度：

```bash
MAGFuse single_easy_bin \
        --self-supervised \
        --cluster-resolution 1.2 \
        -i S1.fa \
        -b S1.sorted.bam \
        -o output
```

### 进阶：拆分单样本流程的各步骤

`single_easy_bin` 内部依次执行 `generate_data_single` → `train_self` → `bin_short`/`bin_long`。你可以手动运行各步骤，以便在计算集群上并行加速。

(1) 生成特征（`data.csv` / `data_split.csv`）：

```bash
MAGFuse generate_sequence_features_single -i S1.fa -b S1.sorted.bam -o S1_output
```

(2) 训练模型（如需）。该步骤受益于 GPU，会自动检测，也可用 `--engine` 指定 CPU/GPU：

```bash
MAGFuse train_self \
    --data S1_output/data.csv \
    --data-split S1_output/data_split.csv \
    -o S1_output
```

如使用预训练模型，可跳过此步。

(3) 分箱：

```bash
MAGFuse bin_short \
    -i S1.fa \
    --model S1_output/model.pt \
    --data S1_output/data.csv \
    -o S1_output
```

长读长用 `bin_long`：

```bash
MAGFuse bin_long \
    -i S1.fa \
    --model S1_output/model.pt \
    --data S1_output/data.csv \
    -o S1_output
```

或用内置预训练模型（把 `--model` 换成 `--environment`）：

```bash
MAGFuse bin_short \
    -i S1.fa \
    --environment human_gut \
    --data S1_output/data.csv \
    -o S1_output
```

`bin` / `bin_short` 同样接受 `--knn-kernel`、`--fusion-weights`、`--no-coabundance-kl`、`--cluster-resolution` 等图融合参数。

---

## 共组装分箱

输入：`contig.fa` 与多个 BAM（`S1.sorted.bam`、`S2.sorted.bam`、`S3.sorted.bam` …）。

共组装与单样本流程基本一致，区别在于生成特征时使用多个样本的 BAM。因此**无法使用预训练模型**（模型依赖样本数量）。

```bash
MAGFuse single_easy_bin \
    -i contig.fa \
    -b S1.sorted.bam S2.sorted.bam S3.sorted.bam \
    -o co-assembly_output
```

进阶（拆分各步骤）：

```bash
# (1) 生成特征（仍用 single 模式，因共组装与单样本流程相近）
MAGFuse generate_sequence_features_single \
    -i contig.fa \
    -b S1.sorted.bam S2.sorted.bam S3.sorted.bam \
    -o contig_output

# (2) 训练
MAGFuse train_self \
    --data contig_output/data.csv \
    --data-split contig_output/data_split.csv \
    -o contig_output

# (3) 分箱
MAGFuse bin_short \
    -i contig.fa \
    --model contig_output/model.pt \
    --data contig_output/data.csv \
    -o output
```

---

## 多样本分箱

多样本分箱准备数据更复杂、计算量更大，但在复杂生境中往往能得到更多 bin。

输入：

- 各样本原始 FASTA：`S1.fa`、`S2.fa`、`S3.fa`、`S4.fa`、`S5.fa`（这里假设 5 个样本）；
- 合并后的 FASTA：`concatenated.fa`（由 `concatenate_fasta` 子命令生成）；
- 各样本读长比对到合并 FASTA 后的 BAM：`S1.sorted.bam` … `S5.sorted.bam`。

### 生成 `concatenated.fa`

```bash
MAGFuse concatenate_fasta \
    --input-fasta S1.fa S2.fa S3.fa S4.fa S5.fa \
    --output output
```

会产生 `output/concatenated.fa`。

**格式说明**：每个 contig 被重命名为 `<sample_name>:<original_contig_name>`，`:` 为默认分隔符（可用 `--separator` 修改，但**之后所有用到它的命令都要带上同一分隔符**）。`concatenate_fasta` 会保证样本名唯一且分隔符不会引入歧义。随后请将每个样本分别比对到合并 FASTA，得到各自的 `sorted.bam`。

### Easy 多样本分箱

短读长（多模态训练默认且始终启用）：

```bash
MAGFuse multi_easy_bin \
        -i concatenated.fa \
        -b S1.sorted.bam S2.sorted.bam S3.sorted.bam S4.sorted.bam S5.sorted.bam \
        -o multi_output
```

长读长：

```bash
MAGFuse multi_easy_bin \
        --sequencing-type long_read \
        -i concatenated.fa \
        -b S1.sorted.bam S2.sorted.bam S3.sorted.bam S4.sorted.bam S5.sorted.bam \
        -o multi_output
```

> 多样本（非 combined）模式下，图融合聚类会在边权上重新引入共丰度 KL 散度调制；可用 `--no-coabundance-kl` 关闭。

### 进阶：拆分多样本流程的各步骤

`multi_easy_bin` 内部依次执行 `generate_data_multi` → `train_self`（如需）→ `bin_short`/`bin_long`，各步骤可独立运行并在集群上并行。

(1) 生成 `data.csv` / `data_split.csv`：

```bash
MAGFuse generate_sequence_features_multi \
    -i concatenated.fa \
    -b S1.sorted.bam S2.sorted.bam S3.sorted.bam S4.sorted.bam S5.sorted.bam \
    -o multi_output
```

(2) 训练（逐样本独立，但使用涵盖全部样本数据的输入特征，可并行）：

```bash
for sample in S1 S2 S3 S4 S5 ; do
    MAGFuse train_self \
        --data multi_output/samples/${sample}/data.csv \
        --data-split multi_output/samples/${sample}/data_split.csv \
        --output ${sample}_output
done
```

(3) 分箱（`bin_short` 短读长 / `bin_long` 长读长，逐样本独立）：

```bash
for sample in S1 S2 S3 S4 S5 ; do
    MAGFuse bin_short \
        -i ${sample}.fa \
        --model ${sample}_output/model.pt \
        --data multi_output/samples/${sample}/data.csv \
        -o output
done
```

---

## 用部分样本预训练模型

你可以从数据集中的一部分样本预训练一个模型，作为折中：比逐样本训练更快，又比来自其它数据集的预训练模型效果更好。

假设已有 `S1.fa`、`S1/data.csv`、`S1/data_split.csv`、`S1/cannot.txt`（`S2`、`S3` 同理），可用 3 个样本训练：

```bash
MAGFuse train \
    -i S1.fa S2.fa S3.fa \
    --data S1/data.csv S2/data.csv S3/data.csv \
    --data-split S1/data_split.csv S2/data_split.csv S3/data_split.csv \
    -c S1/cannot.txt S2/cannot.txt S3/cannot.txt \
    --mode several \
    -o S1_output
```

更多内容见 [training](training.md)。

---

## 完整的 DNABERT 多模态流程

DNABERT 分支需要 DNABERT-S 嵌入。`single_easy_bin` / `multi_easy_bin` 会**自动提取**这些嵌入（含 split 半段，共享 PCA basis），无需手动操作；`generate_berts.py` 则是离线/单独生成嵌入的等价工具。整体流程为：**生成序列特征 →（自动提取 DNABERT 嵌入，或用 `generate_berts.py` 离线生成 whole 与 split 两份嵌入）→ 训练 + 分箱**。

### 前置：获取 DNABERT-S 权重

DNABERT-S 预训练权重较大，已被 gitignore，**不随仓库分发**。请单独获取后放到 `SemiBin/DNABERT-S/`，或在命令中用 `--dnabert-model` / `-md` 指定路径。
来源：<https://github.com/MAGICS-LAB/DNABERT_S>

### 步骤 1：生成序列特征

先生成 `data.csv` 与 `data_split.csv`（DNABERT 嵌入的行序必须与它们一致）：

```bash
MAGFuse generate_sequence_features_single -i S1.fa -b S1.sorted.bam -o output
```

> split 的 contig 命名形如 `h_1` / `h_2`（见 `generate_kmer.py`，由父 contig 自动切半）。`single_easy_bin` / `multi_easy_bin` 的内置自动提取会自动处理 whole 与 split 半段（`h_1`/`h_2`）并共享同一 PCA basis，无需手动。若希望离线/单独生成嵌入，可改用 `generate_berts.py`（等价工具）。

### 步骤 2（可选）：用 `generate_berts.py` 离线生成 whole + split 嵌入

如选择离线生成，需要 `whole.fasta`（整条 contig）与 `split.fasta`（split 半段）两份输入。whole 与 split **共享同一个 PCA basis**（在 whole 上 fit，对 split 用 transform）：

```bash
python SemiBin/generate_berts.py -md /path/DNABERT-S \
  -fd whole.fasta -nd output/dnabert_contig_names.txt -dd output/dnabert_embedding.npy \
  -sfd split.fasta -snd output/dnabert_split_contig_names.txt -sdd output/dnabert_split_embedding.npy
```

要点：

- 输出文件名**必须**是 `dnabert_embedding.npy` 与 `dnabert_split_embedding.npy`，并放在 `data.csv` 同一目录下。
- 两份 FASTA 的行序必须与 `data.csv` / `data_split.csv` 一致（`load_multimodal_embeddings` 会逐行校验）。
- 推理采用批量 + attention_mask 掩码均值池化。

### 步骤 3：训练 + 分箱

正常运行短读长多模态流程即可（多模态默认且始终启用，会自动提取或加载同目录下的 DNABERT 嵌入）：

```bash
MAGFuse single_easy_bin \
        --self-supervised \
        --dnabert-model /path/DNABERT-S \
        --fusion-weights-multimodal 0.45 0.15 0.15 0.25 \
        -i S1.fa \
        -b S1.sorted.bam \
        -o output
```

如需指定运行 DNABERT 推理的 Python 解释器，可用 `--dnabert-python`（或设置环境变量 `SEMIBIN_DNABERT_PYTHON`）。

---

## 半监督模式

> ⚠️ 注意：半监督模式已不再推荐，通常无需使用。

详见 [semi-supervised](semi-supervised.md)。

## 配合 strobealign-aemb

见专门的 [aemb](aemb.md) 页面。
