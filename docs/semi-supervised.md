# 半监督分箱模式

半监督（semi-supervised）模式是 MAGFuse 提供的一种训练方式，依赖比对得到的「cannot-link」约束来辅助训练。

> 说明：半监督模式通常比自监督（self-supervised）模式更慢、占用更多内存，因此在多数情况下推荐使用自监督模式（`train_self`）。这里提供半监督模式主要是为了向后兼容，以及在某些有可用 GTDB 注释的场景下方便复用已有结果。
>
> MAGFuse 的多模态嵌入、图融合聚类、标记基因去污染重聚类等设计，与「半监督 vs 自监督」这一训练方式的选择是**正交的、可叠加的**：你既可以在半监督模式下，也可以在自监督模式下使用多模态 / 图融合相关参数。多模态相关说明见 [usage.md](usage.md) 与 [training.md](training.md)。

## 依赖前提

半监督模式需要额外的外部依赖与参考数据：

- **mmseqs2**：用于把 contigs 比对到 GTDB 分类，从而生成 cannot-link 约束。
- **GTDB**：分类学参考数据库。可用 `MAGFuse download_GTDB` 下载。

其余依赖（bedtools、hmmer、samtools 等）与常规流程相同，见 [install.md](install.md)。

## Easy 模式（一键运行）

最简单的方式是使用 `single_easy_bin` 子命令，它会自动完成全部步骤。若要用半监督模式训练，加上 `--semi-supervised` 标志即可：

```bash
MAGFuse single_easy_bin \
        --semi-supervised \
        -i S1.fa \
        -b S1.sorted.bam \
        -o output
```

> 多模态 / 图融合参数（如 `--knn-kernel`、`--fusion-weights`、`--fusion-weights-multimodal`、`--no-coabundance-kl`、`--dnabert-model` 等）在此处同样可用，与 `--semi-supervised` 互不冲突。各参数含义见 [subcommands.md](subcommands.md) 与 [training.md](training.md)。

## 进阶：单样本 / 多样本分步流程

整体步骤与自监督模式基本一致，区别在于多了一步生成 cannot-link 文件，并用 `train_semi` 替换 `train_self`：

| 步骤 | 自监督模式 | 半监督模式 |
| --- | --- | --- |
| 1. 生成特征 | `generate_sequence_features_single` / `generate_sequence_features_multi` | 同左 |
| 2. 生成约束 | （无） | `generate_cannot_links`（额外步骤，见下文） |
| 3. 训练 | `train_self` | `train_semi` |
| 4. 分箱 | `bin`（短读长）/ `bin_long`（长读长） | 同左 |

### 生成 cannot-link 文件（单样本 / 共组装模式）

```bash
MAGFuse generate_cannot_links -i S1.fa -o S1_output
```

注意：这一步会调用 **mmseqs2**，比较耗时。

如果你的流程中已经用 mmseqs2 针对 GTDB 做过分类注释，可以通过 `--taxonomy-annotation-table` 参数把已有结果传入，从而跳过这一步的重复计算。

### 生成 cannot-link 文件（多样本模式）

需要对每个输入 FASTA 文件**分别**调用 `generate_cannot_links`：

```bash
for sample in S1 S2 S3 S4 S5; do
    MAGFuse generate_cannot_links -i ${sample}.fa -o ${sample}_output
done
```

上面用了 bash 循环，等价于逐条运行：

```bash
MAGFuse generate_cannot_links -i S1.fa -o S1_output
MAGFuse generate_cannot_links -i S2.fa -o S2_output
MAGFuse generate_cannot_links -i S3.fa -o S3_output
MAGFuse generate_cannot_links -i S4.fa -o S4_output
MAGFuse generate_cannot_links -i S5.fa -o S5_output
```

同样地，如果已经用 mmseqs2 对 contigs 做过 GTDB 注释，可用 `--taxonomy-annotation-table` 绕过大部分计算。

## 相关页面

- [usage.md](usage.md)：整体使用流程概览
- [training.md](training.md)：训练模式与多模态训练说明
- [subcommands.md](subcommands.md)：各子命令与参数详解
- [install.md](install.md)：安装与依赖
