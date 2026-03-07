# Materials and Methods

## 方法概述

本文提出了一种面向短读长宏基因组分箱任务的增强型方法框架，同时支持单样本和多样本两种工作模式。该方法以组装后的 contig 及其来源于 reads 比对的丰度信息为输入，整体包括特征构建、contig 表征学习、初始图聚类以及基于 DNABERT 的 bin 精炼四个主要阶段。与仅依赖序列组成和丰度统计信息的方法不同，本文在分箱流程中引入预训练 DNA 语言模型分支，以刻画 contig 层面的上下文序列语义信息。由此，方法能够利用传统 compositional-abundance 特征完成全局 bin 构建，同时借助预训练序列表征提高污染 bin 的细粒度分离能力。

对于输入 contig 集合，首先计算每条 contig 的 tetranucleotide frequency 和 abundance 特征；随后对长 contig 进行切分，以构造自监督表征学习所需的正样本对；与此同时，利用 DNABERT 对完整 contig 序列进行编码，生成序列级 embedding。基于表征学习网络得到的 contig embedding 构建近邻图并执行初始分箱；之后，利用单拷贝标记基因识别潜在污染 bin，并在由 DNABERT 表征与归一化丰度共同构成的特征空间中对其进行精炼。该流程同时适用于短读长单样本分箱和短读长多样本分箱。

## 输入数据与特征生成

### Contig 与丰度输入

本方法的输入包括组装后的 contig FASTA 文件，以及排序后的 BAM/CRAM 文件或预先计算得到的 abundance 文件。在单样本模式下，输入为一个 contig 集合及其对应的丰度信息；在多样本模式下，不同样本的 contig 首先被拼接为一个统一的 FASTA 文件，并在每条 contig 的名称中保留样本标识，以便后续按样本拆分和独立处理。

### 四核苷酸频率特征

对每条 contig 计算 canonical 4-mer 频率作为序列组成特征。考虑反向互补等价后，可得到 136 维特征表示。为降低零计数造成的不稳定性，在归一化前对每一维加入一个很小的伪计数，再进行按行归一化。该四核苷酸频率向量构成了训练与聚类过程中最基本的 composition 特征。

### 自监督正样本对的构建

为了在不依赖外部标签的条件下生成可靠的正样本对，对长度超过预设 must-link 阈值的 contig 按中点切分为两个片段。由同一 contig 切分得到的两个子片段被视为来自同一潜在基因组，因此构成一个正样本对。针对这些切分后的子 contig，同样分别生成对应的 composition 特征和 abundance 特征，并与原始 contig 特征分开保存，用于后续训练。

### 丰度特征生成

当输入为 BAM 或 CRAM 文件时，首先根据 reads 在 contig 上的覆盖深度计算丰度特征。对于短读长单样本分箱，分别计算每条 contig 的平均覆盖度和覆盖度方差；对于组合 abundance 场景，尤其是在多样本条件下，则将 abundance 表示为跨样本 coverage 向量。对于切分后的 contig，同样生成对应的 coverage 特征，以保证原始 contig 和切分 contig 的特征矩阵保持一致。

当输入为 abundance 文件时，直接读取丰度矩阵，并根据 contig 标识及切分后的子片段索引进行重组。在所有情况下，原始 contig 的 composition 特征和 abundance 特征最终合并为 `data.csv`，切分后的子 contig 特征则合并为 `data_split.csv`。

## Contig 表征学习

### 神经网络结构

contig 表征学习模块采用共享参数的双塔编码器结构。设输入特征向量为 $x$，编码器由三层全连接网络组成，维度依次为 `input -> 512 -> 512 -> 100`。前两层之后分别接 Batch Normalization、LeakyReLU 激活函数和 Dropout 层，其中 dropout rate 设为 0.2。最终输出为 100 维 contig embedding，用于后续的图构建和聚类。

在单样本训练中，编码器输入为 136 维四核苷酸频率特征；在组合 abundance 场景中，输入为 composition 特征与 abundance 特征拼接后的联合向量。尽管不同模式下输入维度不同，但编码器的主体结构保持一致。

### 自监督样本对构建

训练阶段使用两类样本对。正样本对来自同一条 contig 切分得到的两个片段；负样本对则通过在 contig 集合中随机抽取两条不同 contig 构造。记正样本对数量为 $N_{+}$，则负样本对数量设为 $\min(500N_{+}, 4{,}000{,}000)$`，以在提供充分对比监督的同时控制内存开销。

### 训练目标

编码器采用基于欧氏距离的对比损失进行训练。对于一个样本对，其 embedding 分别记为 $(z_i, z_j)$，二元标签 $y \in \{0,1\}$ 中，$y=1$ 表示正样本对，$y=0$ 表示负样本对，则损失函数定义为

\[
\mathcal{L} = y \cdot ||z_i-z_j||_2^2 + (1-y)\cdot \max(0, 1-||z_i-z_j||_2)^2.
\]

该目标函数使得来自同一 contig 的两个片段在嵌入空间中更接近，而随机配对的 contig 至少保持单位 margin 的距离。模型训练使用 Adam 优化器，初始学习率为 $10^{-3}$，并使用 StepLR 在每个 epoch 后按 0.9 的系数衰减学习率。除特别说明外，batch size 设为 2048，训练轮数设为 15。

## 基于 DNABERT 的序列表征

### 序列 embedding 提取

为弥补 composition 和 abundance 特征在上下文序列信息建模上的不足，本文引入预训练 DNA 语言模型 DNABERT 对 contig 序列进行编码。对于每条 contig，首先通过 DNABERT tokenizer 将其转化为 token 序列，并截断至最大长度 5000。随后将其输入预训练模型，提取最后一层 hidden states，并在序列维度上进行平均池化，从而得到 contig 的序列级 embedding。

### 降维与顺序对齐

由于 DNABERT 原始输出维度高于后续精炼模块所需维度，因此采用主成分分析将其降至最多 128 维。当 contig 数量少于 128 时，降维维数自动设为样本数。为保证不同特征分支间的一一对应关系，DNABERT embedding 按照 `data.csv` 中 contig 的顺序输出，并且只有在 contig 名称和行数完全一致时，才会被加载进入后续流程。

### DNABERT 分支的作用

DNABERT 分支并非用于替代传统分箱特征，而是作为一种补充性的序列表征来源，用于增强 learned contig embedding 与 abundance 信号。在本文框架中，DNABERT 特征主要作用于两个阶段：其一，用于约束混合模式下初始图的构建；其二，用于驱动潜在污染 bin 的重聚类过程。

## 初始图构建与聚类

### 未使用 DNABERT 时的图构建

当未启用 DNABERT 分支时，首先基于 learned contig embedding 构建 embedding-based 的 k-nearest-neighbor 图。为提升图结构的稳健性，再从原始输入特征空间构建第二个近邻图，并对两图取交集，仅保留在两个空间中同时保持邻近关系的 contig 对。这一操作可以减少由单一特征空间噪声引入的伪连接。

### 使用 DNABERT 时的混合图构建

当 DNABERT embedding 可用时，分别基于 learned contig embedding 和 DNABERT sequence embedding 构建两个独立的近邻图，再对两图取交集，仅保留同时受到任务相关嵌入空间和预训练序列表征空间支持的 contig 对。该混合图设计的目的在于，使保留下来的邻接关系同时反映 abundance-aware 的 contig 表征以及上下文序列相似性。

### 边过滤与丰度校正

图构建完成后，将边权从距离转换为相似度，并通过自适应阈值去除弱边。对于非组合 abundance 场景，还进一步根据 coverage 统计构建 abundance-consistency 矩阵，并与图边权逐元素相乘。该校正步骤可以降低那些在 abundance 模式上不一致、但在 embedding 空间中距离较近的 contig 之间的连接强度。

### 基于 Infomap 的初始聚类

在得到加权 contig 图之后，采用 Infomap 对图进行社区发现，生成初始 bin 划分结果。每条 contig 被赋予一个 bin 标签，并据此导出 bin FASTA 文件。这些初始聚类结果构成后续污染识别与 bin 精炼的基础。

## 基于标记基因的 bin 精炼

### 潜在污染 bin 的识别

初始聚类完成后，将属于同一 bin 的 contig 重新拼接，并利用单拷贝标记基因对各 bin 进行筛查。若某个 bin 对应到多个 marker-derived seed 集合，则将其视为潜在污染 bin。只有这类存在污染迹象的 bin 才会进入精炼阶段；不存在污染证据的 bin 则保持不变。

### 精炼特征空间

对于每个候选污染 bin，提取 bin 内各 contig 的 DNABERT embedding，并将其与归一化后的 abundance 向量进行拼接，构成精炼特征空间：

\[
F = [E_d \,\|\, A],
\]

其中 $E_d$ 表示 DNABERT embedding，$A$ 表示归一化后的 abundance 特征。与初始聚类阶段不同，精炼阶段更强调序列语义信息，因为此时的目标不再是全局组织所有 contig，而是对已经形成的混合 bin 进行更细粒度的拆分。

### 基于种子锚点的硬划分

将 marker-derived seeds 视为精炼特征空间中的锚点。对污染 bin 内每条 contig，计算其到所有种子锚点的欧氏距离，并将其分配给最近的种子簇。为避免保留位于簇边界附近的模糊 contig，仅保留最近锚点距离位于前 60% 百分位以内的 contig 参与最终重分配。该硬过滤步骤能够移除对任一子 bin 支持较弱的 contig。

### 回滚策略

为避免过度分裂，本文进一步引入回滚机制。若精炼后的子 bin 未能有效分离 marker-derived seeds，或者所得子 bin 不满足最小长度和最小 contig 数等约束，则拒绝本次精炼结果并恢复为原始 bin 标签。只有当至少两个精炼后的子 bin 同时满足种子分离和规模约束时，新的划分结果才会被接受。

## 单样本与多样本分箱流程

### 单样本流程

在单样本模式下，完整流程包括：从 contig 集合及其对应 reads 比对结果中构建 composition 与 abundance 特征，生成 split-contig 训练对，提取 DNABERT embedding，训练 contig 表征学习网络，执行基于 Infomap 的初始聚类，以及对污染 bin 进行 DNABERT-guided 的精炼。该流程适用于短读长样本，尤其适合需要依赖序列特征区分近缘基因组的场景。

### 多样本流程

在多样本模式下，首先依据 contig 名称中的样本标识，将拼接 FASTA 拆分为多个样本级 FASTA 文件。随后，对每个样本分别构建 `data.csv` 和 `data_split.csv`，独立提取 DNABERT embedding，并分别执行样本级表征学习、初始聚类和 bin 精炼。最终，再将所有样本的分箱结果合并到统一输出目录中。

尽管最终分箱结果按样本独立生成，但在 abundance 特征构建阶段仍可利用跨样本信息。因此，该设计既保留了 abundance 共变模式所带来的区分能力，又避免了在单一全局图中直接聚类全部样本 contig 所带来的复杂性。

## 实现细节

本方法主体基于 Python 实现。表征学习模块采用 PyTorch，预训练 DNA 语言模型分支基于 Hugging Face Transformers 接口实现。四核苷酸频率由内部特征构建模块生成，coverage 统计依赖 `bedtools genomecov`，主成分分析由 scikit-learn 实现，图聚类采用 `igraph` 中的 Infomap 实现。

除特别说明外，表征学习模型的训练参数设为：batch size 2048、学习率 $10^{-3}$、训练轮数 15；DNABERT embedding 通过最后一层 hidden states 的 mean pooling 获得，并在必要时降至 128 维；在精炼阶段，丰度向量先归一化后再与 DNABERT embedding 拼接，同时使用 60% 百分位距离阈值过滤弱分配 contig。最终输出包括 bin FASTA 文件、bin 汇总表和 contig-to-bin 映射表，以支持后续完整性和污染度评估。
