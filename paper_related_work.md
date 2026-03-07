# Related Work

宏基因组分箱旨在将组装后的 contig 按潜在来源基因组划分为不同的 bin，是获得 metagenome-assembled genomes 的关键步骤。现有研究大体可以分为三条技术路线：基于序列组成和丰度信息的传统分箱方法、基于深度表示学习的分箱方法，以及近年来兴起的 DNA 预训练语言模型驱动的序列表征方法。本文工作与现有研究的关系，主要体现在这三条技术脉络的交汇处，即在短读长单样本与多样本分箱场景中，将预训练 DNA 序列表示与 contig-level 分箱任务进行结合。

## 1. 基于序列组成与丰度信息的经典分箱方法

早期宏基因组分箱方法主要依赖 contig 的寡核苷酸组成特征和跨样本覆盖度信息。CONCOCT 通过联合建模 tetranucleotide composition 与 coverage，在统一概率框架下完成 contig 聚类，是该方向中具有代表性的工作 [1]。MaxBin 2.0 进一步引入单拷贝标记基因来辅助估计 bin 数量和优化聚类结果，在多样本宏基因组数据分析中得到广泛应用 [2]。MetaBAT 2 则通过更高效的相似度计算和自适应参数策略，在复杂群落中实现了较好的分箱精度与运行效率平衡 [3]。这类方法构成了宏基因组分箱的基础范式，其优势在于流程清晰、实现成熟，但由于主要依赖手工设计特征和启发式聚类策略，在高复杂度样本、菌株近缘性较强或 coverage 信号较弱的场景中，往往难以充分刻画 contig 之间更高层次的序列关系。

## 2. 基于深度表示学习的宏基因组分箱

随着表示学习方法的发展，宏基因组分箱逐步从“基于原始特征直接聚类”转向“先学习 contig embedding，再进行聚类或重聚类”。SolidBin 将 must-link 与 cannot-link 约束引入 normalized cut 过程，表明外部约束信息能够有效改善 contig 分箱结果 [4]。随后，SemiBin 利用孪生神经网络学习 contig 表征，并通过人工切分 contig 生成 must-link、通过参考数据库注释生成 cannot-link，将度量学习正式引入分箱任务 [5]。这一类方法的重要贡献在于，它们不再将序列组成特征和丰度特征视为固定输入，而是通过神经网络学习更具判别性的嵌入空间。

在深度生成式和对比式学习方向，VAMB 使用变分自编码器联合建模 tetranucleotide frequency 与 abundance 特征，在大规模宏基因组数据上表现出优于传统方法的基因组恢复能力 [6]。进一步地，SemiBin2 将训练方式扩展为 self-supervised contrastive learning，以降低对外部参考注释的依赖，并在短读长和长读长数据中取得了较强性能 [7]。COMEBin 则从多视图表示学习角度出发，通过构建不同 coverage/composition 视图并进行对比优化，体现了多视图特征融合在分箱任务中的潜力 [8]。总体来看，现有深度分箱方法已经证明，表示学习能够显著缓解传统 k-mer 与 abundance 特征表达能力有限的问题；然而，这些方法的特征来源仍然主要集中于局部组成统计、覆盖度模式以及由 contig 切分构造出的训练约束，对于原始 DNA 序列中蕴含的上下文语义信息挖掘仍然相对有限。

## 3. DNA 预训练语言模型与基因组序列表征

近年来，受自然语言处理领域预训练模型成功经验的启发，研究者开始尝试将 Transformer 架构引入 DNA 序列建模。DNABERT 首次将 BERT 框架系统应用于基因组序列分析任务，通过将 k-mer 视为基本词元学习上下文相关表示，在启动子识别、转录因子结合位点预测等多种任务上取得了优于传统方法的结果 [9]。随后，DNABERT-2 进一步提出面向多物种场景的高效 foundation model，增强了模型在更广泛基因组任务上的泛化能力 [10]。与此同时，Nucleotide Transformer 等大规模基础模型也表明，预训练 DNA 表征可以有效捕获超越局部 k-mer 统计的长程依赖和高层次生物学模式 [11]。

与传统序列组成特征相比，DNA 语言模型的优势在于其表示并非简单的频次统计，而是通过大规模预训练获得的上下文相关序列语义。已有研究表明，这类表征在功能注释、调控元件识别和跨物种迁移任务中具有良好表现 [9-11]。然而，相比于序列分类或功能预测等标准下游任务，预训练 DNA 表征在宏基因组 contig 分箱中的应用仍然相对不足。特别是在短读长单样本与多样本分箱场景中，如何将预训练序列 embedding 与 contig 的 coverage 信息、已有嵌入学习机制以及后续重聚类过程有效结合，仍然缺乏系统研究。

## 4. 现有工作的局限与本文工作的定位

综合现有文献可以看出，经典分箱方法已经较为充分地利用了序列组成和丰度模式 [1-3]，深度分箱方法则进一步强化了 contig embedding 的学习能力 [4-8]。然而，两类方法都在一定程度上受限于其输入特征设计：前者更多依赖浅层统计量，后者虽然引入了深度模型，但其主要信息来源仍多建立在 k-mer、abundance 以及由 contig 切分得到的样本对之上。相比之下，DNA 预训练语言模型直接作用于原始核酸序列，能够提供不同于传统 compositional feature 的上下文表示 [9-11]。这一差异意味着，若能够将预训练序列 embedding 引入宏基因组分箱任务，则有可能为 contig 相似性建模提供新的信息来源，尤其是在 coverage 信号不足或序列组成边界不清晰的短读长样本中。

基于此，本文工作定位于预训练 DNA 序列表征与宏基因组分箱方法的结合。与仅依赖传统 k-mer/abundance 特征或常规深度嵌入的方法不同，本文在短读长分箱场景下引入 DNABERT 提取 contig 级序列表示，并将其用于单样本与多样本工作模式中的特征增强与重聚类过程。该思路并非简单以预训练序列特征替代已有分箱特征，而是强调将预训练 DNA 语言模型提供的上下文语义信息，与宏基因组分箱任务中已有的丰度差异和嵌入学习机制进行融合，从而探索一种兼顾序列语义与生态统计特征的分箱路径。

从相关工作角度看，本文与已有方法的差异主要体现在两个方面。第一，现有深度分箱方法大多关注如何围绕组成特征、丰度特征和对比式约束构造更优的 contig embedding [5-8]，而本文进一步引入了来源于 DNA foundation model 的序列表示，以拓展 contig 特征空间。第二，现有 DNA 语言模型工作主要集中于标准基因组学任务 [9-11]，本文则将其引入短读长宏基因组单样本与多样本分箱场景，关注其在 bin refinement 和复杂样本 contig 区分中的潜在作用。因此，本文可以被视为连接“宏基因组深度分箱”与“DNA 预训练序列表征”两条研究主线的工作，其核心意义在于验证预训练序列语义信息能否为宏基因组分箱提供超越传统 compositional feature 的补充判别能力。

## References

[1] Alneberg, J., Bjarnason, B. S., de Bruijn, I., Schirmer, M., Quick, J., Ijaz, U. Z., Lahti, L., Loman, N. J., Andersson, A. F., and Quince, C. Binning metagenomic contigs by coverage and composition. *Nature Methods*, 11(11):1144-1146, 2014. DOI: https://doi.org/10.1038/nmeth.3103

[2] Wu, Y.-W., Simmons, B. A., and Singer, S. W. MaxBin 2.0: an automated binning algorithm to recover genomes from multiple metagenomic datasets. *Bioinformatics*, 32(4):605-607, 2016. DOI: https://doi.org/10.1093/bioinformatics/btv638

[3] Kang, D. D., Li, F., Kirton, E., Thomas, A., Egan, R., An, H., and Wang, Z. MetaBAT 2: an adaptive binning algorithm for robust and efficient genome reconstruction from metagenome assemblies. *PeerJ*, 7:e7359, 2019. DOI: https://doi.org/10.7717/peerj.7359

[4] Wang, Z., Wang, Z., Lu, Y. Y., Sun, F., and Zhu, S. SolidBin: improving metagenome binning with semi-supervised normalized cut. *Bioinformatics*, 35(21):4229-4238, 2019. DOI: https://doi.org/10.1093/bioinformatics/btz253

[5] Pan, S., Zhu, C., Zhao, X.-M., and Coelho, L. P. A deep siamese neural network improves metagenome-assembled genomes in microbiome datasets across different environments. *Nature Communications*, 13:2326, 2022. DOI: https://doi.org/10.1038/s41467-022-29843-y

[6] Nissen, J. N., Johansen, J., Allesoe, R. L., Armenteros, J. J. A., Groenbech, C. H., Nielsen, H. B., Petersen, T. N., Winther, O., and Rasmussen, S. Improved metagenome binning and assembly using deep variational autoencoders. *Nature Biotechnology*, 39:555-560, 2021. DOI: https://doi.org/10.1038/s41587-020-00777-4

[7] Pan, S., Zhao, X.-M., and Coelho, L. P. SemiBin2: self-supervised contrastive learning leads to better MAGs for short- and long-read sequencing. *Bioinformatics*, 39(Supplement_1):i21-i29, 2023. DOI: https://doi.org/10.1093/bioinformatics/btad209

[8] Wang, Z., You, R., Han, H., Liu, W., Sun, F., and Zhu, S. Effective binning of metagenomic contigs using contrastive multi-view representation learning. *Nature Communications*, 15:585, 2024. DOI: https://doi.org/10.1038/s41467-023-44290-z

[9] Ji, Y., Zhou, Z., Liu, H., and Davuluri, R. V. DNABERT: pre-trained Bidirectional Encoder Representations from Transformers model for DNA-language in genome. *Bioinformatics*, 37(15):2112-2120, 2021. DOI: https://doi.org/10.1093/bioinformatics/btab083

[10] Zhou, Z., Ji, Y., Li, W., Dutta, P., Davuluri, R. V., and Liu, H. DNABERT-2: Efficient foundation model and benchmark for multi-species genome. *arXiv*, arXiv:2306.15006, 2023. URL: https://arxiv.org/abs/2306.15006

[11] Dalla-Torre, H., Gonzalez, L., Mendoza-Revilla, J., Carranza, N. L., Grzywaczewski, A. H., Oteri, F., Dallago, C., Trop, E., Sirelkhatim, H., Richard, G., et al. The Nucleotide Transformer: building and evaluating robust foundation models for human genomics. *Nature Methods*, 21:314-321, 2024. DOI: https://doi.org/10.1038/s41592-024-02523-z
