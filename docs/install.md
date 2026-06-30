# 安装指南

本页介绍如何安装 **Multimodal SemiBin** —— 在 SemiBin2 (v2.2.0) 基础上扩展的多模态宏基因组分箱工具（MIT 许可，派生自 [SemiBin/SemiBin2](https://github.com/BigDataBiology/SemiBin)，© BigDataBiology）。

本项目通过源码安装，命令行入口仍为 `SemiBin2`（向后兼容 `SemiBin`）。

> 关于本项目相对上游 SemiBin2 的改动（多模态嵌入、多视图图融合聚类、标记基因去污染重聚类、DNABERT 特征提取），请参见 [whatsnew.md](whatsnew.md) 与 [usage.md](usage.md)。

## Python 版本要求

支持 Python **3.7 – 3.13**。

## 1. 外部依赖

SemiBin 运行需要以下外部命令行工具，推荐用 conda 从 bioconda 安装：

```bash
conda create -n SemiBin python
conda activate SemiBin
conda install -c conda-forge -c bioconda bedtools hmmer samtools
```

| 工具 | 用途 | 是否必需 |
| --- | --- | --- |
| [bedtools](http://bedtools.readthedocs.org/) | 丰度/覆盖度计算 | 必需 |
| [hmmer](http://hmmer.org/) | 单拷贝标记基因检测 | 必需 |
| [samtools](http://www.htslib.org/) | BAM/CRAM 处理 | 必需 |
| [mmseqs2](https://github.com/soedinglab/MMseqs2) | 半监督模式生成 cannot-link 约束 | 可选 |
| [prodigal](https://github.com/hyattpd/Prodigal) | 基因预测 | 可选 |

可选工具按需安装：

```bash
conda install -c conda-forge -c bioconda mmseqs2 prodigal
```

## 2. 从本仓库源码安装

本项目不通过 conda/PyPI 分发，需从源码安装：

```bash
git clone https://github.com/dengdengf/test_bin
cd test_bin
pip install .
```

安装完成后即可使用 `SemiBin2`（及别名 `SemiBin`）命令。验证：

```bash
SemiBin2 --version
SemiBin2 --help
```

> 如需 GPU 加速，请按 [PyTorch 官网](https://pytorch.org/get-started/locally/) 的说明先安装带 CUDA 支持的 PyTorch，再执行上面的 `pip install .`。

## 3. DNABERT 额外依赖（可选）

只有在使用多模态/DNABERT 特征时才需要这一步。标准自监督流程（或使用 `--disable-multimodal-training`）不依赖这些组件。

### 3.1 Python 依赖

```bash
pip install "transformers>=4.30" biopython tqdm
```

### 3.2 获取并放置 DNABERT-S 模型

DNABERT-S 预训练权重体积较大，已被 `.gitignore`，**不随本仓库分发**，需单独获取：

- 来源：[https://github.com/MAGICS-LAB/DNABERT_S](https://github.com/MAGICS-LAB/DNABERT_S)
- 放置位置：默认放到仓库内的 `SemiBin/DNABERT-S/` 目录；
- 或放在任意位置，运行时用 `--dnabert-model PATH` 指定。

相关命令行参数（`single_easy_bin` / `multi_easy_bin`）：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--dnabert-model PATH` | 内置 `SemiBin/DNABERT-S` 目录 | DNABERT-S 模型路径 |
| `--dnabert-python PATH` | `$SEMIBIN_DNABERT_PYTHON` 或当前解释器 | 运行 DNABERT 推理的 Python 解释器 |
| `--disable-multimodal-training` | — | 关闭多模态，回退标准自监督训练 |

### 3.3 关于 DNABERT 嵌入文件

DNABERT 嵌入需要 **whole + split 两份**，二者共享同一个 PCA basis（在 whole 上 fit、对 split 用 transform）。推荐用 `SemiBin/generate_berts.py` 显式生成：

```bash
python SemiBin/generate_berts.py -md /path/DNABERT-S \
  -fd whole.fasta -nd output/dnabert_contig_names.txt -dd output/dnabert_embedding.npy \
  -sfd split.fasta -snd output/dnabert_split_contig_names.txt -sdd output/dnabert_split_embedding.npy
```

注意事项：

- 输出文件名必须是 `dnabert_embedding.npy` / `dnabert_split_embedding.npy`，放在 `data.csv` 同目录；
- fasta 行序须与 `data.csv` / `data_split.csv` 一致（`load_multimodal_embeddings` 会逐行校验）；
- split 半段名称形如 `h_1` / `h_2`（见 `generate_kmer.py`），原始 fasta 中没有这些半段，因此 `single_easy_bin` / `multi_easy_bin` 内置的自动提取对 split 有局限，推荐用 `generate_berts.py` 显式生成。

DNABERT 的具体用法见 [usage.md](usage.md) 与 [generate.md](generate.md)。

## 引用

如果本工具对你的研究有帮助，请引用上游 SemiBin / SemiBin2：

- Pan et al., *Nat Commun* 13, 2326 (2022). <https://doi.org/10.1038/s41467-022-29843-y>
- Pan et al., *Bioinformatics* 39(Suppl_1): i21–i29 (2023). <https://doi.org/10.1093/bioinformatics/btad209>

如果使用了 DNABERT 特征，请同时引用 DNABERT-S（[MAGICS-LAB/DNABERT_S](https://github.com/MAGICS-LAB/DNABERT_S)）。

引用信息也可通过 `SemiBin2 citation` 获取。
