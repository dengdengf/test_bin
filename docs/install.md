# 安装指南

本页介绍如何安装 **MAGFuse** —— 一个独立的多模态宏基因组分箱工具（MIT 许可）。

本项目通过源码安装，命令行入口为 `MAGFuse`。

> 关于 MAGFuse 的核心方法与特性（多模态嵌入、多视图图融合聚类、标记基因去污染重聚类、DNABERT 特征提取），请参见 [methods.md](methods.md)、[whatsnew.md](whatsnew.md) 与 [usage.md](usage.md)。

## Python 版本要求

支持 Python **3.7 – 3.13**。

## 1. 外部依赖

MAGFuse 运行需要以下外部命令行工具，推荐用 conda 从 bioconda 安装：

```bash
conda create -n MAGFuse python
conda activate MAGFuse
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

安装完成后即可使用 `MAGFuse` 命令。验证：

```bash
MAGFuse --version
MAGFuse --help
```

> 如需 GPU 加速，请按 [PyTorch 官网](https://pytorch.org/get-started/locally/) 的说明先安装带 CUDA 支持的 PyTorch，再执行上面的 `pip install .`。

## 3. DNABERT 依赖

DNABERT 多模态是 MAGFuse 短读长流程默认且始终启用的核心组成部分，因此短读长分箱需要安装以下组件。

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

### 3.3 关于 DNABERT 嵌入文件

DNABERT 嵌入需要 **whole + split 两份**，二者共享同一个 PCA basis（在 whole 上 fit、对 split 用 transform）。`single_easy_bin` / `multi_easy_bin` 会**自动提取 whole 与 split**（`h_1` / `h_2` 由父 contig 自动切半），并共享同一 PCA basis，**无需手动**。如需离线/单独生成嵌入，可用等价工具 `SemiBin/generate_berts.py`：

```bash
python SemiBin/generate_berts.py -md /path/DNABERT-S \
  -fd whole.fasta -nd output/dnabert_contig_names.txt -dd output/dnabert_embedding.npy \
  -sfd split.fasta -snd output/dnabert_split_contig_names.txt -sdd output/dnabert_split_embedding.npy
```

注意事项：

- 输出文件名必须是 `dnabert_embedding.npy` / `dnabert_split_embedding.npy`，放在 `data.csv` 同目录；
- fasta 行序须与 `data.csv` / `data_split.csv` 一致（`load_multimodal_embeddings` 会逐行校验）；
- split 半段名称形如 `h_1` / `h_2`（见 `generate_kmer.py`）：自动提取（含 split，共享 PCA basis）已内置处理；`generate_berts.py` 是离线/单独生成嵌入的等价工具。

DNABERT 的具体用法见 [usage.md](usage.md) 与 [generate.md](generate.md)。
