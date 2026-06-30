# 使用 strobealign-aemb 丰度信息运行

本项目 **Multimodal SemiBin** 在 SemiBin2 (v2.2.0) 基础上扩展（派生自 [SemiBin/SemiBin2](https://github.com/BigDataBiology/SemiBin)，© BigDataBiology，MIT 许可）。本页介绍沿用上游的 `strobealign-aemb` 丰度估计流程。该功能本身未被本项目改动，可照常使用。

`strobealign-aemb` 是一种快速的宏基因组丰度估计方法，由 [strobealign](https://github.com/ksahlin/strobealign) 0.13（或更新）版本提供，可替代传统的 BAM/CRAM 比对来生成丰度信息。出于实现原因，该模式当前要求至少 **5 个样本**。

> 提示：用 aemb 得到的丰度信息会进入丰度分支编码，与本项目的 DNABERT 多模态特征可以叠加使用——两者来自不同模态，互不冲突。DNABERT 嵌入的生成方式见 [usage.md](usage.md) 与各子命令文档（[subcommands.md](subcommands.md)）。

## 准备 aemb 所需数据

需要以 **all-by-all（全对全）** 的方式为每个样本生成丰度文件，即每个样本的 reads 都要比对到所有样本的 contigs 上。

下面以单个样本演示，但该流程 **必须对每个样本重复执行**。

### 1. 用 `split_contigs` 子命令拆分 fasta

```bash
mkdir -p aemb_output/sample1
SemiBin2 split_contigs -i sample1_contigs.fna.gz -o aemb_output/sample1
```

执行后会生成 `aemb_output/sample1/split_contigs.fna.gz`，其中包含原始 contigs 以及拆分后的版本（split.fa，运行 SemiBin2 所必需）。

### 2. 用 strobealign-aemb 映射 reads 生成丰度信息

需要 strobealign 0.13（或更新）版本：

```bash
strobealign --aemb aemb_output/sample1/split_contigs.fna.gz read1.pair.1.fq.gz read1.pair.2.fq.gz -R 6 -t 8 > sample1_sample1.tsv
strobealign --aemb aemb_output/sample1/split_contigs.fna.gz read2.pair.1.fq.gz read2.pair.2.fq.gz -R 6 -t 8 > sample1_sample2.tsv
strobealign --aemb aemb_output/sample1/split_contigs.fna.gz read3.pair.1.fq.gz read3.pair.2.fq.gz -R 6 -t 8 > sample1_sample3.tsv
strobealign --aemb aemb_output/sample1/split_contigs.fna.gz read4.pair.1.fq.gz read4.pair.2.fq.gz -R 6 -t 8 > sample1_sample4.tsv
strobealign --aemb aemb_output/sample1/split_contigs.fna.gz read5.pair.1.fq.gz read5.pair.2.fq.gz -R 6 -t 8 > sample1_sample5.tsv
```

每条命令生成一个 `sample1_sampleX.tsv` 格式的丰度文件。

## 运行 SemiBin2

用 `single_easy_bin` 子命令对该样本运行，通过 `-a` 传入上一步生成的所有 `.tsv` 丰度文件：

```bash
SemiBin2 single_easy_bin -i contig.fa -a sample1_*.tsv -o aemb_output/sample1
```

⚠️ 应当对 **原始 contigs** 进行分箱，**不要** 用拆分后的 split contigs。

结果 bins 会写入 `aemb_output/sample1` 目录。

注意：从 SemiBin2 的角度看，即使丰度信息来自多个样本，这仍属于 **单样本分箱**，因为组装结果来自单一样本。

> 可选：在 `single_easy_bin` 上叠加 DNABERT 多模态时，可加入 `--dnabert-model` 等参数，并相应调整图融合权重（`--fusion-weights-multimodal`）。参数细节见 [subcommands.md](subcommands.md)。

## 一个跑全对全丰度估计的辅助脚本

上述流程需要对每个样本重复，容易繁琐且出错。下面是一个用 [Jug](https://jug.readthedocs.io/en/latest/) 自动化全样本流程的辅助脚本。它最适合配合 Jug 做并行化；若移除 `@TaskGenerator` 装饰器，则会顺序执行。

脚本预期如下目录结构：

- `samples/`：组装好的 contigs，命名为 `sample1_assembled.fna.gz`、`sample2_assembled.fna.gz` ……
- `clean-reads/`：reads，命名为 `sample1.pair.1.fq.gz`、`sample1.pair.2.fq.gz`、`sample2.pair.1.fq.gz` ……
- `aemb_output/`：输出目录

```python
from jug import TaskGenerator
from jug.utils import jug_execute
import subprocess
from os import makedirs, path
import yaml

samples = [
        'sample0',
        'sample1',
        'sample2',
        'sample3',
        'sample4',
        'sample5',
        'sample6',
        'sample7',
        ]

STROBEALIGN_THREADS = 8
SEMIBIN_THREADS = STROBEALIGN_THREADS

@TaskGenerator
def generate_inputs(s):
    contigs = f'samples/{s}_assembled.fna.gz'
    if not path.exists(contigs):
        raise IOError(f'Expected contig file {f} (for sample {s})')
    out = f'aemb_output/{s}'
    makedirs(out, exist_ok=True)
    subprocess.check_call(
            ['SemiBin2', 'split_contigs',
             '-i', contigs,
             '-o', out])
    return out

@TaskGenerator
def cross_map(ref_out, ref_s, s):
    f1 = f'clean-reads/{s}.pair.1.fq.gz'
    f2 = f'clean-reads/{s}.pair.2.fq.gz'
    if not path.exists(f1):
        raise IOError(f'Expected reads file {f1} (for sample {s})')
    if not path.exists(f2):
        raise IOError(f'Expected reads file {f2} (for sample {s}). Note that {f1} does exist!')
    ofile = f'aemb_output/{ref_s}/mapped_{s}.tsv'
    with open(ofile, 'wb') as out:
        subprocess.check_call(
            ['strobealign',
                '--aemb', f'{ref_out}/split_contigs.fna.gz',
                f1, f2,
                '-t', str(STROBEALIGN_THREADS),
                '-R', '6'],
            stdout=out)
    return ofile


for s in samples:
    out = generate_inputs(s)
    tsv = []
    for s2 in samples:
        tsv.append(cross_map(out, s, s2))

    sb = jug_execute(
        ['SemiBin2', 'single_easy_bin',
            '--threads', str(SEMIBIN_THREADS),
            '-i', f'samples/{s}_assembled.fna.gz',
            '-a'] + tsv + [
                '-o', f'aemb_output/{s}'])
```

## 相关文档

- [usage.md](usage.md)：总体用法
- [subcommands.md](subcommands.md)：子命令与参数（含 DNABERT / 图融合参数）
- [generate.md](generate.md)：特征生成
- [output.md](output.md)：输出说明

## 致谢与引用

本流程沿用上游 SemiBin2，请引用：

- Pan et al., *Nat Commun* 13, 2326 (2022). https://doi.org/10.1038/s41467-022-29843-y
- Pan et al., *Bioinformatics* 39(Suppl_1): i21–i29 (2023). https://doi.org/10.1093/bioinformatics/btad209

若使用了 DNABERT 多模态特征，请同时引用 [DNABERT-S](https://github.com/MAGICS-LAB/DNABERT_S)。
