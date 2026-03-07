# Materials and Methods

## Overview of the method

An enhanced short-read metagenomic binning framework was developed to support both single-sample and multiple-sample analysis. The method takes assembled contigs together with read-alignment-derived abundance information as input, and performs binning through four major stages: feature construction, contig representation learning, initial graph-based clustering, and DNABERT-guided bin refinement. In contrast to methods relying solely on composition and abundance statistics, the proposed framework introduces a pretrained DNA language model branch to capture contextual sequence semantics at the contig level. The resulting design allows the method to use conventional contig signatures for global bin formation, while exploiting pretrained sequence representations for the finer separation of contaminated bins.

Given an input contig set, tetranucleotide frequency and abundance features are first generated for each contig. Long contigs are then split into paired fragments to construct self-supervised positive pairs for contrastive representation learning. In parallel, a DNABERT model is used to encode full-length contigs and generate sequence-level embeddings. The learned contig embeddings are employed to construct a nearest-neighbor graph for initial clustering. After that, bins with evidence of contamination are identified using single-copy marker genes and further refined in a feature space composed of DNABERT representations and normalized abundance profiles. The complete workflow is implemented for both short-read single-sample and short-read multiple-sample modes.

## Input data and feature generation

### Contig and abundance inputs

The method accepts assembled contigs in FASTA format and either sorted BAM/CRAM files or precomputed abundance files. In the single-sample mode, one assembled contig set and its corresponding abundance source are used. In the multiple-sample mode, contigs from different samples are concatenated into a single FASTA file, and each contig header contains a sample identifier so that sample-specific contigs can later be separated and processed independently.

### Tetranucleotide frequency features

For each contig, canonical 4-mer frequencies are calculated as composition features. Reverse-complement k-mers are treated as equivalent, resulting in a 136-dimensional representation. To avoid instability caused by zero counts, a small pseudocount is added before row-wise normalization. These normalized tetranucleotide frequencies are used as the basic composition features for both training and clustering.

### Construction of self-supervised positive pairs

To generate reliable positive pairs without external labels, contigs longer than a predefined must-link threshold are split at the midpoint into two fragments. The two fragments are regarded as originating from the same genome and therefore form a positive pair. The split fragments are also used to generate paired composition and abundance features, which are stored separately from the original contig features and later used during training.

### Abundance feature generation

When BAM or CRAM files are provided, abundance features are calculated from read depth along each contig. For short-read single-sample binning, the mean coverage and coverage variance are calculated for each contig. For combined abundance settings, especially when multiple samples are available, abundance is represented as a cross-sample coverage vector. The same procedure is applied to split contigs so that both the original contig set and the split contig set have aligned abundance feature matrices.

When abundance files are used directly, the abundance matrix is parsed and reorganized according to contig identities and split-fragment indices. In all cases, the original contig-level composition and abundance features are merged into `data.csv`, while the corresponding split-contig features are merged into `data_split.csv`.

## Contig representation learning

### Neural network architecture

Contig representation learning is performed with a siamese-style encoder composed of fully connected layers. For an input feature vector $x$, the encoder maps the input through three linear layers with dimensions `input -> 512 -> 512 -> 100`. The first two hidden layers are followed by batch normalization, LeakyReLU activation, and dropout with a rate of 0.2. The output of the final layer is a 100-dimensional embedding used in downstream graph construction and clustering.

For single-sample training, the encoder operates on the 136-dimensional tetranucleotide feature vector. For combined abundance settings, the encoder input is the concatenation of composition and abundance features. Although the input dimensionality differs between modes, the encoder topology remains the same.

### Self-supervised pair construction

Two types of training pairs are used. Positive pairs are generated from the two fragments derived from the same split contig. Negative pairs are constructed by randomly sampling pairs of different contigs from the dataset. Let $N_{+}$ denote the number of positive pairs. The number of negative pairs is set to $\min(500N_{+}, 4{,}000{,}000)$, which provides sufficient contrastive supervision while avoiding excessive memory usage.

### Training objective

The encoder is trained with a contrastive loss based on Euclidean distance. For a training pair with embeddings $(z_i, z_j)$ and binary label $y \in \{0,1\}$, where $y=1$ denotes a positive pair and $y=0$ denotes a negative pair, the loss is defined as

\[
\mathcal{L} = y \cdot ||z_i-z_j||_2^2 + (1-y)\cdot \max(0, 1-||z_i-z_j||_2)^2.
\]

This objective encourages embeddings from the same contig to remain close while separating randomly paired contigs by at least a unit margin. Model training uses the Adam optimizer with an initial learning rate of $10^{-3}$. A StepLR scheduler with decay factor 0.9 is applied after each epoch. Unless otherwise specified, the batch size is 2048 and the number of training epochs is 15.

## DNABERT-based sequence representation

### Extraction of sequence embeddings

To complement composition and abundance features, a pretrained DNABERT model is used to encode each contig sequence. Each contig is tokenized with the DNABERT tokenizer and truncated to a maximum sequence length of 5000. The tokenized sequence is then passed through the pretrained model, and the hidden states from the last layer are averaged along the sequence dimension to obtain a sequence-level embedding for the contig.

### Dimensionality reduction and alignment

Because the raw DNABERT output has a higher dimensionality than the downstream refinement module requires, principal component analysis is used to reduce the embedding dimension to at most 128. When the number of contigs is smaller than 128, the reduced dimensionality is set to the number of available samples. To guarantee strict one-to-one correspondence between feature branches, DNABERT embeddings are saved in the same contig order as the rows in `data.csv`, and are loaded only if both contig identities and row counts match exactly.

### Role of the DNABERT branch

The DNABERT branch is not used as a replacement for conventional binning features. Instead, it provides an additional sequence-level representation that complements the learned contig embedding and abundance signals. In the proposed framework, DNABERT features are mainly used in two places: first, to constrain the construction of the initial graph when the hybrid mode is enabled; second, to drive the reclustering of bins suspected to contain contamination.

## Initial graph construction and clustering

### Graph construction without DNABERT

When the DNABERT branch is not activated, an embedding-based k-nearest-neighbor graph is constructed from the learned contig embeddings. To improve graph robustness, this graph is intersected with a second nearest-neighbor graph built directly from the input feature space. Only contig pairs that remain neighbors in both spaces are retained. This operation reduces spurious edges caused by noise in a single feature space.

### Hybrid graph construction with DNABERT

When DNABERT embeddings are available, two nearest-neighbor graphs are constructed independently, one from the learned contig embeddings and the other from the DNABERT sequence embeddings. The two graphs are then intersected so that only contig pairs supported by both the task-specific embedding space and the pretrained sequence representation space are preserved. This hybrid graph is intended to favor connections that are simultaneously supported by abundance-aware representation learning and contextual sequence similarity.

### Edge filtering and abundance correction

After graph construction, edge weights are transformed from distances into similarities and weak edges are removed by adaptive thresholding. For non-combined abundance settings, an additional abundance-consistency matrix is computed from coverage statistics and multiplied element-wise with the graph weights. This correction step makes contigs with inconsistent abundance patterns less likely to be connected even if they appear close in the embedding space.

### Infomap-based initial clustering

The resulting weighted graph is clustered using Infomap to obtain the initial bin assignments. Each contig is assigned a bin label, and bin FASTA files are written accordingly. These initial clusters serve as the starting point for subsequent contamination detection and bin refinement.

## Marker-gene-guided bin refinement

### Identification of potentially contaminated bins

After initial clustering, the contigs assigned to each bin are concatenated and screened with single-copy marker genes. Bins associated with more than one marker-derived seed set are considered potentially contaminated. Only such bins are passed to the refinement stage; bins without evidence of contamination are retained unchanged.

### Feature space for refinement

For each candidate contaminated bin, the DNABERT embedding of each contig is extracted and concatenated with the normalized abundance vector of that contig. The resulting refinement feature space is defined as

\[
F = [E_d \,\|\, A],
\]

where $E_d$ denotes the DNABERT embedding and $A$ denotes the normalized abundance profile. In contrast to the initial clustering stage, this refinement stage emphasizes sequence semantics more strongly, because the main objective is to separate mixed contigs within already formed bins rather than to organize the full contig set globally.

### Seed-anchored hard assignment

Marker-derived seeds are treated as anchors in the refinement feature space. For every contig in a contaminated bin, the Euclidean distance to all seed anchors is calculated, and the contig is assigned to the nearest anchor. To avoid retaining ambiguous contigs near cluster boundaries, only contigs whose nearest-anchor distance lies within the closest 60th percentile are kept during reassignment. This hard filtering step removes contigs with weak support for any refined sub-bin.

### Rollback strategy

A rollback mechanism is introduced to avoid over-splitting. If the refined sub-bins fail to separate the marker-derived seeds, or if the resulting sub-bins do not satisfy minimum size and contig count requirements, the refinement result is rejected and the original bin assignment is restored. Only when at least two refined sub-bins satisfy the marker-separation and size constraints is the new partition accepted.

## Workflows for single-sample and multiple-sample binning

### Single-sample workflow

In the single-sample mode, the workflow consists of the following steps: generation of composition and abundance features from one contig set and its mapped reads, construction of split-contig training pairs, extraction of DNABERT embeddings, training of the contig representation model, Infomap-based initial clustering, and DNABERT-guided refinement of contaminated bins. This workflow is designed for short-read samples in which sequence-derived information is particularly important for resolving closely related genomes.

### Multiple-sample workflow

In the multiple-sample mode, the concatenated FASTA file is first separated into sample-specific FASTA files according to the sample identifier embedded in each contig header. For each sample, `data.csv` and `data_split.csv` are generated independently, DNABERT embeddings are extracted independently, and the sample-specific representation learning, initial clustering, and bin refinement procedures are executed independently. The final outputs from all samples are then merged into a unified result directory.

Although the final binning is performed separately for each sample, the abundance feature construction stage can still benefit from cross-sample information when multiple samples are available. This design retains the discriminatory power of abundance co-variation while avoiding the complexity of directly clustering all contigs from all samples in a single joint graph.

## Implementation details

The framework is implemented in Python. The representation learning module is built with PyTorch, and the pretrained DNA language model branch is implemented using the Hugging Face Transformers library. Tetranucleotide frequencies are generated by an internal feature-construction module. Coverage statistics are computed from BAM files using `bedtools genomecov`. Principal component analysis is performed with scikit-learn, and graph clustering is carried out with the Infomap implementation provided by `igraph`.

Unless otherwise stated, the representation learning model is trained with batch size 2048, learning rate $10^{-3}$, and 15 epochs. DNABERT embeddings are generated by mean pooling over the last hidden layer and reduced to 128 dimensions when necessary. During refinement, abundance vectors are normalized before concatenation with DNABERT embeddings, and a 60th-percentile distance cutoff is used to remove weakly assigned contigs. Final outputs include bin FASTA files, bin summary tables, and contig-to-bin mapping tables for downstream evaluation.
