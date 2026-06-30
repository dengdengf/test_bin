import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset


def _make_branch(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.LeakyReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden_dim, output_dim),
        nn.BatchNorm1d(output_dim),
        nn.LeakyReLU(),
    )


class _ZeroBranch(nn.Module):
    def __init__(self, output_dim: int):
        super().__init__()
        self.output_dim = output_dim

    def forward(self, x):
        return torch.zeros((x.shape[0], self.output_dim), device=x.device, dtype=x.dtype)


class MultiModalContrastiveModel(nn.Module):
    is_multimodal = True

    def __init__(self, composition_dim: int, abundance_dim: int, dna_dim: int,
                 branch_dim: int = 64, fused_dim: int = 100):
        super().__init__()
        self.composition_dim = composition_dim
        self.abundance_dim = abundance_dim
        self.dna_dim = dna_dim
        self.branch_dim = branch_dim
        self.fused_dim = fused_dim

        self.comp_encoder = _make_branch(composition_dim, 256, branch_dim)
        self.abund_encoder = (
            _make_branch(abundance_dim, 128, branch_dim)
            if abundance_dim > 0 else _ZeroBranch(branch_dim)
        )
        self.dna_encoder = _make_branch(dna_dim, 256, branch_dim)

        self.gate = nn.Sequential(
            nn.Linear(branch_dim * 3, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 3),
        )
        self.fusion = nn.Sequential(
            nn.Linear(branch_dim * 3, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, fused_dim),
        )

    def encode_modalities(self, composition, abundance, dna):
        comp_h = self.comp_encoder(composition)
        abund_h = self.abund_encoder(abundance if abundance is not None else composition)
        dna_h = self.dna_encoder(dna)

        stacked = torch.cat([comp_h, abund_h, dna_h], dim=1)
        gate_logits = self.gate(stacked)
        gate_weights = torch.softmax(gate_logits, dim=1)

        weighted = torch.cat([
            comp_h * gate_weights[:, 0:1],
            abund_h * gate_weights[:, 1:2],
            dna_h * gate_weights[:, 2:3],
        ], dim=1)
        fused = self.fusion(weighted)
        return {
            'composition': comp_h,
            'abundance': abund_h,
            'dna': dna_h,
            'fused': fused,
            'gate_weights': gate_weights,
            'has_abundance': self.abundance_dim > 0,
        }

    def embedding_from_parts(self, composition, abundance, dna):
        return self.encode_modalities(composition, abundance, dna)['fused']

    def forward(self, left, right):
        left_out = self.encode_modalities(
            left['composition'], left.get('abundance'), left['dna'])
        right_out = self.encode_modalities(
            right['composition'], right.get('abundance'), right['dna'])
        return left_out, right_out

    def save_with_params_to(self, path):
        torch.save({
            'model_name': 'MultiModalContrastiveModel',
            'model_state_dict': self.state_dict(),
            'params': {
                'composition_dim': self.composition_dim,
                'abundance_dim': self.abundance_dim,
                'dna_dim': self.dna_dim,
                'branch_dim': self.branch_dim,
                'fused_dim': self.fused_dim,
            },
        }, path)


class MultiModalPairDataset(Dataset):
    def __init__(self, left_comp, left_abund, left_dna,
                 right_comp, right_abund, right_dna, labels):
        self.left_comp = left_comp
        self.left_abund = left_abund
        self.left_dna = left_dna
        self.right_comp = right_comp
        self.right_abund = right_abund
        self.right_dna = right_dna
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'left': {
                'composition': self.left_comp[idx],
                'abundance': self.left_abund[idx],
                'dna': self.left_dna[idx],
            },
            'right': {
                'composition': self.right_comp[idx],
                'abundance': self.right_abund[idx],
                'dna': self.right_dna[idx],
            },
            'label': self.labels[idx],
        }


def contrastive_loss(embedding1, embedding2, label):
    relu = torch.nn.ReLU()
    dist = torch.norm(embedding1 - embedding2, p=2, dim=1)
    positive = torch.square(dist)
    negative = torch.square(relu(1 - dist))
    return torch.mean(label * positive + (1 - label) * negative)


def alignment_loss(left_out, right_out):
    # Align the composition / abundance branches TOWARD the DNABERT branch, but treat the
    # DNABERT embedding as a fixed target via stop-gradient (.detach()). Without the detach
    # the symmetric MSE pulls every branch toward a moving mean, which lets the encoders
    # collapse modality-specific signal into whatever the DNA branch happens to emit. With
    # the detach, DNABERT acts as a stable anchor and only the weaker modalities are nudged.
    mse = torch.nn.MSELoss()
    losses = [
        mse(left_out['composition'], left_out['dna'].detach()),
        mse(right_out['composition'], right_out['dna'].detach()),
    ]
    if left_out.get('has_abundance', False):
        losses.extend([
            mse(left_out['abundance'], left_out['dna'].detach()),
            mse(right_out['abundance'], right_out['dna'].detach()),
        ])
    return sum(losses) / len(losses)


def _split_features(frame_values: np.ndarray):
    composition = frame_values[:, :136].astype(np.float32)
    abundance = frame_values[:, 136:].astype(np.float32) if frame_values.shape[1] > 136 else np.zeros((len(frame_values), 0), dtype=np.float32)
    return composition, abundance


def _validate_embedding_alignment(logger, frame: pd.DataFrame, embedding: np.ndarray, names_path: str):
    if len(frame) != len(embedding):
        raise ValueError(f'Embedding rows ({len(embedding)}) do not match data rows ({len(frame)})')
    if names_path is not None:
        with open(names_path, 'r', encoding='utf-8') as handle:
            names = [line.rstrip('\n') for line in handle]
        if names != frame.index.astype(str).tolist():
            raise ValueError(f'Embedding order in {names_path} does not match data index order')


def load_multimodal_embeddings(logger, data_path: str, data_split_path: str):
    import os
    base_dir = os.path.dirname(data_path)
    whole_path = f'{base_dir}/dnabert_embedding.npy'
    whole_names = f'{base_dir}/dnabert_contig_names.txt'
    split_path = f'{base_dir}/dnabert_split_embedding.npy'
    split_names = f'{base_dir}/dnabert_split_contig_names.txt'
    if not (path_exists(whole_path) and path_exists(split_path)):
        raise FileNotFoundError(f'Missing DNABERT features in {base_dir}')

    whole_frame = pd.read_csv(data_path, index_col=0)
    split_frame = pd.read_csv(data_split_path, index_col=0)

    whole = np.load(whole_path).astype(np.float32)
    split = np.load(split_path).astype(np.float32)

    _validate_embedding_alignment(logger, whole_frame, whole, whole_names if path_exists(whole_names) else None)
    _validate_embedding_alignment(logger, split_frame, split, split_names if path_exists(split_names) else None)
    return whole, split


def path_exists(path: str) -> bool:
    import os
    return os.path.exists(path)


def train_multimodal_self(logger, datapaths, data_splits, dnabert_embeddings,
                          dnabert_split_embeddings, batchsize=2048, epoches=15,
                          device=None, num_process=8, mode='single',
                          alignment_weight=0.1):
    torch.set_num_threads(num_process)

    first_data = pd.read_csv(datapaths[0], index_col=0)
    first_comp, first_abund = _split_features(first_data.values)

    model = MultiModalContrastiveModel(
        composition_dim=first_comp.shape[1],
        abundance_dim=first_abund.shape[1],
        dna_dim=dnabert_embeddings[0].shape[1],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    for epoch in range(epoches):
        logger.info(f'Training multimodal model (epoch {epoch + 1}/{epoches})')
        for data_index, (datapath, data_split_path) in enumerate(zip(datapaths, data_splits)):
            data = pd.read_csv(datapath, index_col=0)
            data.index = data.index.astype(str)
            data_split = pd.read_csv(data_split_path, index_col=0)
            data_split.index = data_split.index.astype(str)

            whole_comp, whole_abund = _split_features(data.values)
            split_comp, split_abund = _split_features(data_split.values)
            # In non-combined mode data_split carries only composition (no abundance), so
            # split_abund is width 0 while whole_abund is width A. The two halves of a contig
            # share their parent's abundance, so fill split abundance from the parent contig
            # (h_1 / h_2 -> h) to keep whole/split abundance dimensions aligned. Without this
            # the np.concatenate below raises and multimodal training silently falls back.
            if split_abund.shape[1] != whole_abund.shape[1]:
                if whole_abund.shape[1] > 0:
                    abund_cols = data.columns[136:]
                    parents = [name.rpartition('_')[0] for name in data_split.index]
                    split_abund = data.reindex(parents)[abund_cols].fillna(0.0).values.astype(np.float32)
                else:
                    split_abund = np.zeros((len(split_comp), 0), dtype=np.float32)
            whole_dna = dnabert_embeddings[data_index]
            split_dna = dnabert_split_embeddings[data_index]

            if len(data) != len(whole_dna):
                raise ValueError(f'Mismatch between {datapath} and DNABERT embeddings')
            if len(data_split) != len(split_dna):
                raise ValueError(f'Mismatch between {data_split_path} and split DNABERT embeddings')

            n_positive = len(split_comp)
            n_negative = min(max(n_positive * 500, 1), 4_000_000)
            data_length = len(whole_comp)
            if data_length < 2:
                raise ValueError(f'Expected at least 2 contigs in {datapath}')

            indices1 = np.random.choice(data_length, size=n_negative)
            indices2 = indices1 + 1 + np.random.choice(data_length - 1, size=n_negative)
            indices2 %= data_length

            left_comp = np.concatenate([whole_comp[indices1], split_comp[::2]], axis=0)
            right_comp = np.concatenate([whole_comp[indices2], split_comp[1::2]], axis=0)
            left_abund = np.concatenate([whole_abund[indices1], split_abund[::2]], axis=0)
            right_abund = np.concatenate([whole_abund[indices2], split_abund[1::2]], axis=0)
            left_dna = np.concatenate([whole_dna[indices1], split_dna[::2]], axis=0)
            right_dna = np.concatenate([whole_dna[indices2], split_dna[1::2]], axis=0)

            labels = np.zeros(len(left_comp), dtype=np.float32)
            labels[n_negative:] = 1.0

            dataset = MultiModalPairDataset(
                left_comp, left_abund, left_dna,
                right_comp, right_abund, right_dna,
                labels,
            )
            loader = DataLoader(
                dataset=dataset,
                batch_size=batchsize,
                shuffle=True,
                num_workers=0,
                drop_last=True,  # BatchNorm1d crashes on a trailing batch of size 1
            )

            for batch in loader:
                left = {
                    'composition': batch['left']['composition'].to(device=device, dtype=torch.float32),
                    'abundance': batch['left']['abundance'].to(device=device, dtype=torch.float32),
                    'dna': batch['left']['dna'].to(device=device, dtype=torch.float32),
                }
                right = {
                    'composition': batch['right']['composition'].to(device=device, dtype=torch.float32),
                    'abundance': batch['right']['abundance'].to(device=device, dtype=torch.float32),
                    'dna': batch['right']['dna'].to(device=device, dtype=torch.float32),
                }
                label = batch['label'].to(device=device, dtype=torch.float32)

                model.train()
                optimizer.zero_grad()
                left_out, right_out = model(left, right)
                contrast = contrastive_loss(left_out['fused'], right_out['fused'], label)
                align = alignment_loss(left_out, right_out)
                loss = contrast + alignment_weight * align
                loss.backward()
                optimizer.step()

        scheduler.step()

    logger.info('Multimodal training finished.')
    return model
