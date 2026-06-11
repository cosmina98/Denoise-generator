import contextlib
from typing import Any, List, Optional

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset, random_split


class _EdgeDiffusionDataset(Dataset):
    def __init__(self, node_features, conditions, adjacencies, node_masks):
        self.node_features = torch.tensor(node_features, dtype=torch.float32)
        self.conditions = torch.tensor(conditions, dtype=torch.float32)
        self.adjacencies = torch.tensor(adjacencies, dtype=torch.float32)
        self.node_masks = torch.tensor(node_masks, dtype=torch.bool)

    def __len__(self):
        return self.node_features.shape[0]

    def __getitem__(self, idx):
        return (
            self.node_features[idx],
            self.conditions[idx],
            self.adjacencies[idx],
            self.node_masks[idx],
        )


class _EdgeDiffusionNet(nn.Module):
    def __init__(
        self,
        node_dim: int,
        condition_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.node_proj = nn.Sequential(
            nn.LayerNorm(node_dim),
            nn.Linear(node_dim, hidden_dim),
            nn.SiLU(),
        )
        self.cond_proj = nn.Sequential(
            nn.LayerNorm(condition_dim),
            nn.Linear(condition_dim, hidden_dim),
            nn.SiLU(),
        )

        in_dim = 4 * hidden_dim + hidden_dim + 2
        layers = []
        for layer_idx in range(num_layers):
            layers.append(nn.Linear(in_dim if layer_idx == 0 else hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, 1))
        self.edge_mlp = nn.Sequential(*layers)

    def forward(self, x, c, a_t, t, pair_index):
        h = self.node_proj(x)
        g = self.cond_proj(c)

        i, j = pair_index[:, 0], pair_index[:, 1]
        h_i = h[:, i, :]
        h_j = h[:, j, :]
        pair_a = a_t[:, i, j].unsqueeze(-1)
        pair_t = t.view(-1, 1, 1).expand(-1, pair_index.shape[0], 1)
        pair_g = g.unsqueeze(1).expand(-1, pair_index.shape[0], -1)

        z = torch.cat([h_i, h_j, torch.abs(h_i - h_j), h_i * h_j, pair_g, pair_a, pair_t], dim=-1)
        return self.edge_mlp(z).squeeze(-1)


class ConditionalEdgeDiffusionGenerator:
    """
    Conditional adjacency denoiser.

    This is intentionally separate from the node-table generator: it learns
    A_t -> A0 conditioned on C and a node scaffold X.
    """
    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.05,
        learning_rate: float = 3e-4,
        maximum_epochs: int = 500,
        batch_size: int = 32,
        total_steps: int = 50,
        edge_diffusion_mode: str = "absorbing_empty",
        balance_edge_loss: bool = True,
        condition_noise: float = 0.0,
        condition_dropout: float = 0.0,
        condition_noise_start_col: int = 3,
        verbose: bool = False,
        device: Optional[str] = None,
        random_state: int = 42,
    ):
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.maximum_epochs = maximum_epochs
        self.batch_size = batch_size
        self.total_steps = total_steps
        self.edge_diffusion_mode = str(edge_diffusion_mode).lower()
        if self.edge_diffusion_mode not in {"absorbing_empty", "random_replace"}:
            raise ValueError("edge_diffusion_mode must be 'absorbing_empty' or 'random_replace'")
        self.balance_edge_loss = bool(balance_edge_loss)
        self.condition_noise = float(condition_noise)
        self.condition_dropout = float(condition_dropout)
        self.condition_noise_start_col = int(condition_noise_start_col)
        if self.condition_noise < 0:
            raise ValueError("condition_noise must be non-negative")
        if not 0.0 <= self.condition_dropout < 1.0:
            raise ValueError("condition_dropout must be in [0, 1)")
        self.verbose = verbose
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.random_state = random_state

        self.model = None
        self.x_scaler = None
        self.c_scaler = None
        self.number_of_rows_per_example = None
        self.pair_index = None
        self.train_edge_density = 0.1

    def _pad_node_features(self, node_encodings_list):
        max_rows = self.number_of_rows_per_example or max(x.shape[0] for x in node_encodings_list)
        padded = []
        masks = []
        for x in node_encodings_list:
            x = np.asarray(x, dtype=float)
            n = x.shape[0]
            if n < max_rows:
                x = np.pad(x, ((0, max_rows - n), (0, 0)), mode="constant", constant_values=0)
            padded.append(x[:max_rows])
            mask = np.zeros(max_rows, dtype=bool)
            mask[:min(n, max_rows)] = True
            masks.append(mask)
        return np.stack(padded, axis=0), np.stack(masks, axis=0)

    def _graphs_to_adjacency(self, graphs: List[nx.Graph], max_rows: int):
        mats = []
        for graph in graphs:
            nodes = list(graph.nodes())
            adj = nx.to_numpy_array(graph, nodelist=nodes, dtype=float)
            if adj.shape[0] < max_rows:
                adj = np.pad(adj, ((0, max_rows - adj.shape[0]), (0, max_rows - adj.shape[1])))
            adj = adj[:max_rows, :max_rows]
            adj = np.maximum(adj, adj.T)
            np.fill_diagonal(adj, 0.0)
            mats.append(adj)
        return np.stack(mats, axis=0)

    def _scale_inputs(self, X, C, fit=False):
        B, N, D = X.shape
        if fit:
            self.x_scaler = MinMaxScaler().fit(X.reshape(-1, D))
            self.c_scaler = MinMaxScaler().fit(C)
        Xs = self.x_scaler.transform(X.reshape(-1, D)).reshape(B, N, D)
        Cs = self.c_scaler.transform(C)
        return Xs, Cs

    def _make_pair_index(self, n_rows: int):
        pairs = [(i, j) for i in range(n_rows) for j in range(i + 1, n_rows)]
        return torch.tensor(pairs, dtype=torch.long, device=self.device)

    def _corrupt_adjacency(self, adj, t):
        B, N, _ = adj.shape
        p = t.view(B, 1, 1)
        if self.edge_diffusion_mode == "absorbing_empty":
            keep = (torch.rand_like(adj) > p).float()
            noisy = adj * keep
        else:
            replace = (torch.rand_like(adj) < p).float()
            random_edges = (torch.rand_like(adj) < self.train_edge_density).float()
            noisy = adj * (1.0 - replace) + random_edges * replace
        noisy = torch.triu(noisy, diagonal=1)
        noisy = noisy + noisy.transpose(1, 2)
        return noisy

    def _corrupt_node_condition(self, x):
        if not self.model.training:
            return x
        start = min(max(self.condition_noise_start_col, 0), x.shape[-1])
        if start >= x.shape[-1]:
            return x
        if self.condition_noise <= 0 and self.condition_dropout <= 0:
            return x

        x = x.clone()
        tail = x[..., start:]
        if self.condition_noise > 0:
            tail = tail + self.condition_noise * torch.randn_like(tail)
        if self.condition_dropout > 0:
            keep = (torch.rand_like(tail) >= self.condition_dropout).to(tail.dtype)
            tail = tail * keep
        x[..., start:] = tail
        return x

    def fit(self, graphs: List[nx.Graph], node_encodings_list: List[np.ndarray], conditional_graph_encodings: Any):
        if len(graphs) == 0:
            raise ValueError("Cannot train edge diffusion on an empty graph list.")

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        self.number_of_rows_per_example = max(x.shape[0] for x in node_encodings_list)
        X, masks = self._pad_node_features(node_encodings_list)
        C = np.asarray(conditional_graph_encodings, dtype=float)
        A = self._graphs_to_adjacency(graphs, self.number_of_rows_per_example)
        Xs, Cs = self._scale_inputs(X, C, fit=True)

        iu = np.triu_indices(self.number_of_rows_per_example, k=1)
        valid_pair_mask = masks[:, iu[0]] & masks[:, iu[1]]
        valid_targets = A[:, iu[0], iu[1]][valid_pair_mask]
        self.train_edge_density = float(valid_targets.mean()) if valid_targets.size else 0.1

        dataset = _EdgeDiffusionDataset(Xs, Cs, A, masks)
        val_size = max(1, int(round(0.1 * len(dataset)))) if len(dataset) > 1 else 0
        train_size = len(dataset) - val_size
        if val_size:
            train_dataset, val_dataset = random_split(
                dataset,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(self.random_state),
            )
        else:
            train_dataset, val_dataset = dataset, None

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False) if val_dataset else None

        self.model = _EdgeDiffusionNet(
            node_dim=Xs.shape[-1],
            condition_dim=Cs.shape[-1],
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)
        self.pair_index = self._make_pair_index(self.number_of_rows_per_example)

        opt = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        best_state = None
        best_val = float("inf")
        patience = min(100, max(20, self.maximum_epochs // 5))
        stale = 0

        for epoch in range(self.maximum_epochs):
            train_loss = self._run_epoch(train_loader, opt)
            val_loss = self._run_epoch(val_loader, None) if val_loader is not None else train_loss

            if val_loss < best_val - 1e-5:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                stale = 0
            else:
                stale += 1

            if self.verbose and (epoch == 0 or (epoch + 1) % 25 == 0):
                print(f"edge diffusion epoch {epoch + 1}: train={train_loss:.4f} val={val_loss:.4f}")

            if stale >= patience:
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()
        return self

    def _run_epoch(self, loader, opt):
        if loader is None:
            return float("nan")
        training = opt is not None
        self.model.train(training)
        total = 0.0
        count = 0
        ctx = contextlib.nullcontext() if training else torch.no_grad()
        with ctx:
            for x, c, adj, mask in loader:
                x = x.to(self.device)
                c = c.to(self.device)
                adj = adj.to(self.device)
                mask = mask.to(self.device)
                t = torch.rand(x.shape[0], 1, device=self.device)
                a_t = self._corrupt_adjacency(adj, t)
                x_cond = self._corrupt_node_condition(x)
                logits = self.model(x_cond, c, a_t, t, self.pair_index)

                i, j = self.pair_index[:, 0], self.pair_index[:, 1]
                valid = mask[:, i] & mask[:, j]
                targets = adj[:, i, j]
                if not valid.any():
                    continue
                logits = logits[valid]
                targets = targets[valid]
                if self.balance_edge_loss:
                    n_pos = targets.sum()
                    n_neg = targets.numel() - n_pos
                    pos_weight = torch.clamp(n_neg / (n_pos + 1e-8), min=1.0, max=20.0)
                    loss = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)
                else:
                    loss = F.binary_cross_entropy_with_logits(logits, targets)

                if training:
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()

                total += float(loss.detach().cpu()) * int(targets.numel())
                count += int(targets.numel())
        return total / max(count, 1)

    def predict_edge_probabilities(
        self,
        conditional_graph_encodings: Any,
        node_encodings_list: List[np.ndarray],
    ) -> Optional[List[np.ndarray]]:
        if self.model is None:
            return None
        if len(node_encodings_list) == 0:
            return []

        X, masks = self._pad_node_features(node_encodings_list)
        C = np.asarray(conditional_graph_encodings, dtype=float)
        Xs, Cs = self._scale_inputs(X, C, fit=False)

        x = torch.tensor(Xs, dtype=torch.float32, device=self.device)
        c = torch.tensor(Cs, dtype=torch.float32, device=self.device)
        mask = torch.tensor(masks, dtype=torch.bool, device=self.device)
        B, N, _ = x.shape

        if self.edge_diffusion_mode == "absorbing_empty":
            a = torch.zeros((B, N, N), dtype=torch.float32, device=self.device)
        else:
            a = (torch.rand((B, N, N), device=self.device) < self.train_edge_density).float()
            a = torch.triu(a, diagonal=1)
            a = a + a.transpose(1, 2)

        self.model.eval()
        with torch.no_grad():
            for step in range(max(1, self.total_steps)):
                t_value = 1.0 - (step / max(1, self.total_steps - 1))
                t = torch.full((B, 1), t_value, dtype=torch.float32, device=self.device)
                logits = self.model(x, c, a, t, self.pair_index)
                probs = torch.sigmoid(logits)
                i, j = self.pair_index[:, 0], self.pair_index[:, 1]
                valid = mask[:, i] & mask[:, j]
                next_a = torch.zeros_like(a)
                next_a[:, i, j] = torch.where(valid, probs, torch.zeros_like(probs))
                next_a[:, j, i] = next_a[:, i, j]
                a = next_a

        prob_matrices = []
        a_np = a.detach().cpu().numpy()
        for enc, pm in zip(node_encodings_list, a_np):
            n = enc.shape[0]
            out = pm[:n, :n].copy()
            np.fill_diagonal(out, 0.0)
            prob_matrices.append(out)
        return prob_matrices
