import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn.base import BaseEstimator, TransformerMixin
from transformers import AutoTokenizer
from typing import List, Literal, Dict
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# Custom Character-Level Tokenizer without EOS management
# -----------------------------
class CharTokenizer:
    def __init__(self, vocab: List[str], max_len: int):
        """
        Simple character-level tokenizer with fixed vocabulary and padding.
        """
        self.max_len = max_len
        self.vocab = vocab
        self.vocab_dict = {ch: i for i, ch in enumerate(vocab)}
        self.inv_vocab = {i: ch for ch, i in self.vocab_dict.items()}

    def __call__(self, texts: List[str], padding='max_length', truncation=True,
                 max_length=None, **kwargs) -> Dict[str, torch.Tensor]:
        max_len = max_length or self.max_len
        input_ids = []
        for text in texts:
            tokens = list(text)
            ids = [self.vocab_dict.get(token, self.vocab_dict['[PAD]']) for token in tokens]
            if truncation:
                ids = ids[:max_len]
            if padding == 'max_length':
                ids = ids + [self.vocab_dict['[PAD]']] * (max_len - len(ids))
            input_ids.append(ids)
        return {'input_ids': torch.tensor(input_ids)}

    def decode(self, ids: List[int]) -> str:
        tokens = [self.inv_vocab.get(i, '') for i in ids if self.inv_vocab.get(i, '') != '[PAD]']
        return ''.join(tokens).strip()

    def get_vocab(self) -> Dict[str, int]:
        return self.vocab_dict

# -----------------------------
# Dataset for Transform Training
# -----------------------------
class TextDataset(Dataset):
    def __init__(self, tokenizer, X: List[str], y: List[str], max_len: int):
        """
        For transform training:
          - X: the prefix (input)
          - y: the full instance (prefix + continuation) without any EOS token.
        """
        self.encodings = tokenizer(X, padding='max_length', truncation=True,
                                   max_length=max_len, return_tensors='pt')
        self.targets = tokenizer(y, padding='max_length', truncation=True,
                                 max_length=max_len, return_tensors='pt')['input_ids']

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {'input_ids': self.encodings['input_ids'][idx],
                'labels': self.targets[idx]}

    

# -----------------------------
# Modified ERIT Block: Full Output vs. Memory Update with Layerwise Top‑k and Headwise Scaling
# -----------------------------
class ERITBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, ff_hidden_dim: int,
                 entropy_weight: float, topk_percent: float,
                 head_decay: float = 0.75, max_len: int = 64):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.entropy_weight = entropy_weight
        self.topk_percent = topk_percent
        self.head_decay = head_decay
        self.max_len = max_len

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.rel_pos_bias = nn.Parameter(torch.zeros(num_heads, 2 * max_len - 1))

        self.ff = nn.Sequential(
            nn.Linear(dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, dim)
        )
        self.norm_full = nn.LayerNorm(dim)
        self.norm_memory = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, memory: torch.Tensor):
        B, T, D = x.size()
        H = self.num_heads
        d = self.head_dim

        combined = torch.cat([x, memory], dim=1)  # [B, T+M, D]
        total_len = combined.size(1)

        q = self.q_proj(x).view(B, T, H, d).transpose(1, 2)
        k = self.k_proj(combined).view(B, total_len, H, d).transpose(1, 2)
        v = self.v_proj(combined).view(B, total_len, H, d).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1))
        pos = torch.arange(T, device=x.device).unsqueeze(1) - torch.arange(total_len, device=x.device).unsqueeze(0)
        pos_clipped = torch.clamp(pos + self.max_len - 1, 0, 2 * self.max_len - 2)
        rel_bias = self.rel_pos_bias[:, pos_clipped]
        scores = scores + rel_bias.unsqueeze(0)

        attn_weights = F.softmax(scores / (d ** 0.5), dim=-1)
        entropy = -torch.sum(attn_weights * torch.log(attn_weights + 1e-8), dim=-1).mean()
        weighted_entropy = self.entropy_weight * entropy

        context = torch.matmul(attn_weights, v)
        attn_output = context.transpose(1, 2).contiguous().view(B, T, D)
        full_output = self.norm_full(x + attn_output)

        # Token importance scores: mean attention over heads and target tokens
        input_attn_weights = attn_weights[:, :, :, :T].mean(dim=1)
        token_scores = input_attn_weights.mean(dim=1)  # shape (B, T)

        # Multi-scale top-k per head
        pruned_tokens = []
        for h in range(H):
            scale = self.head_decay ** h
            k_frac = self.topk_percent * scale
            k_top = max(1, int(k_frac * T))
            _, topk_indices = torch.topk(token_scores, k=k_top, dim=1)
            gathered = torch.gather(full_output, 1, topk_indices.unsqueeze(-1).expand(-1, -1, D))
            pruned_tokens.append(gathered)

        memory_out = torch.cat(pruned_tokens, dim=1)  # concat over heads
        memory_out = self.norm_memory(memory_out + self.ff(memory_out))

        return full_output, memory_out, weighted_entropy

# -----------------------------
# ERIT Core Model
# -----------------------------
class ERITModel(pl.LightningModule):
    def __init__(self, vocab_size: int, dim: int = 128, depth: int = 4, num_heads: int = 4,
                 ff_hidden_dim: int = 256, max_len: int = 64,
                 entropy_schedule: list = None, topk_percent_schedule: list = None,
                 head_decay: float = 0.75):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, dim))
        self.blocks = nn.ModuleList([
            ERITBlock(dim, num_heads, ff_hidden_dim,
                      entropy_weight=entropy_schedule[i],
                      topk_percent=topk_percent_schedule[i],
                      head_decay=head_decay,
                      max_len=max_len)
            for i in range(depth)
        ])
        self.output_head = nn.Linear(dim, vocab_size)

    def forward(self, x: torch.Tensor):
        B, T = x.size()
        if T > self.pos_embedding.size(1):
            x = x[:, -self.pos_embedding.size(1):]
            T = self.pos_embedding.size(1)
        x = self.embedding(x) + self.pos_embedding[:, :T, :]
        memory = torch.zeros(B, 0, x.size(-1), device=x.device)
        entropy_list = []
        for block in self.blocks:
            full_output, memory, block_entropy = block(x, memory)
            entropy_list.append(block_entropy)
            x = full_output
        logits = self.output_head(x)
        total_entropy = sum(entropy_list)
        return logits, total_entropy, entropy_list

# -----------------------------
# ERIT Transformer Wrapper
# -----------------------------
class ERITTransformer(BaseEstimator, TransformerMixin):
    @staticmethod
    def _adjust_dim(dim, num_heads):
        if dim % num_heads != 0:
            dim = num_heads * (dim // num_heads + 1)
        return dim
    def __init__(self, model_name='gpt2', dim=128, depth=4, num_heads=4,
                 ff_hidden_dim=256, max_len=64, entropy_schedule=None,
                 topk_percent_schedule=None, head_decay=0.75, lr=1e-3,
                 mode: Literal['word', 'char'] = 'word', vocab=list("abc"), verbose: bool = False):
        self.model_name = model_name
        self.dim = self._adjust_dim(dim, num_heads)  # auto-adjust to be divisible
        self.depth = depth
        self.num_heads = num_heads
        self.ff_hidden_dim = ff_hidden_dim
        self.max_len = max_len
        self.lr = lr
        self.mode = mode
        self.verbose = verbose
        self.head_decay = head_decay

        if entropy_schedule is None:
            max_entropy_weight = 0.95
            epsilon = 0.05
            entropy_schedule = [
                epsilon + (max_entropy_weight - epsilon) * i / (depth - 1)
                for i in range(depth)
            ]
        self.entropy_schedule = entropy_schedule

        if topk_percent_schedule is None:
            topk_percent_schedule = [0.75] * depth
        self.topk_percent_schedule = topk_percent_schedule

        if self.mode == 'word':
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.tokenizer.pad_token = '[PAD]'
            self.get_vocab = self.tokenizer.get_vocab
            self.encode = lambda s: self.tokenizer(s, padding='max_length', truncation=True,
                                                   max_length=self.max_len, return_tensors='pt')
            self.decode = lambda ids: self.tokenizer.decode(ids, skip_special_tokens=True).strip()
        elif self.mode == 'char':
            if '[PAD]' not in vocab:
                vocab.append('[PAD]')
            self.tokenizer = CharTokenizer(vocab=vocab, max_len=self.max_len)
            self.decode = self.tokenizer.decode
            self.get_vocab = self.tokenizer.get_vocab
        else:
            raise ValueError("Mode must be 'word' or 'char'")

        self.vocab_size = len(self.get_vocab())
        self.model = ERITModel(
            vocab_size=self.vocab_size,
            dim=self.dim,
            depth=depth,
            num_heads=num_heads,
            ff_hidden_dim=ff_hidden_dim,
            max_len=max_len,
            entropy_schedule=self.entropy_schedule,
            topk_percent_schedule=self.topk_percent_schedule,
            head_decay=self.head_decay
        )
        
    def fit(self, X: List[str], y: List[str], epochs: int = 5, batch_size: int = 16, val_fraction: float = 0.2):
        """
        In transform mode, X is the prefix and y is the full instance.
        Randomly splits data into training and validation sets.

        Args:
            X (List[str]): Input prefix sequences.
            y (List[str]): Target full sequences.
            epochs (int): Number of training epochs.
            batch_size (int): Mini-batch size.
            val_fraction (float): Fraction of data to use for validation.
        """
        # Zip, shuffle, and split
        paired = list(zip(X, y))
        np.random.shuffle(paired)
        X, y = zip(*paired)

        split_idx = int((1 - val_fraction) * len(X))
        X_train, y_train = X[:split_idx], y[:split_idx]
        X_val, y_val = X[split_idx:], y[split_idx:]

        train_dataset = TextDataset(self.tokenizer, X_train, y_train, self.max_len)
        val_dataset = TextDataset(self.tokenizer, X_val, y_val, self.max_len)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        lit_module = LitModule(self.model, self.lr)
        trainer = pl.Trainer(
            max_epochs=epochs,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False
        )
        trainer.fit(lit_module, train_loader, val_loader)
        self.lit_module = lit_module  # store for visualization if desired
        return self


    def transform(self, X: List[str]) -> List[str]:
        self.model.eval()
        with torch.no_grad():
            encodings = self.tokenizer(X, padding='max_length', truncation=True, max_length=self.max_len)
            logits, _, _ = self.model(encodings['input_ids'])
            preds = torch.argmax(logits, dim=-1)
            return [self.decode(p.tolist()) for p in preds]

    def embed(self, X: List[str]) -> torch.Tensor:
        """
        Returns a single vector per instance by pooling the last layer representations
        and the final write memory, excluding [PAD] tokens from the input sequence.
        """
        self.model.eval()
        with torch.no_grad():
            encodings = self.tokenizer(X, padding='max_length', truncation=True, max_length=self.max_len)
            input_ids = encodings['input_ids']
            B, T = input_ids.size()
            # Get pad token ID from the vocabulary.
            pad_id = self.get_vocab()['[PAD]']
            # Create a mask for non-[PAD] tokens (1 for valid tokens, 0 for [PAD]).
            mask_x = (input_ids != pad_id).float()  # shape (B, T)
            # Compute input embeddings and add positional encodings.
            x = self.model.embedding(input_ids) + self.model.pos_embedding[:, :T, :]
            memory = torch.zeros(B, 0, x.size(-1), device=x.device)
            for block in self.model.blocks:
                full_output, memory, _ = block(x, memory)
                x = full_output
            # x has shape (B, T, D) and memory has shape (B, M, D)
            # For memory tokens, assume all are valid (mask of ones).
            mask_mem = torch.ones(memory.shape[:2], device=memory.device) if memory.size(1) > 0 else None
            # Combine x and memory along the sequence dimension.
            if mask_mem is not None:
                combined = torch.cat([x, memory], dim=1)  # shape (B, T+M, D)
                combined_mask = torch.cat([mask_x, mask_mem], dim=1)  # shape (B, T+M)
            else:
                combined = x
                combined_mask = mask_x
            # Expand mask to match dimensions of combined representation.
            combined_mask_exp = combined_mask.unsqueeze(-1)
            # Compute masked sum and normalize by the count of valid tokens.
            pooled = (combined * combined_mask_exp).sum(dim=1) / combined_mask.sum(dim=1, keepdim=True).clamp(min=1)
            return pooled

# -----------------------------
# Lightning Module for Training and Logging
# -----------------------------
class LitModule(pl.LightningModule):
    def __init__(self, model: nn.Module, lr: float):
        super().__init__()
        self.model = model
        self.lr = lr
        self.train_losses = []
        self.val_losses = []
        self.train_xent = []
        self.val_xent = []
        self.train_entropy_history = {i: [] for i in range(len(model.blocks))}
        self.val_entropy_history = {i: [] for i in range(len(model.blocks))}

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        logits, total_entropy, entropy_list = self.model(batch['input_ids'])
        B, T_out, V = logits.shape
        labels = batch['labels'][:, :T_out]
        xent = F.cross_entropy(logits.view(-1, V), labels.contiguous().view(-1))
        total_loss = xent + total_entropy
        self.train_losses.append(total_loss.item())
        self.train_xent.append(xent.item())
        self.log('train_loss', total_loss, prog_bar=True)
        self.log('train_xent', xent, prog_bar=True)
        for i, ent in enumerate(entropy_list):
            self.log(f'entropy_layer_{i}', ent, prog_bar=True)
            self.train_entropy_history[i].append(ent.item())
        return total_loss

    def validation_step(self, batch, batch_idx):
        logits, total_entropy, entropy_list = self.model(batch['input_ids'])
        B, T_out, V = logits.shape
        labels = batch['labels'][:, :T_out]
        xent = F.cross_entropy(logits.view(-1, V), labels.contiguous().view(-1))
        self.val_losses.append(xent.item())
        self.val_xent.append(xent.item())
        self.log('val_loss', xent, prog_bar=True)
        self.log('val_xent', xent, prog_bar=True)
        for i, ent in enumerate(entropy_list):
            self.log(f'val_entropy_layer_{i}', ent, prog_bar=True)
            self.val_entropy_history[i].append(ent.item())
        return xent

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

# -----------------------------
# Function for Plotting Training Curves and Per-Layer Entropy
# -----------------------------
def plot_training_curves(model, title_prefix="", smooth_window=20):
    """
    Plot training and validation curves with aligned x-axis.
    Shows both raw and smoothed data in the same color.
    """
    def smooth(x, window):
        if len(x) < window:
            return np.array(x)
        return np.convolve(x, np.ones(window) / window, mode='valid')

    train_xent = model.lit_module.train_xent
    val_xent = model.lit_module.val_xent

    n_train = len(train_xent)
    n_val = len(val_xent)

    val_x = np.linspace(0, n_train - 1, n_val)
    val_xent_interp = np.interp(np.arange(n_train), val_x, val_xent)

    plt.figure(figsize=(15, 5))
    
    # --- Loss Curves ---
    plt.subplot(1, 2, 1)

    x_raw = np.arange(n_train)
    x_smooth = np.arange(len(smooth(train_xent, smooth_window)))

    # Plot Train Xent
    color_train, = plt.plot(x_smooth, smooth(train_xent, smooth_window), label='Train')
    plt.plot(x_raw, train_xent, alpha=0.3, color=color_train.get_color(), label='_nolegend_')

    # Plot Val Xent
    color_val, = plt.plot(x_smooth, smooth(val_xent_interp, smooth_window), label='Val')
    plt.plot(x_raw, val_xent_interp, alpha=0.3, color=color_val.get_color(), label='_nolegend_')

    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('Loss (log scale)')
    plt.title(f'{title_prefix} Loss Curves (Aligned)')
    plt.legend()
    plt.grid(True)

    # --- Entropy Curves ---
    plt.subplot(1, 2, 2)
    for layer, history in model.lit_module.train_entropy_history.items():
        x_raw_ent = np.arange(len(history))
        smoothed = smooth(history, smooth_window)
        x_smooth_ent = np.arange(len(smoothed))
        line, = plt.plot(x_smooth_ent, smoothed, label=f'Layer {layer}')
        plt.plot(x_raw_ent, history, alpha=0.3, color=line.get_color(), label='_nolegend_')

    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('Entropy')
    plt.title(f'{title_prefix} Entropy per Layer')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()