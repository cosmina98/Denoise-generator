'''
This module wraps a pretrained **nucleotide language-model (DNABERT-style)** in a lightweight, scikit-learn–friendly estimator so you can go from raw DNA/RNA strings to task-specific predictions with minimal code.  The workflow is:

1. **Sequence understanding.**
   A Hugging Face Transformer (default ≈ 50 M parameters) converts each input sequence into a single CLS embedding that captures motifs, spacing and DNA shape.  The helper `_cls_batch` streams sequences in micro-batches so you never run out of GPU RAM.

2. **“Naturalness” scoring.**
   The same model’s masked-LM head (loaded lazily the first time it’s needed) is used to estimate a per-nucleotide probability; an optional Gaussian HMM segments low-probability stretches.  Those probabilities feed downstream tools (e.g. mutation bias, quality control) and are summarised by `likelihood()`.

3. **Task head – an MLP with early stopping.**
   On top of frozen (or optionally fine-tuned) CLS embeddings, an M-layer perceptron is trained to solve either classification or regression.  All key hyper-parameters—hidden sizes, dropout, activation, learning-rate, batch-size, patience—are exposed in `mlp_layers` / `mlp_params`.

4. **Memory-friendly masked inference.**
   When computing nucleotide probabilities the code masks only `batch_mask` tokens per forward pass, so even very long sequences can be analysed on a single GPU or, by toggling `mlm_on_cpu`, entirely on the CPU.

5. **Biological utilities.**
   `cut_points()` returns candidate crossover positions located in low-LM-confidence regions (useful for the GA), while `transform()` projects token probabilities back to nucleotides so any external algorithm can exploit base-level uncertainty.

In short, the class turns a giant self-supervised DNA model into a plug-and-play **feature extractor + slim predictor**, while exposing rich “how natural is this base?” signals that downstream optimisation algorithms (like the GA) can leverage to design or refine synthetic sequences.

'''


"""
NucleotideTransformerEstimator: A scikit-learn–friendly wrapper for pretrained nucleotide language models (DNABERT-style, Nucleotide Transformer, HyenaDNA, etc).

Parameters
----------
encoder_name : str, default "InstaDeepAI/nucleotide-transformer-50m-patch"
    HF repo id for the backbone model (any masked-LM trained on DNA/RNA).
mlm_name : str | None, default None
    Repo id of a compatible MLM head. If None, uses `encoder_name`.
revision : str | None, default None
    Branch / tag / commit to pin. Use None to grab the latest.
kmer : int | None, default None
    K-mer size for nucleotide probability estimation. If None,
    auto-deduced from env or defaults to 6.
device : str | None, default None
    Device for model inference: "cuda", "cpu", etc.  If None,
    will use "cuda" if available, else "cpu".
mlp_layers : tuple of int, default (128, 64)
    Hidden layer sizes for the MLP head.
mlp_params : dict, default None
    MLP training parameters including:
    - dropout: float, dropout rate
    - activation: str, "relu" or "gelu" 
    - lr: float, learning rate
    - batch_size: int
    - max_epochs: int
    - early_stopping: int, patience
conv_layers : tuple of int, default ()
    If non-empty, activates a 1-D convolutional neck (number of filters per layer).
conv_params : dict, default None
    Extra options for the ConvHead (kernel_size, stride, dropout,
    activation={"relu","gelu"}, global_pool={"max","mean"}).
finetune_bert : bool, default False
    Whether to fine-tune the encoder.
finetune_top_n_layers : int | None, default None
    If set, only the top N encoder layers are fine-tuned. If None, fine-tuning is determined by `finetune_bert`.
cls_batch_size : int, default 4
    Batch size for CLS embedding computation.
batch_mask : int, default 16
    Tokens masked *per forward call* when estimating nucleotide
    probs with the MLM head (controls GPU memory, not sequence length).
mlm_on_cpu : bool, default True
    If True, runs the MLM head on CPU to save GPU memory.
use_hmm : bool, default True
    If True, enables HMM-based segmentation for likelihood/cut_points.
verbose : bool, default False
    If True, enables logging of training progress.
max_len : int | None, default 1024
    Maximum sequence context length for the encoder and MLM head. Sequences longer than this will be truncated unless sliding-window support is added.
pooling : str, default "cls"
    Pooling strategy for sequence embeddings. One of {"cls", "mean", "max"}.
"""

import logging
import os
from typing import Sequence, Union, Tuple, List, Optional  # List & Optional for ≤3.9
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from transformers import (
    AutoModel,
    AutoModelForMaskedLM,
    AutoTokenizer,
    AutoConfig,
    # BertConfig,   # ← remove if unused
    EsmModel,
)
from hmmlearn import hmm

logger = logging.getLogger(__name__)

class MLPHead(nn.Module):
    """
    Multi-layer perceptron head for classification or regression.
    Args:
        in_dim: Input feature dimension.
        hidden: Tuple of hidden layer sizes.
        n_out: Output dimension (number of classes or 1 for regression).
        dropout: Dropout rate.
        activation: Activation function ('relu' or 'gelu').
    """
    def __init__(self, in_dim: int, hidden: Tuple[int, ...],
                 n_out: int, dropout: float = 0.1,
                 activation: str = "relu"):
        super().__init__()
        try:
            act = {"relu": nn.ReLU(), "gelu": nn.GELU()}[activation]
        except KeyError:
            raise ValueError("activation must be 'relu' or 'gelu'")
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), act, nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class ConvHead(nn.Module):
    """
    Stack of 1-D convolutions + global max/mean pool.
    Input  : [B, L, D]  (tokens × embedding_dim)
    Output : [B, F]     (fixed-length features)
    Args:
        in_channels: Input channels (embedding dim).
        filters: Tuple of output channels for each conv layer.
        kernel_size: Convolution kernel size.
        stride: Convolution stride.
        dropout: Dropout rate.
        activation: Activation function ('relu' or 'gelu').
        global_pool: Pooling type ('max' or 'mean').
    """
    def __init__(
        self,
        in_channels: int,
        filters: tuple[int, ...],
        *,
        kernel_size: int = 3,
        stride: int = 1,
        dropout: float = 0.1,
        activation: str = "relu",
        global_pool: str = "max",
    ):
        super().__init__()
        act = {"relu": nn.ReLU(), "gelu": nn.GELU()}[activation]
        layers, prev = [], in_channels
        for f in filters:
            layers += [
                nn.Conv1d(prev, f, kernel_size,
                          stride=stride, padding=kernel_size // 2),
                act,
                nn.Dropout(dropout),
            ]
            prev = f
        self.net  = nn.Sequential(*layers)
        self.pool = global_pool

    def forward(self, x):          # x: [B, L, D]
        x = x.permute(0, 2, 1)     # → [B, D, L]  (Conv1d wants C,L)
        x = self.net(x)            # → [B, F, L]
        return x.max(-1).values if self.pool == "max" else x.mean(-1)

# ────────────────────────────────────────────────────────────────────────────
#  Generic nucleotide-LM mix-in + estimator
# ────────────────────────────────────────────────────────────────────────────
class _NTMixin:
    """
    Mixin for nucleotide sequence modeling with transformer or CNN backbones.
    Handles HuggingFace model/tokenizer loading, per-nucleotide scoring, MLM head,
    HMM segmentation, and feature extraction. Used as a base for estimators.
    Args:
        encoder_name: HuggingFace repo id for backbone model.
        mlm_name: Repo id for compatible MLM head (or None to use encoder_name).
        revision: Model version/tag/commit.
        use_transformer: If True, use transformer backbone; else use CNN-only mode.
        kmer: K-mer size for MLM scoring (auto-detected if None).
        device: Device for computation ('cuda', 'cpu', etc.).
        mlp_layers: Tuple of MLP hidden layer sizes.
        mlp_params: Dict of MLP training hyperparameters.
        finetune_bert: Whether to fine-tune transformer encoder.
        finetune_top_n_layers: Fine-tune only top N transformer layers.
        cls_batch_size: Batch size for CLS embedding computation.
        batch_mask: Number of tokens masked per MLM forward pass.
        verbose: Enable verbose logging.
        mlm_on_cpu: Run MLM head on CPU to save GPU memory.
        max_len: Max sequence length for encoder/MLM.
        pooling: Pooling strategy for sequence embeddings ('cls', 'mean', 'max').
        use_hmm: Enable HMM-based segmentation for likelihood/cut_points.
    """
    def __init__(
        self,
        *,
        encoder_name: str = "InstaDeepAI/nucleotide-transformer-50m-patch",
        mlm_name: Optional[str] = None,
        revision: Optional[str] = None,
        use_transformer: bool = True,
        kmer: Optional[int] = None,
        device: Optional[str] = None,
        mlp_layers: Tuple[int, ...] = (128, 64),
        mlp_params: Optional[dict] = None,
        finetune_bert: bool = False,
        finetune_top_n_layers: int = None,
        cls_batch_size: int = 4,
        batch_mask: int = 16,
        verbose: bool = False,
        mlm_on_cpu: bool = True,
        max_len: Optional[int] = 1024,
        pooling: str = "cls",
        use_hmm: bool = True,
        **kwargs
    ):
        self.encoder_name = encoder_name
        self.mlm_name = mlm_name or encoder_name
        self.revision = revision
        self.use_hmm = use_hmm
        self.use_transformer = use_transformer
        self.verbose = verbose
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_mask = batch_mask
        self.mlp_layers = mlp_layers
        self.mlp_params = mlp_params or {"dropout": 0.1, "activation": "relu",
                                         "lr": 1e-4, "batch_size": 32,
                                         "max_epochs": 20, "early_stopping": 5}
        self.finetune_bert = finetune_bert
        self.cls_bs = cls_batch_size
        self.mlm_on_cpu = mlm_on_cpu
        self.max_len = max_len
        self.pooling = pooling.lower()
        self.finetune_top_n_layers = finetune_top_n_layers
        if not self.use_transformer:
            self.tokenizer = None
            self.encoder = None
            self._hf_opts = {}
            self.kmer = kmer or 6
            return
        _hf_opts = {"trust_remote_code": True}
        if revision is not None:
            _hf_opts["revision"] = revision
        self._hf_opts = _hf_opts
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_name, **_hf_opts)
        cfg = AutoConfig.from_pretrained(encoder_name, **_hf_opts)
        if cfg.model_type == "esm":
            self.encoder = EsmModel.from_pretrained(encoder_name, **_hf_opts).to(self.device)
        else:
            self.encoder = AutoModel.from_pretrained(encoder_name, **_hf_opts).to(self.device)
        resized = False
        cfg_max_pos = getattr(self.encoder.config, "max_position_embeddings", None)
        if (
            self.max_len
            and cfg_max_pos is not None
            and cfg_max_pos < self.max_len
            and hasattr(self.encoder, "resize_position_embeddings")
        ):
            try:
                self.encoder.resize_position_embeddings(self.max_len)
                resized = True
            except NotImplementedError:
                if self.verbose:
                    logger.warning(
                        "%s cannot resize positional embeddings; "
                        "longer sequences will be truncated to %d tokens.",
                        self.encoder_name,
                        cfg_max_pos,
                    )
        self._need_resize_mlm = self.max_len if resized else None
        if kmer is not None:
            self.kmer = kmer
        else:
            try:
                vocab = self.tokenizer.get_vocab()
                example_token = next(iter(vocab))
                if all(len(tok) == 1 for tok in vocab):
                    self.kmer = 1
                else:
                    self.kmer = len(example_token)
                if self.verbose:
                    logger.info(f"[Auto-detected] kmer = {self.kmer}")
            except Exception as e:
                if self.verbose:
                    logger.warning(f"Could not auto-detect kmer size: {e}")
                self.kmer = 6  # default fallback

    # ───────────────────────── helper ──────────────────────────
    def _chunk_iter(self, seq: str, win: int, stride: int) -> List[str]:
        "Yield overlapping windows (w/ CLS/BOS & SEP/EOS automatically handled)."
        if len(seq) <= win:
            yield seq
            return
        for s in range(0, len(seq), stride):
            chunk = seq[s : s + win]
            if len(chunk) < 10:                # safety guard
                break
            yield chunk

    def _cls_batch(self,
                   seq_batch: List[str],
                   grad: bool = False) -> torch.Tensor:
        win     = getattr(self.encoder.config, "max_position_embeddings", 512)
        stride  = win // 2
        outs    = []
        for seq in seq_batch:
            cls_vecs = []
            for chunk in self._chunk_iter(seq, win, stride):
                toks = self.tokenizer(
                    chunk,
                    padding=True,
                    truncation=True,
                    max_length=win,
                    return_tensors="pt",
                ).to(self.device)
                with torch.set_grad_enabled(grad):
                    out = self.encoder(**toks)
                    hidden = out.last_hidden_state     # shape: [B, L, D]
                    mask = toks["attention_mask"].unsqueeze(-1)   # shape: [B, L, 1]
                    if self.pooling == "mean":
                        h = (hidden * mask).sum(1) / mask.sum(1)
                    elif self.pooling == "max":
                        h = (hidden * mask).masked_fill(mask == 0, -1e9).max(1).values
                    else:  # "cls"
                        h = hidden[:, 0]
                cls_vecs.append(h if grad else h.detach())
            cls_vec = torch.mean(torch.stack(cls_vecs, 0), 0).squeeze(0)
            outs.append(cls_vec)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return torch.stack(outs, 0).to(torch.float32)

    # ------------------------------------------------------------
    # Helper: return last-hidden-state grid  [B, L, D]
    # ------------------------------------------------------------
    def _emb_batch(self, seq_batch: list[str], *, grad: bool = False):
        toks = self.tokenizer(
            seq_batch,
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        ).to(self.device)
        with torch.set_grad_enabled(grad):
            return self.encoder(**toks).last_hidden_state

    # ------------------------------------------------------------------
    #  (re-)add the lazy MLM loader that was accidentally removed
    # ------------------------------------------------------------------
    def _load_mlm(self):
        if not self.use_transformer:
            self.mlm = None
            self.mlm_device = self.device
            return

        """Lazy-load the masked-LM head the first time we need it."""
        if hasattr(self, "mlm"):                 # already loaded
            return

        vanilla_cfg = self.encoder.config       # keep hidden-size etc.
        tgt_device  = "cpu" if self.mlm_on_cpu else self.device
        try:
            self.mlm = AutoModelForMaskedLM.from_pretrained(
                self.mlm_name or self.encoder_name,
                **self._hf_opts,                    # carries revision / token / trust_remote_code
                config=vanilla_cfg,
                ignore_mismatched_sizes=True,       # allow ∣pos_emb∣ mismatch – we fix below
            ).to(tgt_device).eval()

            # resize MLM PE table if the encoder was resized
            if (
                self._need_resize_mlm
                and hasattr(self.mlm, "resize_position_embeddings")
                and self.mlm.config.max_position_embeddings < self._need_resize_mlm
            ):
                self.mlm.resize_position_embeddings(self._need_resize_mlm)

            self.mlm_device = tgt_device
            self.mask_id    = self.tokenizer.mask_token_id
        except (ValueError, OSError) as err:
            if self.verbose:
                logger.warning(
                    "Could not load MLM head for %s (%s). "
                    "Falling back to uniform nucleotide probabilities.",
                    self.encoder_name,
                    err,
                )
            self.mlm         = None          # marks “dummy” mode
            self.mlm_device  = self.device

    @torch.inference_mode()
    def _token_probs(self, seq: str) -> np.ndarray:
        # Sliding-window masked-LM overlong sequences
        self._load_mlm()          # ensure MLM is loaded
        # --------------------------------------------------------------
        # Fallback: if we couldn’t load an MLM head, return a uniform
        # probability (p = 0.25 per base for k-mer = 1, or
        # p = 1/|vocab| for k-mer > 1) so downstream code never crashes.
        # --------------------------------------------------------------
        if self.mlm is None:
            L = len(seq)
            p = 1.0 / (4 ** self.kmer)          # 0.25 for single-base
            return np.full(L, p, dtype=np.float32)
        win    = getattr(self.mlm.config, "max_position_embeddings", 512)
        stride = win // 2
        all_p  = []
        offset = 0
        for chunk in self._chunk_iter(seq, win, stride):
            ids = self.tokenizer(chunk, truncation=False,
                                 max_length=win,
                                 return_tensors="pt")["input_ids"][0]
            Lc   = len(ids)
            p_chunk = np.zeros(Lc, dtype=np.float32)
            for s in range(0, Lc, self.batch_mask):
                e = min(s + self.batch_mask, Lc)
                batch = ids.to(self.mlm_device).repeat(e - s, 1)
                for i, pos in enumerate(range(s, e)):
                    batch[i, pos] = self.mask_id
                with torch.no_grad():
                    logits = self.mlm(batch).logits
                for i, pos in enumerate(range(s, e)):
                    p = torch.softmax(logits[i, pos], -1)[ids[pos]].item()
                    p_chunk[pos] = max(p, 1e-12)
            all_p.append((offset, p_chunk))
            offset += stride
        # stitch overlapping chunks via simple averaging
        full = np.zeros(len(seq), dtype=np.float32)
        cnt  = np.zeros(len(seq), dtype=np.float32)
        for start, p in all_p:
            end = min(start + len(p), len(seq))
            seg = p[: end - start]
            full[start:end] += seg
            cnt[start:end]  += 1
        cnt[cnt == 0] = 1
        return full / cnt

    def transform(self, X: Sequence[str]) -> List[np.ndarray]:
        """
        Map each raw sequence to a per-nucleotide probability (mean-pooled
        over overlapping k-mers). Returns a list of float32 arrays.
        Only used in transformer mode.
        """
        out = []
        k = self.kmer
        for seq in X:
            tok_p = self._token_probs(seq)
            L = len(seq)
            nuc   = np.zeros(L, dtype=np.float32)
            count = np.zeros(L, dtype=np.float32)
            for i, p in enumerate(tok_p):
                # spread this k-mer’s probability across all k covered bases
                for j in range(k):               # k == self.kmer
                    pos = i + j
                    if pos < L:
                        nuc[pos]   += p
                        count[pos] += 1
            count[count == 0] = 1
            out.append(nuc / count)
        return out

    def _train_hmm(self, seqs: Sequence[str], n_iter: int = 150):
        """Train a 2-component GaussianHMM on -log10 nucleotide probs."""
        obs = np.concatenate([
            -np.log10(p + 1e-12)[:, None] for p in self.transform(seqs)
        ])
        self.hmm_ = hmm.GaussianHMM(
            n_components=2, covariance_type="diag",
            n_iter=n_iter, params="stmc", init_params="st"
        )
        self.hmm_.means_  = np.array([[0.1], [1.0]])
        self.hmm_.covars_ = np.array([[0.1], [0.3]])
        self.hmm_.fit(obs)

    # window: rolling-mean size (used only when self.use_hmm is False)
    def _states(self, probs: np.ndarray, window: int = 50) -> np.ndarray:
        if self.use_hmm:
            return self.hmm_.predict((-np.log10(probs + 1e-12)).reshape(-1, 1))

        # --- lightweight fallback: rolling mean over -log10 p ---
        logp   = -np.log10(probs + 1e-12)
        pad    = np.pad(logp, (window//2, window-1-window//2), mode="edge")
        smth   = np.convolve(pad, np.ones(window)/window, mode="valid")
        thresh = np.median(smth)       # or fixed value
        return (smth > thresh).astype(int)   # 0 = high-confidence, 1 = low

    def likelihood(self, X: Sequence[str]) -> np.ndarray:
        """
        Mean log10-probability per sequence (higher = more likely).
        Only used in transformer mode.
        """
        if self.use_transformer:
            return super().likelihood(X)
        self.mlp_.eval()
        if self.use_conv:
            self.conv_.eval()
        scores = []
        for seq in X:
            indices = np.array([self.nt_vocab.get(nt, 0) for nt in seq], dtype=np.int64)
            indices_tensor = torch.from_numpy(indices).unsqueeze(0).to(self.device)  # [1, L]
            input_embed = self.nt_embedding(indices_tensor).detach().requires_grad_(True)  # [1, L, D]
            feats = self.conv_(input_embed)
            logits = self.mlp_(feats)
            if self.task == "regression":
                score = logits.squeeze()
            else:
                target_class = logits.argmax(dim=-1).item()
                score = logits[:, target_class].squeeze()
            self.mlp_.zero_grad()
            if self.use_conv:
                self.conv_.zero_grad()
            score.backward()
            grad = input_embed.grad               # [1, L, D]
            saliency = grad.norm(dim=-1).squeeze(0).cpu().numpy()  # [L]
            pseudo_log_likelihood = -np.mean(np.log1p(1.0 / (saliency + 1e-6)))
            scores.append(pseudo_log_likelihood)
        return np.array(scores, dtype=np.float32)

    def cut_points(
        self,
        seq: str,
        min_len: int = 200,
        max_len: int = 400,
        low_state: str = "L",
    ) -> List[int]:
        """
        Return candidate crossover points in low-likelihood regions.
        Args:
            seq: Input sequence.
            min_len: Minimum segment length.
            max_len: Maximum segment length.
            low_state: State label for low-likelihood ('L' or 'H').
        Returns:
            List of cut positions.
        """
        # use window = average(min_len, max_len)
        win    = (min_len + max_len) // 2
        probs  = self.transform([seq])[0]
        states = self._states(probs, window=win)
        label  = {0: "H", 1: "L"}
        cuts, last = [], 0
        for i in range(1, len(seq)):
            if states[i] != states[i-1] and label[states[i-1]] == low_state:
                mid, seg_len = (i + last) // 2, (i + last) // 2 - last
                if seg_len >= min_len:
                    while seg_len > max_len:
                        mid = last + max_len
                        cuts.append(mid)
                        last = mid
                        seg_len = mid - last
                    cuts.append(mid)
                    last = mid
            if states[i] != states[i-1]:
                last = i
        if cuts and len(seq) - cuts[-1] < min_len:
            cuts.pop()
        return cuts

    # ------------------------------------------------------------------
    #  Sliding-window helpers (⬇ only used if caller passes `window=…`)
    # ------------------------------------------------------------------
    def _window_slices(self, L: int, window: int, stride: int):
        """Yield `(start, end)` indices for an overlapping sliding window."""
        if L <= window:
            yield 0, L
            return
        for s in range(0, L - window + 1, stride):
            yield s, s + window

class NucleotideTransformerEstimator(_NTMixin):
    """
    Scikit-learn–friendly estimator for DNA/RNA sequence tasks using either:
    - Transformer backbone (with MLM/HMM per-nucleotide scoring)
    - Pure CNN backbone (with saliency-based per-nucleotide scoring)
    Supports classification and regression, with optional convolutional neck.
    Args:
        task: 'classification' or 'regression'.
        encoder_name: HuggingFace repo id for backbone model.
        mlm_name: Repo id for compatible MLM head (or None to use encoder_name).
        revision: Model version/tag/commit.
        use_hmm: Enable HMM-based segmentation for likelihood/cut_points.
        use_transformer: If True, use transformer backbone; else use CNN-only mode.
        kmer: K-mer size for MLM scoring (auto-detected if None).
        device: Device for computation ('cuda', 'cpu', etc.).
        mlp_layers: Tuple of MLP hidden layer sizes.
        mlp_params: Dict of MLP training hyperparameters.
        finetune_bert: Whether to fine-tune transformer encoder.
        finetune_top_n_layers: Fine-tune only top N transformer layers.
        cls_batch_size: Batch size for CLS embedding computation.
        batch_mask: Number of tokens masked per MLM forward pass.
        verbose: Enable verbose logging.
        mlm_on_cpu: Run MLM head on CPU to save GPU memory.
        max_len: Max sequence length for encoder/MLM.
        pooling: Pooling strategy for sequence embeddings ('cls', 'mean', 'max').
        embedding_dim: Embedding dimension for CNN-only mode.
        conv_layers: Tuple of output channels for each conv layer.
        conv_params: Dict of extra ConvHead options (kernel_size, stride, etc.).
        kernel_size: Convolution kernel size (overrides conv_params if set).
    """
    def __init__(
        self,
        task: str = "classification",
        *,
        encoder_name: str = "InstaDeepAI/nucleotide-transformer-50m-patch",
        mlm_name: Optional[str] = None,
        revision: Optional[str] = None,
        use_hmm: bool = True,
        use_transformer: bool = True,
        kmer: Optional[int] = None,
        device: Optional[str] = None,
        mlp_layers: Tuple[int, ...] = (128, 64),
        mlp_params: Optional[dict] = None,
        finetune_bert: bool = False,
        finetune_top_n_layers: Optional[int] = None,
        cls_batch_size: int = 4,
        batch_mask: int = 16,
        verbose: bool = False,
        mlm_on_cpu: bool = True,
        max_len: Optional[int] = 1024,
        pooling: str = "cls",
        embedding_dim: int = 128,
        conv_layers: Tuple[int, ...] = (),
        conv_params: Optional[dict] = None,
        kernel_size: int = 5,
        **kwargs
    ):
        self.conv_layers = conv_layers  # Ensure always set before any use
        super().__init__(
            encoder_name=encoder_name,
            mlm_name=mlm_name,
            revision=revision,
            use_hmm=use_hmm,
            use_transformer=use_transformer,
            kmer=kmer,
            device=device,
            mlp_layers=mlp_layers,
            mlp_params=mlp_params,
            finetune_bert=finetune_bert,
            finetune_top_n_layers=finetune_top_n_layers,
            cls_batch_size=cls_batch_size,
            batch_mask=batch_mask,
            verbose=verbose,
            mlm_on_cpu=mlm_on_cpu,
            max_len=max_len,
            pooling=pooling,
        )
        self.task = task
        self.embedding_dim = embedding_dim
        conv_params = dict(conv_params or {})
        conv_params.setdefault("kernel_size", kernel_size)
        self.kernel_size = conv_params["kernel_size"]
        self.conv_params = {
            "kernel_size": self.kernel_size,
            "stride": 1,
            "dropout": 0.1,
            "activation": "relu",
            "global_pool": "max",
            **conv_params,
        }
        self.use_conv = bool(conv_layers)
        if not use_transformer:
            if not self.use_conv:
                raise ValueError("`conv_layers` must be non-empty when `use_transformer=False`")
            self.nt_vocab = {"A": 0, "C": 1, "G": 2, "T": 3}
            self.nt_embedding = nn.Embedding(4, embedding_dim).to(self.device)
        self.conv_ = None
    def _nt_emb_batch(self, seq_batch: list[str]):
        """
        Embed a batch of nucleotide sequences as one-hot indices for CNN-only mode.
        Args:
            seq_batch: List of DNA/RNA strings.
        Returns:
            embeddings: Tensor of shape [B, L, D].
            mask: Float mask of shape [B, L].
        """
        max_len = max(len(seq) for seq in seq_batch)
        indices = np.zeros((len(seq_batch), max_len), dtype=np.int64)
        mask = np.zeros((len(seq_batch), max_len), dtype=np.float32)
        for i, seq in enumerate(seq_batch):
            for j, nt in enumerate(seq):
                indices[i, j] = self.nt_vocab.get(nt, 0)
                mask[i, j] = 1
        indices_tensor = torch.from_numpy(indices).to(self.device)
        assert self.nt_embedding.weight.device == indices_tensor.device, \
            f"Device mismatch: embedding on {self.nt_embedding.weight.device}, input on {indices_tensor.device}"
        embeddings = self.nt_embedding(indices_tensor)
        return embeddings, mask
    def fit(self, X: Sequence[str], y: Sequence[Union[int, float]]):
        """
        Train the estimator on input sequences and labels.
        Handles both transformer and CNN-only modes, with optional fine-tuning.
        Args:
            X: List of DNA/RNA strings.
            y: List/array of labels (int for classification, float for regression).
        Returns:
            self
        """
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.1, stratify=y, random_state=0
        )
        if self.use_transformer:
            if self.finetune_top_n_layers is not None and self.finetune_bert:
                total_layers = 0
                for name, _ in self.encoder.named_parameters():
                    if "encoder.layer." in name:
                        num = int(name.split("encoder.layer.")[1].split(".")[0])
                        total_layers = max(total_layers, num)
                total_layers += 1
                if self.verbose:
                    print(f"Detected {total_layers} encoder layers.")
                start_layer = max(0, total_layers - self.finetune_top_n_layers)
                for name, param in self.encoder.named_parameters():
                    if "encoder.layer." in name:
                        layer_num = int(name.split("encoder.layer.")[1].split(".")[0])
                        param.requires_grad = (layer_num >= start_layer)
                    else:
                        param.requires_grad = False
                if self.verbose:
                    print(f"Fine-tuning only the top {self.finetune_top_n_layers} layers "
                          f"(layers {start_layer} to {total_layers-1})")
            self.encoder.train(self.finetune_bert)
        y_train = torch.tensor(y_train, dtype=torch.float32 if self.task == "regression" 
                        else torch.int64, device=self.device)
        bs = self.mlp_params["batch_size"]
        self.train_losses: List[float] = []
        self.val_losses:   List[float] = []
        if self.finetune_bert and self.use_transformer:
            dataset = list(zip(X_train, y_train))
            def _collate(samples):
                seqs, ys = zip(*samples)
                return list(seqs), torch.stack(ys)
            loader = DataLoader(dataset, batch_size=bs, shuffle=True,
                              collate_fn=_collate)
        else:
            with torch.no_grad():
                if self.use_conv:
                    if not self.use_transformer:
                        X_emb, _ = self._nt_emb_batch(list(X_train))
                        X_emb = X_emb.cpu()
                    else:
                        X_emb = self._emb_batch(list(X_train)).cpu()
                else:
                    X_emb = self._cls_batch(list(X_train)).cpu()
                dataset = TensorDataset(X_emb, y_train.cpu())
                loader = DataLoader(dataset, batch_size=bs, shuffle=True)
        if self.use_conv:
            if not self.use_transformer:
                D = self.embedding_dim
            else:
                if self.finetune_bert:
                    sample_emb = self._emb_batch([X_train[0]], grad=True)[0]
                else:
                    with torch.no_grad():
                        sample_emb = self._emb_batch([X_train[0]], grad=False)[0]
                D = sample_emb.shape[-1]
            self.conv_ = ConvHead(D, self.conv_layers, **self.conv_params).to(self.device)
            feature_dim = self.conv_layers[-1]
        else:
            sample_emb = self._cls_batch([X_train[0]], grad=self.finetune_bert)
            feature_dim = sample_emb.shape[1]
        n_out = 1 if self.task == "regression" else int(torch.max(y_train).item() + 1)
        self.mlp_ = MLPHead(feature_dim, self.mlp_layers, n_out,
                           dropout=self.mlp_params["dropout"],
                           activation=self.mlp_params["activation"]).to(self.device)
        params = list(self.mlp_.parameters())
        if self.use_conv:
            params += list(self.conv_.parameters())
        if self.finetune_bert and self.use_transformer:
            if self.verbose:
                logger.info("Fine-tuning DNABERT-2 encoder enabled")
            params += list(self.encoder.parameters())
        optim = AdamW(params, lr=self.mlp_params["lr"])
        criterion = nn.MSELoss() if self.task == "regression" else nn.CrossEntropyLoss()
        best_loss, wait = float("inf"), 0
        best_state = {k: v.cpu() for k, v in self.mlp_.state_dict().items()}
        for epoch in range(self.mlp_params["max_epochs"]):
            self.mlp_.train()
            epoch_loss = 0.0
            for batch in loader:
                if self.use_conv:
                    if self.finetune_bert and self.use_transformer:
                        seqs, yb = batch
                        yb = yb.to(self.device)
                        emb = self._emb_batch(list(seqs), grad=True)
                    else:
                        emb, yb = batch
                        emb = emb.to(self.device)
                    feats = self.conv_(emb)
                else:
                    if self.finetune_bert and self.use_transformer:
                        seqs, yb = batch
                        yb = yb.to(self.device)
                        feats = self._cls_batch(list(seqs), grad=True)
                    else:
                        cls, yb = batch
                        feats = cls.to(self.device)
                        yb = yb.to(self.device)
                logits = self.mlp_(feats)
                logits = logits.squeeze(-1) if self.task == "regression" else logits
                loss = criterion(logits, yb.to(self.device))  # Ensures same device
                optim.zero_grad()
                loss.backward()
                optim.step()
                epoch_loss += loss.item() * len(yb)
            epoch_loss /= len(loader.dataset)
            self.train_losses.append(epoch_loss)
            self.mlp_.eval()
            with torch.no_grad():
                val_feats = self._featurise(X_val, grad=False)
                yt = torch.tensor(y_val, dtype=torch.float32 if self.task == "regression" else torch.int64, device=self.device)
                v_logits = self.mlp_(val_feats)
                if self.task == "regression":
                    v_logits = v_logits.squeeze(-1)
                val_loss = criterion(v_logits, yt).item()
            self.val_losses.append(val_loss)
            if self.verbose:
                print(f"Epoch {epoch:03d}: train={epoch_loss:.4f}, val={val_loss:.4f}")
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                wait = 0
                best_state = {k: v.cpu() for k, v in self.mlp_.state_dict().items()}
            else:
                wait += 1
                if wait >= self.mlp_params["early_stopping"]:
                    if self.verbose:
                        logger.info("Early stopping at epoch %d", epoch)
                    break
        self.mlp_.load_state_dict(best_state)
        self.mlp_.eval()
        if self.use_hmm:
            self._train_hmm(X_train)
        if self.verbose and self.val_losses:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 5))
            plt.plot(self.train_losses, label="train loss")
            plt.plot(self.val_losses,   label="val   loss")
            plt.xlabel("epoch")
            plt.ylabel("loss")
            plt.yscale("log")  # Log scale for loss
            plt.legend()
            plt.tight_layout()
            plt.show()
        return self
    def _featurise(self, X: Sequence[str], *, grad: bool = False):
        """
        Extract features for downstream MLP from input sequences.
        Uses transformer CLS/embedding or CNN features as appropriate.
        Args:
            X: List of DNA/RNA strings.
            grad: If True, enables gradient tracking (for saliency).
        Returns:
            Feature tensor for MLP.
        """
        if not self.use_transformer:
            emb, _ = self._nt_emb_batch(list(X))
            return self.conv_(emb)
        if self.use_conv:
            emb = self._emb_batch(list(X), grad=grad)
            return self.conv_(emb)
        return self._cls_batch(list(X), grad=grad)
    def predict(self, X: Sequence[str], batch_size: int = 32):
        """
        Predict class labels or regression values for input sequences.
        Uses microbatching for memory efficiency.
        Args:
            X: List of DNA/RNA strings.
            batch_size: Microbatch size for inference.
        Returns:
            np.ndarray of predictions.
        """
        preds = []
        self.mlp_.eval()
        for i in range(0, len(X), batch_size):
            chunk = X[i : i + batch_size]
            with torch.no_grad():
                feats = self._featurise(chunk, grad=False)
                logits = self.mlp_(feats)
                if self.task == "regression":
                    preds.append(logits.squeeze(-1).cpu())
                else:
                    preds.append(logits.softmax(-1).argmax(-1).cpu())
        return torch.cat(preds).numpy()

    def predict_proba(self, X: Sequence[str], batch_size: int = 32):
        """
        Predict class probabilities for input sequences (classification only).
        Uses microbatching for memory efficiency.
        Args:
            X: List of DNA/RNA strings.
            batch_size: Microbatch size for inference.
        Returns:
            np.ndarray of probabilities.
        """
        if self.task != "classification":
            raise AttributeError("predict_proba is classification-only")
        probs = []
        self.mlp_.eval()
        for i in range(0, len(X), batch_size):
            chunk = X[i : i + batch_size]
            with torch.no_grad():
                feats = self._featurise(chunk, grad=False)
                probs.append(self.mlp_(feats).softmax(-1).cpu())
        return torch.cat(probs).numpy()
    def transform(self, X: Sequence[str], *a, **k) -> List[np.ndarray]:
        """
        Return per-nucleotide scores for each sequence:
        - If using transformer: masked-LM likelihood (via super().transform)
        - If using CNN-only: saliency scores via input gradients (∇output/∇embedding)
        Args:
            X: List of DNA/RNA strings.
        Returns:
            List of float arrays, one per sequence, with per-nucleotide scores.
        """
        if self.use_transformer:
            return super().transform(X, *a, **k)
        self.mlp_.eval()
        if self.use_conv:
            self.conv_.eval()
        out = []
        for seq in X:
            indices = np.array([self.nt_vocab.get(nt, 0) for nt in seq], dtype=np.int64)
            indices_tensor = torch.from_numpy(indices).unsqueeze(0).to(self.device)  # [1, L]
            input_embed = self.nt_embedding(indices_tensor).detach().requires_grad_(True)  # [1, L, D]
            feats = self.conv_(input_embed)       # [1, F]
            logits = self.mlp_(feats)             # [1, C] or [1]
            if self.task == "regression":
                score = logits.squeeze()
            else:
                target_class = logits.argmax(dim=-1).item()
                score = logits[:, target_class].squeeze()
            self.mlp_.zero_grad()
            if self.use_conv:
                self.conv_.zero_grad()
            score.backward()
            grad = input_embed.grad               # [1, L, D]
            saliency = grad.norm(dim=-1).squeeze(0).cpu().numpy()  # [L]
            out.append(saliency)
        return out
    def likelihood(self, X: Sequence[str]) -> np.ndarray:
        """
        Estimate how 'natural' each input sequence is.
        - Transformer: mean masked-LM log-probability (true likelihood)
        - CNN-only: mean saliency-derived pseudo-likelihood
        Args:
            X: List of DNA/RNA strings.
        Returns:
            np.ndarray of likelihood scores (higher = more natural/confident).
        """
        return super().likelihood(X)
    def cut_points(self, *a, **k):
        """
        Return candidate crossover points in low-likelihood regions.
        Delegates to mix-in implementation.
        """
        return super().cut_points(*a, **k)