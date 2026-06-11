"""
dnabert2_xgb_estimator.py
-------------------------
Minimal, copy-pasteable wrapper for DNABERT-2 + XGBoost with built-in
HMM segmentation.

Public API
----------
• DNABERT2XGBEstimator(task="classification" | "regression", …)
      .fit(X, y)           - train on CLS embeddings
      .predict(X)          - class labels or regression values
      .predict_proba(X)    - only if task == "classification"

Extra helpers (exposed for GA integration)
      .likelihood(X, reduction="mean_log_prob" | "geom_mean" | "perplexity")
      .cut_points(seq, min_len=200, max_len=400, low_state="L")
"""

from __future__ import annotations
from typing import Sequence, List, Tuple, Union
import numpy as np, torch, xgboost as xgb
import logging
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM, BertConfig
from sklearn.base import BaseEstimator
from hmmlearn import hmm
import random
from deap import base, creator, tools, algorithms

# ----------------------------------------------------------------------
# logging helper
# ----------------------------------------------------------------------
logger = logging.getLogger("dnabert2xgb")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.WARNING)          # raised to INFO when verbose=True


# ────────────────────────────────────────────────────────────────────────
# Mixin: DNABERT helpers (CLS extraction, nucleotide probs, global HMM)
# ────────────────────────────────────────────────────────────────────────
class _DNABERTMixin(BaseEstimator):
    def __init__(
        self,
        model_name: str = "zhihan1996/DNABERT-2-117M",
        revision: str = "6617c7e",
        kmer: int = 6,
        device: str | None = None,
        xgb_params: dict | None = None,
        batch_mask: int = 16,
        verbose: bool = False,
    ):
        self.model_name = model_name
        self.kmer = kmer
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.batch_mask = batch_mask
        self.xgb_params = xgb_params or {"n_estimators": 200, "max_depth": 5}
        self.revision = revision
        self.verbose = verbose

        if self.verbose:
            logger.setLevel(logging.INFO)

        # load HF models now
        self._load_models()          # ←—— restores self.tokenizer / encoder / mlm

    def _load_models(self):
        if self.verbose:
            logger.info("Loading DNABERT-2 (%s, rev=%s) …", self.model_name, self.revision)

        # ───────── 1) tokenizer ─────────
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            revision=self.revision,
            trust_remote_code=True,
        )

        # ───────── 2) encoder — repo config ─────────
        self.encoder = AutoModel.from_pretrained(
            self.model_name,
            revision=self.revision,
            trust_remote_code=True,
        ).to(self.device).eval()
        # config.return_dict=True removed (doesn't work with MosaicBERT)

        # repo_cfg *now* exists → clone it to a vanilla BertConfig
        repo_cfg     = self.encoder.config
        vanilla_cfg  = BertConfig.from_dict(repo_cfg.to_dict())

        self.mlm = AutoModelForMaskedLM.from_pretrained(
            self.model_name,
            revision=self.revision,
            trust_remote_code=True,
            config=vanilla_cfg,
        ).to(self.device).eval()
        # config.return_dict=True removed here too

        self.mask_id = self.tokenizer.mask_token_id
        if self.verbose:
            logger.info("Finished loading models (mask_id=%d)", self.mask_id)

    # ---------- CLS embeddings ----------
    @torch.inference_mode()
    def _cls_batch(self, seq_batch: List[str]) -> np.ndarray:
        if self.verbose:
            logger.debug("Encoding %d sequences → CLS", len(seq_batch))

        toks = self.tokenizer(
            seq_batch,
            padding=True,
            truncation=True,
            max_length=4096,            # silence warning
            return_tensors="pt",
        ).to(self.device)
        outs = self.encoder(**toks)          # MosaicBERT always -> tuple
        hidden = outs.last_hidden_state if hasattr(outs, "last_hidden_state") else outs[0]
        h = hidden[:, 0, :]
        return h.cpu().numpy()

    # ---------- per-token probability ----------
    @torch.inference_mode()
    def _token_probs(self, seq: str) -> np.ndarray:
        ids = self.tokenizer(
            seq,
            truncation=False,          # keep full sequence
            max_length=4096,           # length guard (silences warning)
            return_tensors="pt",
        )["input_ids"][0]
        L   = len(ids)
        probs = np.zeros(L)
        for s in range(0, L, self.batch_mask):
            e = min(s + self.batch_mask, L)
            batch = ids.repeat(e - s, 1)
            for i, pos in enumerate(range(s, e)):
                batch[i, pos] = self.mask_id
            # Same tuple/dict duality for the MLM head
            out    = self.mlm(batch.to(self.device))
            logits = out.logits if hasattr(out, "logits") else out[0]
            for i, pos in enumerate(range(s, e)):
                p = torch.softmax(logits[i, pos], dim=-1)[ids[pos]].item()
                probs[pos] = p
        return probs

    # ---------- nucleotide-level probs ----------
    def transform(self, X: Sequence[str]) -> List[np.ndarray]:
        out = []
        for seq in X:
            tok = self._token_probs(seq)                       # per-token
            # assign central nucleotide of each k-mer the token prob
            L, k = len(seq), self.kmer
            nuc = np.zeros(L); count = np.zeros(L)
            for i in range(len(tok)):
                centre = i + k // 2
                if centre < L:
                    nuc[centre]  += tok[i]
                    count[centre] += 1
            count[count == 0] = 1
            out.append((nuc / count).astype(np.float32))
        return out

    # ---------- global 2-state HMM ----------
    def _train_hmm(self, seqs: Sequence[str], n_iter=150):
        obs = np.concatenate([ -np.log10(self.transform([s])[0] + 1e-12) for s in seqs ])[:, None]
        self.hmm_ = hmm.GaussianHMM(n_components=2, covariance_type="diag", n_iter=n_iter, params="stmc", init_params="stmc")
        self.hmm_.means_  = np.array([[0.1], [1.0]])
        self.hmm_.covars_ = np.array([[0.1], [0.3]])
        self.hmm_.fit(obs)

    def _states(self, probs: np.ndarray) -> np.ndarray:
        return self.hmm_.predict((-np.log10(probs + 1e-12)).reshape(-1, 1))


# ────────────────────────────────────────────────────────────────────────
# Main estimator: selectable regression / classification
# ────────────────────────────────────────────────────────────────────────
class DNABERT2XGBEstimator(_DNABERTMixin):
    def __init__(self, task: str = "classification", verbose: bool = False, **kwargs):
        assert task in {"classification", "regression"}
        self.task = task
        super().__init__(verbose=verbose, **kwargs)

    # --------------- sklearn core ----------------
    def fit(self, X: Sequence[str], y: Sequence[Union[int, float]]):
        X_emb = self._cls_batch(list(X))
        if self.task == "classification":
            self.model_ = xgb.XGBClassifier(**self.xgb_params)
        else:
            self.model_ = xgb.XGBRegressor(**self.xgb_params)
        self.model_.fit(X_emb, y)
        self._train_hmm(X)                           # train HMM once
        return self

    def predict(self, X: Sequence[str]):
        return self.model_.predict(self._cls_batch(list(X)))

    def predict_proba(self, X: Sequence[str]):
        if self.task != "classification":
            raise AttributeError("predict_proba is only available in classification mode.")
        return self.model_.predict_proba(self._cls_batch(list(X)))

    # --------------- GA helpers ------------------
    def likelihood(self, X: Sequence[str], reduction: str = "mean_log_prob") -> np.ndarray:
        scores = []
        for probs in self.transform(X):
            lp = np.log10(probs + 1e-12).mean()
            if reduction == "mean_log_prob":
                scores.append(lp)
            elif reduction == "geom_mean":
                scores.append(10 ** lp)
            elif reduction == "perplexity":
                scores.append(10 ** (-lp))
            else:
                raise ValueError("unknown reduction")
        return np.asarray(scores, dtype=np.float32)

    def cut_points(
        self,
        seq: str,
        min_len: int = 200,
        max_len: int = 400,
        low_state: str = "L",
    ) -> List[int]:
        probs  = self.transform([seq])[0]
        states = self._states(probs)
        label  = {0: "H", 1: "L"}                    # state → label
        cuts, last = [], 0
        for i in range(1, len(seq)):
            if states[i] != states[i-1] and label[states[i-1]] == low_state:
                mid = (i + last) // 2
                seg_len = mid - last
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
        # drop final cut if tail too small
        if cuts and len(seq) - cuts[-1] < min_len:
            cuts.pop()
        return cuts
