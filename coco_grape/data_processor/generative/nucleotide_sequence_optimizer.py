from __future__ import annotations
from typing import Sequence, List, Tuple, Optional
import numpy as np
import copy
import torch
from sklearn.base import BaseEstimator
import random
from deap import base, creator, tools, algorithms


import random
import numpy as np

def extract_best_salient_segment(seq, saliency, cutpoints, Lmin=20, Lmax=100):
    """
    Select the segment around cutpoints that has the highest cumulative saliency score.
    
    Parameters
    ----------
    seq : str
        Input DNA sequence.
    saliency : List[float]
        Per-base importance/uncertainty scores.
    cutpoints : List[int]
        Positions in the sequence flagged as candidate breakpoints.
    Lmin, Lmax : int
        Bounds for candidate segment lengths.
    
    Returns
    -------
    Tuple[int, str]
        (start_index, selected_segment)
    """
    best_score = -float('inf')
    best_start = 0
    best_len = Lmin
    
    for cp in cutpoints:
        for L in range(Lmin, min(Lmax + 1, len(seq))):
            start = max(0, min(len(seq) - L, cp - L // 2))
            score = sum(saliency[start:start + L])
            if score > best_score:
                best_score = score
                best_start = start
                best_len = L
                
    return best_start, seq[best_start:best_start + best_len]

def cx_salient_segment_swap(p1, p2, saliency1, saliency2, cutpoints1, cutpoints2,
                            Lmin=20, Lmax=100):
    """
    Cross over two sequences by extracting their most salient segments and swapping them,
    preserving length and ensuring child lengths lie strictly between parent lengths.
    
    Parameters
    ----------
    p1, p2 : list of str
        Parent sequences (lists of characters).
    saliency1, saliency2 : List[float]
        Saliency values for each parent.
    cutpoints1, cutpoints2 : List[int]
        Candidate salient cut locations.
    
    Returns
    -------
    Tuple[list[str], list[str]]
        Modified offspring (in-place if successful; otherwise unchanged).
    """
    s1 = ''.join(p1)
    s2 = ''.join(p2)
    len1, len2 = len(s1), len(s2)
    min_len, max_len = sorted([len1, len2])
    
    start1, seg1 = extract_best_salient_segment(s1, saliency1, cutpoints1, Lmin, Lmax)
    start2, seg2 = extract_best_salient_segment(s2, saliency2, cutpoints2, Lmin, Lmax)
    
    if len(seg1) != len(seg2):
        return p1, p2  # require same length segment for safe swap
    
    child1 = s1[:start1] + seg2 + s1[start1 + len(seg1):]
    child2 = s2[:start2] + seg1 + s2[start2 + len(seg2):]
    
    if not (min_len < len(child1) < max_len):
        return p1, p2
    if not (min_len < len(child2) < max_len):
        return p1, p2
    
    # In-place overwrite
    p1[:] = list(child1)
    p2[:] = list(child2)
    return p1, p2



'''
The two GA classes implement **evolutionary search that is guided, not blind**. The base `GASequenceOptimizer` treats sequence design as a bi-objective problem: it maximises the classifier's predicted activity **and** the language-model likelihood that the sequence still “looks natural.”  To make genetic operations respectful of biology, it (i) *crosses over* only at low-confidence regions flagged by the language model and (ii) *mutates* individual nucleotides with a probability proportional to the model's uncertainty at each position.  NSGA-II keeps a Pareto front so you later choose the activity/likelihood trade-off you like.  The `GASequenceOptimizerCR` extends this with a third objective—the **centroid-radius score**—which pulls every candidate toward the geometric centre (capturing consensus promoter grammar) while softly pushing it away from any one natural promoter (ensuring novelty).  Practically, that means each generation is steered simultaneously by (1) predicted functional strength, (2) plausibility as DNA, and (3) “looks like everybody, copies nobody,” giving you synthetic promoters that are broadly compatible yet not clones of the training data.
'''
# ---------------------------------------------------------------------
# GA optimiser
# ---------------------------------------------------------------------
class GASequenceOptimizer(BaseEstimator):
    """
    Parameters
    ----------
    estimator : 
        A *trained* classifier (task='classification') whose
        .predict_proba and .likelihood are the dual objectives.
    pop_size  : int
        Population size.
    ngen      : int
        Number of generations.
    cxpb, mutpb : float
        Crossover and mutation probabilities.
    mut_rate  : float
        Per-nucleotide mutation probability.
    """

    def __init__(
        self,
        estimator,
        pop_size: int = 200,
        ngen: int = 50,
        cxpb: float = 0.7,
        mutpb: float = 0.2,
        mut_rate: float = 0.01,
        min_seg: int = 200,
        max_seg: int = 400,
        verbose: bool = False,
    ):
        self.estimator = estimator
        self.pop_size  = pop_size
        self.ngen      = ngen
        self.cxpb      = cxpb
        self.mutpb     = mutpb
        self.mut_rate  = mut_rate
        self.min_seg   = min_seg
        self.max_seg   = max_seg
        self.alphabet  = ["A", "C", "G", "T"]   # ← restored helper list
        self.verbose = verbose

    # -----------------------------------------------------------------
    # scikit API
    # -----------------------------------------------------------------
    def fit(self, X: Sequence[str], y=None):
        """
        X : iterable of seed sequences (may be variable length)
        """
        # seeds may now be variable-length; no assertion needed

        # ---------- DEAP setup ----------
        # Register only one variable-length individual type
        if "FitnessMulti" not in creator.__dict__:
            creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0))
        if "IndividualVar" not in creator.__dict__:
            creator.create("IndividualVar", list, fitness=creator.FitnessMulti)

        toolbox = base.Toolbox()
        toolbox.register("clone", copy.deepcopy)
        toolbox.register("attr_base", random.choice, self.alphabet)

        # ---------- evaluation ----------
        def eval_ind(ind):
            seq = ''.join(ind)
            proba = self.estimator.predict_proba([seq])[0, 1]
            like  = self.estimator.likelihood([seq])[0]
            return proba, like

        # ---------- saliency-driven segment‐swap crossover ----------
        def mate_salient(p1, p2):
            s1, s2 = ''.join(p1), ''.join(p2)
            sal1 = self.estimator.transform([s1])[0]
            sal2 = self.estimator.transform([s2])[0]
            cp1  = self.estimator.cut_points(s1, method="threshold",
                                             percentile=85, min_gap=self.min_seg)
            cp2  = self.estimator.cut_points(s2, method="threshold",
                                             percentile=85, min_gap=self.min_seg)
            return cx_salient_segment_swap(
                p1, p2, sal1, sal2, cp1, cp2,
                Lmin=self.min_seg, Lmax=self.max_seg
            )
        toolbox.register("mate", mate_salient)

        # ---------- mutation identical to base GA (length-aware)
        def mutate(ind):
            """Length-aware point mutation identical to the base GA."""
            seq = ''.join(ind)
            probs = self.estimator.transform([seq])[0]
            for i in range(min(len(ind), len(probs))):
                if random.random() < self.mut_rate * (1.0 - probs[i]):
                    ind[i] = random.choice(self.alphabet)
            return (ind,)

        toolbox.register("mutate", mutate)
        toolbox.register("select", tools.selNSGA2)
        toolbox.register("evaluate", eval_ind)

        # ---------- build initial population (all IndividualVar) ----------
        pop = []
        for s in X:
            pop.append(creator.IndividualVar(list(s)))
        for _ in range(self.pop_size - len(X)):
            L = [toolbox.attr_base() for _ in range(random.randint(self.min_seg, self.max_seg))]
            pop.append(creator.IndividualVar(L))

        # ---------- run GA ----------
        pop, _ = algorithms.eaSimple(
            pop, toolbox,
            cxpb=self.cxpb,
            mutpb=self.mutpb,
            ngen=self.ngen,
            verbose=self.verbose
        )

        # keep non-dominated front
        self.best_sequences_ = [''.join(ind) for ind in tools.sortNondominated(pop, k=len(pop), first_front_only=True)[0]]
        return self

    # optional helper
    def predict(self, X: Sequence[str]) -> List[str]:
        """
        For each *query* sequence, return the GA-improved sequence on
        the Pareto front that has **highest proba** while being >= query likelihood.
        """
        best = []
        pros  = np.array([self.estimator.predict_proba([s])[0,1] for s in self.best_sequences_])
        likes = np.array([self.estimator.likelihood([s])[0]       for s in self.best_sequences_])
        for q in X:
            base_like = self.estimator.likelihood([q])[0]
            mask      = likes >= base_like
            idx       = np.argmax(pros * mask)
            best.append(self.best_sequences_[idx])
        return best

# ─────────────────────────────────────────────────────────────────────────────
#  GASequenceOptimizer  ⟶  GASequenceOptimizerCR   (C = centroid, R = radius)
# ─────────────────────────────────────────────────────────────────────────────
class GASequenceOptimizerCR(GASequenceOptimizer):
    """
    GA with 3 objectives:
        1. classifier proba  (max)
        2. centroid-radius score  (max)      ← NEW
        3. LM likelihood      (max)          (keep the old one)
    Soft version:  sim_centroid − β * logsumexp(k * sim_train)/k
    """
    # ------------------------- new parameters --------------------------------
    def __init__(
        self,
        estimator,
        train_seqs,
        beta: float = 1.0,            # weight of novelty penalty
        temp: float = 10.0,           # "k" in softmax ≈ temperature
        **kwargs                      # pop_size, ngen, etc. stay the same
    ):
        super().__init__(estimator, **kwargs)
        self.beta = beta
        self.temp = temp

        # -------- pre-compute unit-norm CLS embeddings & centroid ------------
        with torch.no_grad():
            E = estimator.get_sequence_embeddings(list(train_seqs)).cpu()  # (N, d)
            E = torch.nn.functional.normalize(E, p=2, dim=1)            # unit L2
        self.E_train = E                                                # cache
        c = torch.mean(E, dim=0, keepdim=True)                          # (1,d)
        self.centroid = torch.nn.functional.normalize(c, p=2, dim=1)    # unit

        # -------------- upgrade DEAP fitness from 2->3 objectives -----------
        if "FitnessTri" not in creator.__dict__:
            creator.create("FitnessTri", base.Fitness, weights=(1.0, 1.0, 1.0))
        # define a variable-length CR individual
        if "IndividualCRVar" not in creator.__dict__:
            creator.create("IndividualCRVar", list, fitness=creator.FitnessTri)

    def get_sequence_embeddings(self, seqs, grad: bool = False):
            return self.estimator.get_sequence_embeddings(seqs, grad=grad)

    # ------------------------ helper: CR score -------------------------------
    def _cr_score(self, emb: torch.Tensor) -> float:
        """
        emb : (1, d) already L2-normalised
        Return soft centroid-radius scalar (higher is better).
        """
        sim_centroid = torch.matmul(emb, self.centroid.T)               # (1,1)
        sims_train   = torch.matmul(emb, self.E_train.T)                # (1,N)

        # soft max   max_i sim ≈ (1/t) * logsumexp(t * sim_i)
        s = self.temp
        soft_max = (1.0 / s) * torch.logsumexp(s * sims_train, dim=1)   # (1,)
        return (sim_centroid - self.beta * soft_max).item()

    # -------------------------- main GA loop ---------------------------------
    def fit(self, X, y=None):
        # seeds are variable-length; no fixed reference length required

        # === rebuild individuals using the variable-length CR class ===========
        def make_ind(seq=None):
            if seq is None:
                # random length between min_seg/max_seg
                L = [random.choice(self.alphabet)
                     for _ in range(random.randint(self.min_seg, self.max_seg))]
            else:
                L = list(seq)
            return creator.IndividualCRVar(L)

        # build CR population with variable lengths
        pop = [make_ind(s) for s in X]
        for _ in range(self.pop_size - len(pop)):
            pop.append(make_ind())

        # ---------- evaluation with 3 objectives -----------------------------
        def eval_ind(ind):
            seq = ''.join(ind)

            # (i) classifier
            proba = self.estimator.predict_proba([seq])[0, 1]

            # (ii) centroid-radius
            with torch.no_grad():
                e = self.estimator.get_sequence_embeddings([seq]).cpu()
                e = torch.nn.functional.normalize(e, p=2, dim=1)
            cr = self._cr_score(e)

            # (iii) LM likelihood (reuse parent transform)
            like = self.estimator.likelihood([seq])[0]

            return proba, cr, like

        # ---------- saliency-driven segment‐swap crossover ----------
        def mate_salient(p1, p2):
            s1, s2 = ''.join(p1), ''.join(p2)
            sal1 = self.estimator.transform([s1])[0]
            sal2 = self.estimator.transform([s2])[0]
            cp1  = self.estimator.cut_points(s1, method="threshold",
                                             percentile=85, min_gap=self.min_seg)
            cp2  = self.estimator.cut_points(s2, method="threshold",
                                             percentile=85, min_gap=self.min_seg)
            return cx_salient_segment_swap(
                p1, p2, sal1, sal2, cp1, cp2,
                Lmin=self.min_seg, Lmax=self.max_seg
            )

        # ---------------- mutation (same logic as base GA) -------------------
        def mutate(ind):
            seq = ''.join(ind)
            probs = self.estimator.transform([seq])[0]
            for i in range(min(len(ind), len(probs))):
                if random.random() < self.mut_rate * (1.0 - probs[i]):
                    ind[i] = random.choice(self.alphabet)
            return (ind,)

        # register to toolbox (clone + shared ops)
        toolbox = base.Toolbox()
        toolbox.register("clone", copy.deepcopy)
        toolbox.register("evaluate", eval_ind)
        toolbox.register("mate", mate_salient)
        toolbox.register("mutate", mutate)
        toolbox.register("select", tools.selNSGA2)

        pop, _ = algorithms.eaSimple(
            pop, toolbox,
            cxpb=self.cxpb, mutpb=self.mutpb, ngen=self.ngen,
            verbose=self.verbose
        )

        # Pareto front saved just like before
        self.best_sequences_ = [
            ''.join(ind) for ind
            in tools.sortNondominated(pop, len(pop), first_front_only=True)[0]
        ]
        return self
    
    def predict_proba(self, seqs):
        return self.estimator.predict_proba(seqs)

    def likelihood(self, seqs):
        return self.estimator.likelihood(seqs)



def rank_solutions(
    sequences: Sequence[str],
    estimator,
    alpha: float = 0.7,
    raw_like: np.ndarray | None = None,   # ← new (optional) arg
) -> List[Tuple[str, float, float, float]]:
    """
    Rank candidate DNA sequences by a scalar score that blends
    classifier activity (proba) with *normalised* likelihood.

    Returns (seq, score, proba, norm_like), sorted by score desc.
    """
    # -- metrics -------------------------------------------------------
    probas = estimator.predict_proba(list(sequences))[:, 1]

    if raw_like is None:
        raw_like = estimator.likelihood(list(sequences))  # mean log10 P

    # robust min-max (add eps to avoid /0) -----------------------------
    mn, mx = raw_like.min(), raw_like.max()
    norm_like = (raw_like - mn) / (mx - mn + 1e-6)

    # scalar blend -----------------------------------------------------
    scores = alpha * probas + (1.0 - alpha) * norm_like

    return sorted(
        zip(sequences, scores, probas, norm_like),
        key=lambda t: t[1],
        reverse=True,
    )

def cr_rank_solutions(
    sequences: Sequence[str],
    estimator,
    weights: Tuple[float, float, float] = (0.5, 0.3, 0.2),
    raw_like: Optional[np.ndarray] = None,
    cr_scores: Optional[np.ndarray] = None,
) -> List[Tuple[str, float, float, float, float]]:
    """
    Blend three metrics into a scalar score and return a list sorted by it.

    Metrics (all *higher is better*):
    ---------------------------------
    1. proba      : classifier P(active promoter)
    2. norm_like  : language-model likelihood (normalised 0‒1)
    3. cr_score   : centroid-radius novelty/consensus score

    Parameters
    ----------
    sequences : iterable of str
    estimator : a fitted NucleotideTransformerEstimator *with*
                ._cls_batch(), .likelihood(), and attributes:
                  • centroid  (1 × d tensor, unit-norm)
                  • E_train   (N × d tensor, unit-norm)
                These exist automatically if you used GASequenceOptimizerCR.
    weights   : (α, β, γ) blending the three metrics.
    raw_like  : optional np.array with per-sequence mean log10 P.
    cr_scores : optional np.array with per-sequence centroid-radius scores.

    Returns
    -------
    List of tuples: (sequence, blended_score, proba, norm_like, cr_score),
    sorted descending by blended_score.
    """
    α, β, γ = weights
    seqs = list(sequences)

    # ------------------------------------------------------------------
    # (1) Classifier probabilities
    probas = estimator.predict_proba(seqs)[:, 1]          # shape (n,)

    # ------------------------------------------------------------------
    # (2) Language-model likelihood (if not supplied)
    if raw_like is None:
        raw_like = estimator.likelihood(seqs)             # (n,)

    # robust min-max normalisation
    mn, mx = raw_like.min(), raw_like.max()
    norm_like = (raw_like - mn) / (mx - mn + 1e-6)

    # ------------------------------------------------------------------
    # (3) Centroid-radius scores (soft version) ------------------------
    if cr_scores is None:
        with torch.no_grad():
            emb = estimator.get_sequence_embeddings(seqs).cpu()       # (n,d)
            emb = torch.nn.functional.normalize(emb, 2, 1)      # unit L2
            # similarities ------------------------------------------------
            sim_cent  = (emb @ estimator.centroid.T).squeeze(1) # (n,)
            sims_tr   = emb @ estimator.E_train.T               # (n,N)
            temp      = 10.0                                    # same "k" as GA
            soft_max  = (1 / temp) * torch.logsumexp(temp * sims_tr, dim=1)
            cr_scores = (sim_cent - estimator.beta * soft_max).numpy()

    # ------------------------------------------------------------------
    # blended scalar score ---------------------------------------------
    scores = α * probas + β * norm_like + γ * cr_scores

    # ------------------------------------------------------------------
    # sort & return -----------------------------------------------------
    ranked = sorted(
        zip(seqs, scores, probas, norm_like, cr_scores),
        key=lambda t: t[1],
        reverse=True,
    )
    return ranked