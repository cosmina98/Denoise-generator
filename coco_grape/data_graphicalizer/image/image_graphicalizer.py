# Standard library
import warnings
import os
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional, Any, Union

# Third party imports
import numpy as np
import torch
import networkx as nx
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.spatial.distance import euclidean
from sklearn.base import BaseEstimator, TransformerMixin
from joblib import Parallel, delayed

# Deep learning frameworks
import clip
from transformers import pipeline
from segment_anything import sam_model_registry, SamPredictor
from ultralytics import YOLO

import os
import json

# Utility functions

def save_annotations(data: dict, filepath: str) -> None:
    """
    Save a dictionary of annotations to a JSON file.
    Creates parent directories if they don’t exist, and overwrites the file if it does.

    :param data: Dict mapping filenames to lists of tags.
    :param filepath: Path to the output JSON file.
    """
    # Ensure parent directory exists
    dirpath = os.path.dirname(filepath)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)

    # Write out JSON (mode 'w' will overwrite)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def load_annotations(filepath: str) -> dict:
    """
    Load a dictionary of annotations from a JSON file.

    :param filepath: Path to the JSON file containing annotations.
    :return: Dict mapping filenames to lists of tags.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)
    
def mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Compute IoU between two boolean masks.
    Masks must be the same shape.
    """
    inter = np.logical_and(mask1, mask2).sum()
    union = mask1.sum() + mask2.sum() - inter
    return float(inter / union) if union > 0 else 0.0


def mask_intersection(mask1: np.ndarray, mask2: np.ndarray) -> int:
    """Compute intersection pixel count between two boolean masks."""
    return int(np.logical_and(mask1, mask2).sum())


def filter_by_size(
    segments: List[Dict],
    image_size: Tuple[int, int],
    min_size: Optional[float] = None,
    max_size: Optional[float] = None,
    mask_key: str = 'mask'
) -> List[Dict]:
    """
    Remove segments outside size thresholds.
    - If min_size is not None:
      <=1: fraction of total image area; >1: absolute pixels.
    - If max_size is not None:
      <=1: fraction of total image area; >1: absolute pixels.
    """
    if min_size is None and max_size is None:
        return segments
    total_pixels = image_size[0] * image_size[1]
    kept: List[Dict] = []
    for seg in segments:
        mask = seg.get(mask_key)
        if mask is None:
            continue
        area = int(np.count_nonzero(mask))
        # check min
        if min_size is not None:
            if min_size <= 1:
                if area < min_size * total_pixels:
                    continue
            else:
                if area < min_size:
                    continue
        # check max
        if max_size is not None:
            if max_size <= 1:
                if area > max_size * total_pixels:
                    continue
            else:
                if area > max_size:
                    continue
        kept.append(seg)
    return kept


def filter_overlapping_by_iou(
    segments: List[Dict],
    iou_threshold: Optional[float],
    mask_key: str = 'mask',
    conf_key: str = 'score'
) -> List[Dict]:
    """
    Suppress segments whose overlap with a higher-confidence segment exceeds threshold.
    - If iou_threshold is None: no filtering.
    - If <=1: treat as IoU fraction; >1: treat as absolute intersection pixels.
    """
    if iou_threshold is None:
        return segments
    sorted_segs = sorted(segments, key=lambda x: x.get(conf_key, 0.0), reverse=True)
    keep: List[Dict] = []
    for seg in sorted_segs:
        mask = seg.get(mask_key)
        if mask is None:
            continue
        skip = False
        for kept_seg in keep:
            kept_mask = kept_seg.get(mask_key)
            if kept_mask is None:
                continue
            if iou_threshold <= 1:
                if mask_iou(mask, kept_mask) > iou_threshold:
                    skip = True
                    break
            else:
                if mask_intersection(mask, kept_mask) > iou_threshold:
                    skip = True
                    break
        if not skip:
            keep.append(seg)
    return keep

# 1) Base interface
class BaseSegmenter(ABC):
    @abstractmethod
    def segment(self, image_np: np.ndarray, **kwargs) -> List[Dict]:
        """
        returns list of segments, each having at least:
          - 'bbox': [x1,y1,x2,y2]
          - 'score': float
          - optional 'label', 'mask', 'crop', etc.
        """
        pass

# 2) Grounding DINO wrapper
class GroundingDinoSegmenter(BaseSegmenter):
    def __init__(self, model_name="IDEA-Research/grounding-dino-base", conf_threshold: float=0.3):
        from transformers import pipeline
        device = "cuda" if torch.cuda.is_available() else "cpu"
        device_idx = 0 if device.startswith("cuda") else -1
        self.zsod = pipeline(
            "zero-shot-object-detection",
            model=model_name,
            trust_remote_code=True,
            device=device_idx
        )
        self.threshold = conf_threshold

    def segment(self, image_np: np.ndarray, prompts: List[str], **kwargs) -> List[Dict]:
        if not prompts:
            raise ValueError("GroundingDinoSegmenter requires at least one prompt in `candidate_labels`")

        pil = Image.fromarray(image_np)
        outputs = self.zsod(pil, candidate_labels=prompts, threshold=self.threshold)
        return [{
            "bbox": [o["box"]["xmin"], o["box"]["ymin"], o["box"]["xmax"], o["box"]["ymax"]],
            "score": float(o["score"]),
            "label": o["label"],
        } for o in outputs]

# 3) YOLOv8 wrapper
class YoloSegmenter(BaseSegmenter):
    def __init__(
        self,
        model_path: str = 'yolov8n.pt',
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        verbose: bool = False  
    ):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.verbose = verbose # <-- Store the verbose setting

    def segment(self, image_np: np.ndarray, **kwargs) -> List[Dict]:
        # pass your thresholds AND verbose setting directly into the model call
        results = self.model(
            image_np,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=self.verbose
        )
        dets = []
        # Check if results[0].boxes is not None before iterating
        if results[0].boxes is not None:
            for box in results[0].boxes:
                dets.append({
                    "label":  self.model.names[int(box.cls)],
                    "score":  float(box.conf),
                    "bbox":   [int(x) for x in box.xyxy[0]]
                })
        return dets


# 4) (Optional) SAM refiner
class SamMaskRefiner(BaseSegmenter):
    def __init__(
        self,
        checkpoint: str = "models/sam/sam_vit_h_4b8939.pth",
        model_type: str = "vit_h",
        device: Optional[str] = None
    ):
        """
        Loads the SAM model and wraps it in a predictor.

        :param checkpoint: Path to SAM checkpoint file
        :param model_type: Type of SAM backbone (e.g. 'vit_h')
        :param device: Torch device string; auto‑detects if None
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[SAM] loading {model_type} checkpoint on {device}")
        sam = sam_model_registry[model_type](checkpoint=checkpoint)
        sam.to(device)
        self.sam_predictor = SamPredictor(sam)

    def segment(
        self,
        image_np: np.ndarray,
        proposals: List[Dict],
        **kwargs
    ) -> List[Dict]:
        """
        Refines bounding‑box proposals into pixel masks using SAM.

        :param image_np: H×W×3 uint8 array
        :param proposals: list of dicts, each with 'bbox' key
        :returns: list of dicts with added 'mask' and 'mask_score'
        """
        self.sam_predictor.set_image(image_np)
        out: List[Dict] = []
        for p in proposals:
            box = np.array(p["bbox"], dtype=float)[None, :]
            masks, scores, _ = self.sam_predictor.predict(
                box=box,
                multimask_output=False
            )
            out.append({
                **p,
                "mask": masks[0].astype(bool),
                "mask_score": float(scores[0])
            })
        return out


class SegmentationEstimator:
    def __init__(
        self,
        detector: BaseSegmenter,
        refiner: Optional[BaseSegmenter] = None,
        min_size: Optional[float] = None,
        max_size: Optional[float] = None,
        iou_threshold: Optional[float] = None
    ):
        self.detector = detector
        self.refiner = refiner
        self.min_size = min_size
        self.max_size = max_size
        self.iou_threshold = iou_threshold

    def transform(
        self,
        image_path: str,
        candidate_labels: Optional[List[str]] = None
    ) -> List[Dict]:
        # 1) load image
        img = np.array(Image.open(image_path).convert("RGB"))

        # 2) run detector, passing candidate_labels for DINO (or ignored by YOLO)
        segs = self.detector.segment(
            img,
            prompts=candidate_labels or []        # if using DINO
            # YOLOSegmenter will simply ignore unknown kwargs
        )

        # 3) optional SAM refinement
        if self.refiner:
            segs = self.refiner.segment(img, proposals=segs)

        # 4) size + overlap filters
        segs = filter_by_size(segs, image_size=img.shape[:2],
                              min_size=self.min_size, max_size=self.max_size)
        segs = filter_overlapping_by_iou(segs, self.iou_threshold)

        return img, segs


# Filters unchanged
def filter_by_size(
    segments: List[Dict],
    image_size: Tuple[int, int],
    min_size: Optional[float] = None,
    max_size: Optional[float] = None,
    mask_key: str = 'mask'
) -> List[Dict]:
    if min_size is None and max_size is None:
        return segments
    total_pixels = image_size[0] * image_size[1]
    kept: List[Dict] = []
    for seg in segments:
        area = None
        if mask_key in seg:
            area = int(np.count_nonzero(seg[mask_key]))
        else:
            x1, y1, x2, y2 = seg['bbox']
            area = (x2 - x1) * (y2 - y1)
        if min_size is not None:
            thr = min_size * total_pixels if min_size <= 1 else min_size
            if area < thr:
                continue
        if max_size is not None:
            thr = max_size * total_pixels if max_size <= 1 else max_size
            if area > thr:
                continue
        kept.append(seg)
    return kept

def filter_overlapping_by_iou(
    segments: List[Dict],
    iou_threshold: Optional[float],
    mask_key: str = 'mask',
    conf_key: str = 'score'
) -> List[Dict]:
    if iou_threshold is None:
        return segments

    def compute_iou(box1, box2):
        x1, y1, x2, y2 = box1
        x1p, y1p, x2p, y2p = box2
        xi1, yi1 = max(x1, x1p), max(y1, y1p)
        xi2, yi2 = min(x2, x2p), min(y2, y2p)
        inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        area1 = (x2 - x1) * (y2 - y1)
        area2 = (x2p - x1p) * (y2p - y1p)
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0.0

    segs = sorted(segments, key=lambda s: s.get(conf_key, 0.0), reverse=True)
    keep: List[Dict] = []
    for seg in segs:
        box = seg['bbox']
        if all(compute_iou(box, o['bbox']) <= iou_threshold for o in keep):
            keep.append(seg)
    return keep

# VisionText interface
class VisionTextModel(ABC):
    @abstractmethod
    def preprocess_image(self, img: Image.Image) -> torch.Tensor:
        pass

    @abstractmethod
    def encode_image(self, img_t: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        pass

# CLIP implementation
class ClipVisionTextModel(VisionTextModel):
    def __init__(self, model_name: str = "ViT-B/32", device: Optional[str] = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model, self.preprocess = clip.load(model_name, device=device)

    def preprocess_image(self, img: Image.Image) -> torch.Tensor:
        return self.preprocess(img).unsqueeze(0).to(self.device)

    def encode_image(self, img_t: torch.Tensor) -> torch.Tensor:
        feats = self.model.encode_image(img_t)
        return feats / feats.norm(dim=-1, keepdim=True)

    def encode_text(self, texts: List[str]) -> torch.Tensor:
        text_t = clip.tokenize(texts).to(self.device)
        feats = self.model.encode_text(text_t)
        return feats / feats.norm(dim=-1, keepdim=True)

# Simplified SemanticClassifier without captions\
class SemanticClassifier:
    def __init__(
        self,
        vision_model: VisionTextModel,
        base_labels: Optional[List[str]] = None
    ):
        self.vision_model = vision_model
        self.base_labels = base_labels or ["object"]

    def transform(
        self,
        image_np: np.ndarray,
        segments: List[Dict],
        candidate_labels: Optional[List[str]] = None,
        min_confidence_score: float = 0.1,
        max_size: Optional[float] = 0.25,
        overlap_threshold: Optional[float] = 0.85,
        bbox_key: str = 'bbox',
        conf_key: str = 'score',
        mask_key: str = 'mask'
    ) -> List[Dict]:
        # 1. Confidence filter
        filtered = [s for s in segments if s.get(conf_key, 0.0) >= min_confidence_score]
        if not filtered:
            return []

        # 2. Size & overlap filters
        filtered = filter_by_size(filtered, image_size=image_np.shape[:2], max_size=max_size)
        filtered = filter_overlapping_by_iou(filtered, iou_threshold=overlap_threshold)
        if not filtered:
            return []

        # 3. Prepare text candidates
        if candidate_labels:
            candidates = candidate_labels
        else:
            detected = [s.get('label') for s in filtered if s.get('label')]
            candidates = list(dict.fromkeys(detected)) or self.base_labels

        # 4. Encode text candidates
        txt_feats = self.vision_model.encode_text(candidates)

        enriched: List[Dict] = []
        pil_img = Image.fromarray(image_np)
        # 5. Process each segment
        for seg in filtered:
            # crop by mask or bbox
            if mask_key in seg:
                mask = seg[mask_key]
                masked_np = (image_np * mask[..., None]).astype(np.uint8)
                crop = Image.fromarray(masked_np)
            else:
                x1, y1, x2, y2 = seg[bbox_key]
                crop = pil_img.crop((x1, y1, x2, y2))

            # encode image crop
            img_t = self.vision_model.preprocess_image(crop)
            with torch.no_grad():
                img_feat = self.vision_model.encode_image(img_t)
                sims = (img_feat @ txt_feats.T).squeeze(0)
                best = sims.argmax().item()
                seg['semantic_label'] = candidates[best]
                seg['semantic_confidence'] = sims[best].item()

            enriched.append(seg)

        return enriched


def compute_bbox_centroid(bbox: Tuple[int,int,int,int]) -> Tuple[float, float]:
    x0, y0, x1, y1 = bbox
    return ((x0 + x1) / 2, (y0 + y1) / 2)


def compute_mask_centroid(mask: np.ndarray) -> Tuple[float, float]:
    ys, xs = np.nonzero(mask)
    return (xs.mean(), ys.mean())


def mask_extents(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    return xs.min(), ys.min(), xs.max(), ys.max()


def extract_geometric_relations_graph(
    segments: List[Dict[str, Any]],
    selected_labels: Optional[List[str]] = None,
    min_size: Optional[float] = None,
    max_size: Optional[float] = None,
    use_masks: bool = True,
    near_threshold: float = 0.05,
    overlap_area_threshold: float = 0.0,
    containment_area_threshold: float = 1.0,
    include_overlapping: bool = True,
    include_contained: bool = True,
    include_near: bool = True,
    include_left_of: bool = True,
    include_above: bool = True,
    n_iter: int = 1
) -> nx.MultiDiGraph:
    """
    Builds a scene graph (MST-based) with geometric relations among segments.

    Optionally filters segments by label, size thresholds (min_size/max_size),
    then computes pairwise relations and extracts up to `n_iter` MST layers.

    Nodes are keyed by the original segment index.
    """
    G = nx.MultiDiGraph()

    # 0) initial filter by selected labels
    indexed = list(enumerate(segments))
    if selected_labels is not None:
        sel = set(selected_labels)
        indexed = [(i, seg) for i, seg in indexed
                   if (seg.get('semantic_label') or seg.get('label')) in sel]
    if not indexed:
        return G

    # 1) estimate canvas size from first available mask; else from bbox
    h, w = None, None
    if use_masks:
        for _, seg in indexed:
            m = seg.get('mask')
            if m is not None:
                h, w = m.shape[:2]
                break

    # fallback to bbox if no mask found or not using masks
    if h is None or w is None:
        w = max(seg['bbox'][2] for _, seg in indexed) + 1
        h = max(seg['bbox'][3] for _, seg in indexed) + 1

    min_dim = min(w, h)
    abs_near = (near_threshold * min_dim
                if near_threshold <= 1
                else near_threshold)

    # 1b) filter by min/max size thresholds
    if min_size is not None or max_size is not None:
        img_area = w * h
        filtered = []
        for orig_i, seg in indexed:
            # compute segment area
            if use_masks and seg.get('mask') is not None:
                seg_area = int(np.sum(seg['mask']))
            else:
                x0, y0, x1, y1 = seg['bbox']
                seg_area = (x1 - x0 + 1) * (y1 - y0 + 1)
            # apply min_size
            if min_size is not None:
                if min_size <= 1:
                    if seg_area / img_area < min_size:
                        continue
                else:
                    if seg_area < min_size:
                        continue
            # apply max_size
            if max_size is not None:
                if max_size <= 1:
                    if seg_area / img_area > max_size:
                        continue
                else:
                    if seg_area > max_size:
                        continue
            filtered.append((orig_i, seg))
        indexed = filtered
        if not indexed:
            return G

    # 2) precompute geometry for each kept segment
    orig_idxs = [i for i, _ in indexed]
    geoms: List[Dict[str, Any]] = []
    for _, seg in indexed:
        if use_masks and seg.get('mask') is not None:
            m = seg['mask']
            area = int(m.sum())
            centroid = compute_mask_centroid(m)
            min_x, min_y, max_x, max_y = mask_extents(m)
        else:
            m = None
            x0, y0, x1, y1 = seg['bbox']
            area = (x1 - x0 + 1) * (y1 - y0 + 1)
            centroid = compute_bbox_centroid((x0, y0, x1, y1))
            min_x, min_y, max_x, max_y = x0, y0, x1, y1
        geoms.append({
            'mask': m,
            'area': area,
            'centroid': centroid,
            'min_x': min_x, 'min_y': min_y,
            'max_x': max_x, 'max_y': max_y
        })

    # 3) build relation tuples using internal indices
    relations_internal: List[Tuple[int, int, str]] = []
    n = len(geoms)
    for a_idx, ga in enumerate(geoms):
        for b_idx, gb in enumerate(geoms):
            if a_idx == b_idx:
                continue
            # overlap/containment calculation
            if include_overlapping or include_contained:
                if ga['mask'] is not None and gb['mask'] is not None:
                    ov = int(np.sum(ga['mask'] & gb['mask']))
                else:
                    xi0 = max(ga['min_x'], gb['min_x'])
                    yi0 = max(ga['min_y'], gb['min_y'])
                    xi1 = min(ga['max_x'], gb['max_x'])
                    yi1 = min(ga['max_y'], gb['max_y'])
                    ov = max(0, xi1 - xi0 + 1) * max(0, yi1 - yi0 + 1)
            if include_overlapping:
                thresh_ov = (overlap_area_threshold * ga['area']
                             if overlap_area_threshold <= 1 else overlap_area_threshold)
                if ov >= thresh_ov:
                    relations_internal.append((a_idx, b_idx, 'is_overlapping'))
            if include_contained:
                thresh_cont = (containment_area_threshold * ga['area']
                                if containment_area_threshold <= 1 else containment_area_threshold)
                if ov >= thresh_cont:
                    relations_internal.append((a_idx, b_idx, 'is_contained'))
            if include_near and euclidean(ga['centroid'], gb['centroid']) < abs_near:
                relations_internal.append((a_idx, b_idx, 'is_near'))
            if include_left_of and ga['max_x'] < gb['min_x']:
                relations_internal.append((a_idx, b_idx, 'is_left_of'))
            if include_above and ga['max_y'] < gb['min_y']:
                relations_internal.append((a_idx, b_idx, 'is_above'))

    # 4) build undirected weighted edges for MST
    base_edges: Dict[Tuple[int, int], float] = {}
    for a_idx, b_idx, _ in relations_internal:
        u, v = (a_idx, b_idx) if a_idx < b_idx else (b_idx, a_idx)
        w = euclidean(geoms[u]['centroid'], geoms[v]['centroid'])
        if (u, v) not in base_edges or w < base_edges[(u, v)]:
            base_edges[(u, v)] = w

    def kruskal(edges: Dict[Tuple[int, int], float]) -> List[Tuple[int, int]]:
        parent = list(range(n))
        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def unite(a: int, b: int):
            parent[find(a)] = find(b)
        mst = []
        for (u, v), _ in sorted(edges.items(), key=lambda kv: kv[1]):
            if find(u) != find(v):
                unite(u, v)
                mst.append((u, v))
                if len(mst) == n - 1:
                    break
        return mst

    # 5) extract up to n_iter MST layers
    remaining = dict(base_edges)
    layered = []
    for _ in range(n_iter):
        if not remaining:
            break
        layer = kruskal(remaining)
        if not layer:
            break
        for edge in layer:
            layered.append(edge)
            remaining.pop(edge, None)
    pair_set = {tuple(sorted(e)) for e in layered}

    # 6) add nodes keyed by original segment index
    for orig_i, seg in indexed:
        label = seg.get('semantic_label') or seg.get('label') or 'object'
        caption = seg.get('caption')
        bbox = seg.get('bbox')
        if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
            warnings.warn(f"Segment {orig_i} missing valid bbox, setting None")
            bbox = None
        pos = geoms[orig_idxs.index(orig_i)]['centroid']
        confidence = (
            seg.get('semantic_confidence') or seg.get('score') or seg.get('mask_score') or 0.0
        )
        G.add_node(orig_i,
                   label=label,
                   caption=caption,
                   bbox=bbox,
                   confidence=confidence,
                   pos=pos)

    # 7) add MST edges with original indices and relation labels
    for a_idx, b_idx, rel in relations_internal:
        u, v = orig_idxs[a_idx], orig_idxs[b_idx]
        if tuple(sorted((a_idx, b_idx))) in pair_set and G.has_node(u) and G.has_node(v):
            G.add_edge(u, v, relation=rel)

    return G


def visualize_scene_graph_on_image(
    image: np.ndarray,
    segments: List[Dict],
    G: nx.MultiDiGraph,
    show_image: bool = True,
    show_masks: bool = True,
    show_bbox: bool = True,
    show_graph: bool = True,
    show_objects: bool = True,
    alpha: float = 0.25,
    offset: float = 10.0,
    edge_text_color: str = 'w'
) -> None:
    """
    Visualize a scene graph overlaid on an image.

    Args:
        image: HxWx3 RGB image array.
        segments: List of dicts with keys 'mask' (HxW bool array, optional) and 'bbox' (x0, y0, x1, y1).
        G: NetworkX MultiDiGraph where nodes correspond to segment indices.
        show_image: If True, display the full image as the background; if False, skip full image.
        show_masks: Overlay instance masks.
        show_bbox: Draw bounding boxes.
        show_graph: Draw graph nodes and edges at bbox centers.
        show_objects: If True, restrict visible image regions to object masks (or bbox fallback) on white or overlaid.
        alpha: Mask overlay transparency.
        offset: Distance in pixels to offset edge labels from midpoint.
        edge_text_color: Color of edge label text.
    """
    H, W, _ = image.shape

    # Prepare object-only canvas (using mask or bbox if no mask)
    object_canvas = None
    if show_objects:
        object_canvas = np.ones_like(image, dtype=np.uint8) * 255
        for i, seg in enumerate(segments):  # <-- Fixed: proper enumeration
            if i not in G.nodes:
                continue

            # Try to get an existing mask; if missing or None, fall back to bbox
            mask = seg.get('mask', None)
            if mask is None:
                x0, y0, x1, y1 = seg['bbox']
                # ensure integer indices
                x0, y0, x1, y1 = map(int, (x0, y0, x1, y1))
                mask = np.zeros((H, W), dtype=bool)
                mask[y0:y1, x0:x1] = True

            object_canvas[mask] = image[mask]

    # Set up figure
    fig, ax = plt.subplots(figsize=(12, 12))

    # Determine base image to display and base array for overlays
    if show_image:
        if show_objects and not (show_masks or show_bbox):
            ax.imshow(object_canvas)
            base = object_canvas.astype(float)
        else:
            ax.imshow(image)
            base = image.astype(float)
    else:
        if show_objects and object_canvas is not None and not (show_masks or show_bbox):
            ax.imshow(object_canvas)
            base = object_canvas.astype(float)
        else:
            ax.set_facecolor('white')
            base = np.ones_like(image, dtype=float) * 255.0

    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)

    # Masks overlay
    if show_masks:
        overlay = base.copy()
        for i, seg in enumerate(segments):  # <-- Fixed: proper enumeration
            if i not in G.nodes:
                continue

            mask = seg.get('mask', None)
            if mask is None:
                x0, y0, x1, y1 = seg['bbox']
                x0, y0, x1, y1 = map(int, (x0, y0, x1, y1))
                mask = np.zeros((H, W), dtype=bool)
                mask[y0:y1, x0:x1] = True

            color_arr = (np.random.rand(3) * 255).astype(float)
            overlay[mask] = overlay[mask] * (1 - alpha) + color_arr * alpha

        ax.imshow(overlay.astype(np.uint8), extent=[0, W, H, 0])

    # Bounding boxes
    if show_bbox:
        for i, seg in enumerate(segments):  # <-- Fixed: proper enumeration
            if i not in G.nodes:
                continue
            x0, y0, x1, y1 = seg['bbox']
            rect = patches.Rectangle(
                (x0, y0), x1 - x0, y1 - y0,
                linewidth=2, edgecolor=np.random.rand(3,), facecolor='none', zorder=2
            )
            ax.add_patch(rect)

    # Scene graph overlay
    if show_graph:
        # positions at bbox centers
        pos: Dict[int, Tuple[float, float]] = {}
        for n in G.nodes:
            x0, y0, x1, y1 = segments[n]['bbox']
            pos[n] = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)

        # draw nodes
        for n, (x, y) in pos.items():
            node_color = np.random.rand(3,)
            ax.scatter(x, y, s=100, color=node_color, edgecolors='black', zorder=3)
            label = G.nodes[n].get('label', str(n))
            ax.text(x, y, label, fontsize=10,
                    ha='center', va='center_baseline',
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='black', alpha=0.8),
                    zorder=4)

        # draw edges and labels
        for u, v, data in G.edges(data=True):
            x0, y0 = pos[u]; x1, y1 = pos[v]
            # arrow outlines
            ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle='->', color='white', lw=2), zorder=2)
            ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle='->', color='black', lw=0.5), zorder=3)

            # label offset
            mx, my = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            perp = np.array([-(y1 - y0), (x1 - x0)])
            norm = np.linalg.norm(perp)
            if norm != 0:
                perp = perp / norm
            mx_off, my_off = mx + perp[0]*offset, my + perp[1]*offset
            angle = np.degrees(np.arctan2(y0 - y1, x1 - x0))
            rel = data.get('relation', '')
            ax.text(mx_off, my_off, rel, fontsize=8, color=edge_text_color,
                    ha='center', va='center_baseline',
                    rotation=angle, rotation_mode='anchor', zorder=4)

    ax.axis('off')
    plt.show()

class ImageToSceneGraphTransformer(BaseEstimator, TransformerMixin):
    """
    A scikit-learn compatible transformer that converts a list of images
    into a list of scene graphs (NetworkX graphs).

    It orchestrates segmentation, optional mask refinement, optional semantic
    classification, and geometric relation extraction.

    Parameters
    ----------
    detector : BaseSegmenter
        An instantiated object detector (e.g., GroundingDinoSegmenter, YoloSegmenter).
        Responsible for initial segment proposal (bbox, optionally label/score).

    refiner : Optional[SamMaskRefiner], default=None
        An instantiated SAM model to refine bounding boxes into masks.
        If None, mask-based operations in subsequent steps might fall back to
        bounding boxes.

    classifier : Optional[SemanticClassifier], default=None
        An instantiated semantic classifier (likely using a VisionTextModel like CLIP).
        Assigns 'semantic_label' and 'semantic_confidence' to segments.
        Note: This classifier might perform its own internal filtering.

    n_jobs : int, default=1
        Number of parallel jobs to run for transforming images. -1 means using
        all processors.

    Attributes
    ----------
    # Attributes are typically fitted parameters, but here we mostly store config.
    # No specific fitted attributes needed for this transformer.

    """
    def __init__(
        self,
        detector: Optional[BaseSegmenter] = None,
        refiner: Optional[SamMaskRefiner] = None,
        classifier: Optional[SemanticClassifier] = None,
        n_jobs: int = 1
    ):
        # Set up default components if not provided
        if detector is None:
            detector = YoloSegmenter(conf_threshold=0.05, iou_threshold=0.75)
        if classifier is None:
            classifier = SemanticClassifier(ClipVisionTextModel())
            
        self.detector = detector
        self.refiner = refiner
        self.classifier = classifier
        self.n_jobs = n_jobs

    def fit(self, X: List[np.ndarray], y: Optional[List[List[str]]] = None) -> "ImageToSceneGraphTransformer":
            """
            Fit method (does nothing, just returns self).

            Parameters
            ----------
            X : list of np.ndarray
                List of input images (HxWx3).
            y : list of list of str, optional
                List of corresponding label lists (prompts) for each image.

            Returns
            -------
            self
                The fitted transformer instance.
            """
            # No fitting required for this transformer
            return self

    def graphicalize(
        self, 
        graphs: Union[nx.MultiDiGraph, List[nx.MultiDiGraph]],
        **kwargs
    ) -> Union[nx.MultiDiGraph, List[nx.MultiDiGraph]]:
        """
        Extract geometric relations between segments and reconstruct graph(s).
        Works on either a single graph or list of graphs.
        """
        # Handle single graph
        if isinstance(graphs, nx.MultiDiGraph):
            if not (graphs.graph.get('image') is not None and graphs.graph.get('segments') is not None):
                return graphs
                
            graph = extract_geometric_relations_graph(
                segments=graphs.graph['segments'],
                **kwargs
            )
            
            # Preserve image, segments and filename
            graph.graph['image'] = graphs.graph['image']
            graph.graph['segments'] = graphs.graph['segments']
            if 'filename' in graphs.graph:
                graph.graph['filename'] = graphs.graph['filename']
            return graph
            
        # Handle list of graphs
        elif isinstance(graphs, list):
            return [self.graphicalize(G, **kwargs) for G in graphs]
            
        else:
            raise TypeError("Input must be a MultiDiGraph or list of MultiDiGraphs")

    def _process_single_image(
        self,
        image_np: np.ndarray,
        labels: Optional[List[str]],
        filename: Optional[str] = None,
        detector_kwargs: Optional[dict] = None,
        refiner_kwargs: Optional[dict] = None,
        classifier_kwargs: Optional[dict] = None,
        **graphicalize_kwargs
    ) -> nx.MultiDiGraph:
        """Helper function to process one image with component-specific kwargs."""
        try:
            current_segments = self.detector.segment(
                image_np, 
                prompts=labels or [], 
                **(detector_kwargs or {})
            )
        except Exception as e:
            print(f"Warning: Detector failed for an image. Error: {e}")
            G = nx.MultiDiGraph()
            G.graph['image'] = image_np.copy()
            G.graph['segments'] = []
            G.graph['filename'] = filename
            return G  # Return empty graph with image if detection fails

        if not current_segments:
            G = nx.MultiDiGraph()
            G.graph['image'] = image_np.copy()
            G.graph['segments'] = []
            G.graph['filename'] = filename
            return G  # Return empty graph with image if no segments

        # 2. Refine masks (optional)
        if self.refiner:
            try:
                current_segments = self.refiner.segment(
                    image_np, 
                    proposals=current_segments,
                    **(refiner_kwargs or {})
                )
            except Exception as e:
                print(f"Warning: Refiner failed for an image. Error: {e}")
                # Continue with unrefined segments if refinement fails

        # 3. Classify semantics (optional)
        if self.classifier:
            try:
                current_segments = self.classifier.transform(
                    image_np,
                    segments=current_segments,
                    candidate_labels=labels,
                    **(classifier_kwargs or {})
                )
            except Exception as e:
                print(f"Warning: Classifier failed for an image. Error: {e}")
                # Continue with unclassified segments if classification fails

        if not current_segments:
             G = nx.MultiDiGraph()
             G.graph['image'] = image_np.copy()
             G.graph['segments'] = []
             G.graph['filename'] = filename
             return G  # Return empty graph with image if segments filtered out

        # 4. Extract scene graph 
        try:
            G = nx.MultiDiGraph()
            G.graph['image'] = image_np.copy()
            G.graph['segments'] = current_segments
            G.graph['filename'] = filename
            graph = self.graphicalize(G, **graphicalize_kwargs)
            
        except Exception as e:
            print(f"Warning: Graph extraction failed for an image. Error: {e}")
            G = nx.MultiDiGraph()
            G.graph['image'] = image_np.copy()
            G.graph['segments'] = current_segments 
            G.graph['filename'] = filename
            return G

        return graph

    def transform(
        self, 
        X: List[np.ndarray], 
        y: Optional[List[List[str]]] = None, 
        filenames: Optional[List[str]] = None,
        detector_kwargs: Optional[dict] = None,
        refiner_kwargs: Optional[dict] = None,
        classifier_kwargs: Optional[dict] = None,
        **graphicalize_kwargs
    ) -> List[nx.MultiDiGraph]:
        """
        Transforms a list of images into a list of scene graphs.
        
        Parameters
        ----------
        X : List[np.ndarray]
            List of input images
        y : Optional[List[List[str]]]
            Optional labels/prompts for each image
        filenames : Optional[List[str]]
            Optional filenames for each image
        detector_kwargs : Optional[dict]
            Kwargs passed to detector.segment()
        refiner_kwargs : Optional[dict]
            Kwargs passed to refiner.segment()
        classifier_kwargs : Optional[dict]
            Kwargs passed to classifier.transform()
        graphicalize_kwargs : dict
            Kwargs passed to graphicalize() for geometric relation extraction
        """
        if not isinstance(X, list):
            raise TypeError("Input X must be a list of numpy arrays.")
        if y is not None and not isinstance(y, list):
             raise TypeError("Input y must be None or a list of lists of strings.")
        if y is not None and len(X) != len(y):
            raise ValueError("Inputs X and y must have the same length.")

        # Ensure kwargs are dicts or empty dicts
        detector_kwargs = detector_kwargs or {}
        refiner_kwargs = refiner_kwargs or {}
        classifier_kwargs = classifier_kwargs or {}

        names = filenames if filenames is not None else [None] * len(X)
        image_label_pairs = list(zip(X, y if y is not None else [None] * len(X), names))

        graphs = Parallel(n_jobs=self.n_jobs)(
            delayed(self._process_single_image)(
                image_np, labels, fname,
                detector_kwargs=detector_kwargs,
                refiner_kwargs=refiner_kwargs,
                classifier_kwargs=classifier_kwargs,
                **graphicalize_kwargs
            )
            for image_np, labels, fname in image_label_pairs
        )

        return graphs

    @staticmethod
    def load(directory: str, suffix: str = '.jpg', return_names: bool = False) -> Union[List[np.ndarray], Tuple[List[np.ndarray], List[str]]]:
        """
        Load all images with given suffix from directory in lexicographic order.
        
        Parameters
        ----------
        directory : str
            Path to directory containing images
        suffix : str, default='.jpg'
            File extension to filter images
        return_names : bool, default=False
            If True, also return list of filenames in same order as images

        Returns
        -------
        Union[List[np.ndarray], Tuple[List[np.ndarray], List[str]]]
            If return_names is False: List of loaded images as numpy arrays (HxWx3 RGB)
            If return_names is True: Tuple of (list of images, list of filenames)
        """
        # Get sorted list of image files
        image_files = sorted(
            f for f in os.listdir(directory) 
            if f.lower().endswith(suffix.lower())
        )
        
        if not image_files:
            warnings.warn(f"No images found with suffix {suffix} in {directory}")
            return ([], []) if return_names else []
            
        # Load each image
        images = []
        names = []
        for fname in image_files:
            try:
                fpath = os.path.join(directory, fname)
                img = np.array(Image.open(fpath).convert('RGB'))
                images.append(img)
                names.append(fname)
            except Exception as e:
                warnings.warn(f"Failed to load {fname}: {str(e)}")
                
        return (images, names) if return_names else images

    def display(self, graphs: List[nx.MultiDiGraph], verbose: bool = False, **kwargs) -> None:
        """
        Display multiple scene graphs using visualize_scene_graph_on_image.
        
        Parameters
        ----------
        graphs : List[nx.MultiDiGraph]
            List of graphs with 'image' and 'segments' attributes.
        verbose : bool, default=False
            If True, print additional information including filenames
        **kwargs : dict
            Additional arguments passed to visualize_scene_graph_on_image.
        """
        for i, G in enumerate(graphs):
            if not isinstance(G, nx.MultiDiGraph):
                print(f"Warning: Graph {i} is not a MultiDiGraph, skipping")
                continue
                
            # Get image and segments from graph attributes
            image = G.graph.get('image')
            segments = G.graph.get('segments')
            
            if image is None or segments is None:
                print(f"Warning: Graph {i} missing image or segments, skipping")
                continue
                
            if verbose:
                filename = G.graph.get('filename', 'unknown')
                print(f"\nDisplaying graph {i} from {filename}")
                print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
                
            visualize_scene_graph_on_image(image, segments, G, **kwargs)

