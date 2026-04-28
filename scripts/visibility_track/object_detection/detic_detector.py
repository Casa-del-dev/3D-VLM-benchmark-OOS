from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np

from .detection_types import DetectionResult


class DeticDetector:
    """
    Thin wrapper around FIction-Detic / facebookresearch Detic.

    Expected repo structure:
        detic_root/
          demo.py
          configs/
          detic/
          third_party/CenterNet2/
          datasets/metadata/

    This wrapper returns the same DetectionResult format as your Grounding DINO
    code, so it can be used by ROIVisibilityEstimator.
    """

    BUILTIN_CLASSIFIER = {
        "lvis": "datasets/metadata/lvis_v1_clip_a+cname.npy",
        "objects365": "datasets/metadata/o365_clip_a+cnamefix.npy",
        "openimages": "datasets/metadata/oid_clip_a+cname.npy",
        "coco": "datasets/metadata/coco_clip_a+cname.npy",
    }

    BUILTIN_METADATA = {
        "lvis": "lvis_v1_val",
        "objects365": "objects365_v2_val",
        "openimages": "oid_val_expanded",
        "coco": "coco_2017_val",
    }

    def __init__(
        self,
        detic_root: Union[str, Path],
        config_file: Union[str, Path],
        weights: Union[str, Path],
        vocabulary: str = "custom",
        custom_vocabulary: Optional[Sequence[str]] = None,
        confidence_threshold: float = 0.25,
        device: str = "cuda",
        one_class_per_proposal: bool = True,
    ) -> None:
        self.detic_root = Path(detic_root).expanduser().resolve()
        self.config_file = self._resolve_path(config_file)
        self.weights = self._resolve_path(weights)
        self.vocabulary = vocabulary
        self.custom_vocabulary = list(custom_vocabulary or [])
        self.confidence_threshold = float(confidence_threshold)
        self.device = device
        self.one_class_per_proposal = bool(one_class_per_proposal)

        if not self.detic_root.exists():
            raise FileNotFoundError(f"detic_root does not exist: {self.detic_root}")
        if not self.config_file.exists():
            raise FileNotFoundError(f"Detic config file does not exist: {self.config_file}")
        if not self.weights.exists():
            raise FileNotFoundError(f"Detic weights file does not exist: {self.weights}")

        self._add_detic_to_pythonpath()
        self._lazy_imports()
        self.predictor = self._build_predictor()
        self.metadata = None
        self.class_names: List[str] = []

        # Initialize classifier once. For ROI testing this is usually enough.
        self.set_vocabulary(vocabulary=self.vocabulary, custom_vocabulary=self.custom_vocabulary)

    def _resolve_path(self, path_like: Union[str, Path]) -> Path:
        p = Path(path_like).expanduser()
        if p.is_absolute():
            return p.resolve()
        return (self.detic_root / p).resolve()

    def _add_detic_to_pythonpath(self) -> None:
        """
        Add FIction-Detic and CenterNet2 to Python path.

        Your FIction-Detic checkout has CenterNet2 here:
            third_party/CenterNet2/centernet

        Some other Detic checkouts have it here:
            third_party/CenterNet2/projects/CenterNet2/centernet

        We support both layouts.
        """
        candidate_paths = [
            self.detic_root,
            self.detic_root / "third_party" / "CenterNet2",
            self.detic_root / "third_party" / "CenterNet2" / "projects" / "CenterNet2",
        ]

        for path in candidate_paths:
            path = path.resolve()
            if path.exists():
                path_str = str(path)
                if path_str not in sys.path:
                    sys.path.insert(0, path_str)

        possible_centernet_pkgs = [
            self.detic_root / "third_party" / "CenterNet2" / "centernet",
            self.detic_root / "third_party" / "CenterNet2" / "projects" / "CenterNet2" / "centernet",
        ]

        if not any(p.exists() for p in possible_centernet_pkgs):
            raise FileNotFoundError(
                "Could not find the CenterNet2 centernet package. Tried:\n"
                + "\n".join(f"  {p}" for p in possible_centernet_pkgs)
                + "\n\nFrom inside FIction-Detic, run:\n"
                "  git submodule update --init --recursive\n"
            )

    def _lazy_imports(self) -> None:
        try:
            from detectron2.config import get_cfg
            from detectron2.engine import DefaultPredictor
            from detectron2.data import MetadataCatalog
            from centernet.config import add_centernet_config
            from detic.config import add_detic_config
            from detic.modeling.utils import reset_cls_test
            from detic.modeling.text.text_encoder import build_text_encoder
        except Exception as e:
            raise ImportError(
                "Failed to import Detic/Detectron2 dependencies. "
                "Install FIction-Detic first, preferably in Linux/WSL/Colab with CUDA. "
                f"Original error: {e!r}"
            ) from e

        self.get_cfg = get_cfg
        self.DefaultPredictor = DefaultPredictor
        self.MetadataCatalog = MetadataCatalog
        self.add_centernet_config = add_centernet_config
        self.add_detic_config = add_detic_config
        self.reset_cls_test = reset_cls_test
        self.build_text_encoder = build_text_encoder

    def _build_predictor(self):
        cfg = self.get_cfg()
        cfg.MODEL.DEVICE = self.device

        self.add_centernet_config(cfg)
        self.add_detic_config(cfg)

        # Detic configs contain relative paths such as:
        # datasets/metadata/lvis_v1_train_cat_info.json
        # These are relative to the Detic repo, not your benchmark repo.
        old_cwd = os.getcwd()
        try:
            os.chdir(str(self.detic_root))

            cfg.merge_from_file(str(self.config_file))
            cfg.MODEL.WEIGHTS = str(self.weights)

            # These are the same threshold fields used by the Detic demo.
            cfg.MODEL.RETINANET.SCORE_THRESH_TEST = self.confidence_threshold
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.confidence_threshold
            cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = self.confidence_threshold

            # Important for CLIP/custom vocabulary mode. The actual classifier is reset later.
            cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = "rand"

            if self.one_class_per_proposal:
                cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True

            cfg.freeze()
            predictor = self.DefaultPredictor(cfg)

        finally:
            os.chdir(old_cwd)

        return predictor

    def _get_clip_embeddings(self, vocabulary: Sequence[str], prompt: str = "a "):
        text_encoder = self.build_text_encoder(pretrain=True)
        text_encoder.eval()
        texts = [prompt + x for x in vocabulary]
        emb = text_encoder(texts).detach().permute(1, 0).contiguous().cpu()
        return emb

    @staticmethod
    def _normalize_custom_vocabulary(
        text_prompt: Union[str, Sequence[str], None],
        fallback: Sequence[str],
    ) -> List[str]:
        if text_prompt is None:
            items = list(fallback)
        elif isinstance(text_prompt, str):
            # Allow "mug,cup" or "mug. cup." style input.
            raw = text_prompt.replace(".", ",").split(",")
            items = [x.strip() for x in raw]
        else:
            items = [str(x).strip() for x in text_prompt]

        # Detic custom vocabulary expects class names, not long phrases if possible.
        cleaned = []
        for item in items:
            item = item.strip().lower().rstrip(".")
            if item and item not in cleaned:
                cleaned.append(item)
        if not cleaned:
            raise ValueError("custom vocabulary is empty.")
        return cleaned

    def set_vocabulary(
        self,
        vocabulary: str = "custom",
        custom_vocabulary: Optional[Sequence[str]] = None,
    ) -> None:
        if vocabulary not in {"lvis", "objects365", "openimages", "coco", "custom"}:
            raise ValueError(f"Unsupported Detic vocabulary: {vocabulary}")

        self.vocabulary = vocabulary

        if vocabulary == "custom":
            names = self._normalize_custom_vocabulary(None, custom_vocabulary or self.custom_vocabulary)
            self.metadata = self.MetadataCatalog.get(f"custom_detic_{time.time()}")
            self.metadata.thing_classes = names
            classifier = self._get_clip_embeddings(names)
            num_classes = len(names)
            self.class_names = names
        else:
            self.metadata = self.MetadataCatalog.get(self.BUILTIN_METADATA[vocabulary])
            classifier = str(self.detic_root / self.BUILTIN_CLASSIFIER[vocabulary])
            num_classes = len(self.metadata.thing_classes)
            self.class_names = list(self.metadata.thing_classes)

        self.reset_cls_test(self.predictor.model, classifier, num_classes)

        # Some Detic configs use cascade box predictors; update their test threshold too.
        roi_heads = getattr(self.predictor.model, "roi_heads", None)
        box_predictor = getattr(roi_heads, "box_predictor", None)
        if box_predictor is not None:
            if isinstance(box_predictor, (list, tuple)):
                predictors = box_predictor
            else:
                predictors = [box_predictor]
            for pred in predictors:
                if hasattr(pred, "test_score_thresh"):
                    pred.test_score_thresh = self.confidence_threshold

    def detect(
        self,
        image_bgr: np.ndarray,
        text_prompt: Union[str, Sequence[str], None] = None,
        box_threshold: float = 0.25,
        text_threshold: Optional[float] = None,
    ) -> List[DetectionResult]:
        """
        Run Detic on a BGR image.

        For custom vocabulary, text_prompt can override the initialized custom
        vocabulary for this call. This is useful for object-specific ROI tests.
        """
        if image_bgr is None or image_bgr.size == 0:
            return []

        if self.vocabulary == "custom" and text_prompt is not None:
            names = self._normalize_custom_vocabulary(text_prompt, self.custom_vocabulary)
            # Reset only when the requested class list changed.
            if names != self.class_names:
                self.set_vocabulary(vocabulary="custom", custom_vocabulary=names)

        # Update runtime threshold.
        self.confidence_threshold = float(box_threshold)

        outputs = self.predictor(image_bgr)
        instances = outputs.get("instances", None)
        if instances is None:
            return []

        instances = instances.to("cpu")
        if not instances.has("pred_boxes") or not instances.has("scores") or not instances.has("pred_classes"):
            return []

        boxes = instances.pred_boxes.tensor.numpy()
        scores = instances.scores.numpy()
        classes = instances.pred_classes.numpy()

        preds: List[DetectionResult] = []
        for box, score, cls_idx in zip(boxes, scores, classes):
            score = float(score)
            if score < box_threshold:
                continue

            x1, y1, x2, y2 = [int(round(float(v))) for v in box.tolist()]
            cls_idx = int(cls_idx)
            if 0 <= cls_idx < len(self.class_names):
                label = self.class_names[cls_idx]
            else:
                label = str(cls_idx)

            preds.append(
                DetectionResult(
                    bbox_xyxy=(x1, y1, x2, y2),
                    confidence=score,
                    phrase=label,
                )
            )

        preds.sort(key=lambda d: d.confidence, reverse=True)
        return preds
