"""Thin SAM2 wrapper for interactive image segmentation."""
from __future__ import annotations

import numpy as np
import torch


class SAM2Interactive:
    """Lazy-loading wrapper around ``SAM2ImagePredictor``.

    The SAM2 model (~2 GB) is downloaded from HuggingFace on the first call to
    ``set_image`` or ``predict``.  Subsequent calls within the same process reuse
    the cached weights.
    """

    def __init__(self, device: str = "cuda") -> None:
        self.device = device
        self._predictor = None
        self._image_set  = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if self._predictor is not None:
            return
        from sam2.sam2_image_predictor import SAM2ImagePredictor  # noqa: PLC0415
        self._predictor = SAM2ImagePredictor.from_pretrained(
            "facebook/sam2.1-hiera-large",
            device=self.device,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_image(self, image_np: np.ndarray) -> None:
        """Encode *image_np* (H×W×3 uint8 RGB) into SAM2's image embedding."""
        self._load()
        with torch.inference_mode():
            self._predictor.set_image(image_np)
        self._image_set = True

    def predict(
        self,
        points: list,
        labels: list[int],
        multimask: bool = True,
    ) -> tuple[np.ndarray | None, float]:
        """Run SAM2 prediction for *points* / *labels*.

        Parameters
        ----------
        points:
            List of ``[x, y]`` pixel coordinates.
        labels:
            Per-point label: ``1`` = foreground (positive), ``0`` = background (negative).
        multimask:
            When True, SAM2 returns 3 candidate masks; the highest-scoring one is
            selected automatically.

        Returns
        -------
        (mask, score)
            *mask* is a boolean ``(H, W)`` ndarray; *score* is the model confidence.
            Returns ``(None, 0.0)`` when prediction is not possible.
        """
        if not self._image_set or not points:
            return None, 0.0

        coords = np.array([[p[0], p[1]] for p in points], dtype=np.float32)
        lbls   = np.array(labels,                         dtype=np.int32)

        with torch.inference_mode():
            masks, scores, _ = self._predictor.predict(
                point_coords=coords,
                point_labels=lbls,
                multimask_output=multimask,
            )

        best = int(np.argmax(scores))
        return masks[best].astype(bool), float(scores[best])

    def reset_image(self) -> None:
        """Clear the cached image embedding (frees GPU memory)."""
        self._image_set = False
        if self._predictor is not None:
            try:
                self._predictor.reset_predictor()
            except Exception:
                pass
