from __future__ import annotations

import logging
from typing import Any, Dict

from monailabel.interfaces.datastore import Datastore
from monailabel.interfaces.tasks.scoring import ScoringMethod

from lib.infers import CFOSActiveLearningInfer

logger = logging.getLogger(__name__)


class CFOSUncertaintyScoring(ScoringMethod):
    def __init__(self, conf: Dict[str, Any], infer_model: str = "cfos_unet"):
        super().__init__("Voxelwise entropy scoring for 3D cFos active learning")
        self.model_name = infer_model
        self.infer_task = CFOSActiveLearningInfer(conf)

    def __call__(self, request, datastore: Datastore):
        label_tag = request.get("label_tag")
        labels = request.get("labels")
        images = request.get("images")
        if not images:
            images = datastore.get_unlabeled_images(label_tag, labels)

        results = []
        for image_id in images:
            image_uri = datastore.get_image_uri(image_id)
            infer_result = self.infer_task.infer_array(image_uri)
            entropy = infer_result["entropy"]
            score = float(entropy.mean())
            stats = {
                "score": score,
                "mean_entropy": score,
                "max_entropy": float(entropy.max()),
                "foreground_ratio": float(infer_result["prediction"].mean()),
            }
            info = datastore.get_image_info(image_id) or {}
            strategy_info = dict(info.get("strategy", {}))
            strategy_info[self.model_name] = stats
            datastore.update_image_info(image_id, {"strategy": strategy_info})
            results.append({"id": image_id, **stats})
            logger.info("Scoring %s => %.6f", image_id, score)

        results.sort(key=lambda item: item["score"], reverse=True)
        return {"method": self.model_name, "results": results}

