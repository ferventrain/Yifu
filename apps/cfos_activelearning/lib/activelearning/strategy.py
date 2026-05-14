from __future__ import annotations

import logging

from monailabel.interfaces.datastore import Datastore
from monailabel.interfaces.tasks.strategy import Strategy

logger = logging.getLogger(__name__)


class HighestScoreStrategy(Strategy):
    def __init__(self, score_key: str):
        super().__init__("Pick unlabeled image with highest cached uncertainty score")
        self.score_key = score_key

    def __call__(self, request, datastore: Datastore):
        label_tag = request.get("label_tag")
        labels = request.get("labels")
        images = datastore.get_unlabeled_images(label_tag, labels)
        if not images:
            return None

        best = None
        best_score = float("-inf")
        for image_id in images:
            info = datastore.get_image_info(image_id) or {}
            strategy_info = info.get("strategy", {}).get(self.score_key, {})
            score = strategy_info.get("score")
            if score is None:
                continue
            if float(score) > best_score:
                best = image_id
                best_score = float(score)

        if best is None:
            logger.info("No cached score found for strategy=%s; returning first unlabeled image", self.score_key)
            images.sort()
            return {"id": images[0]}

        logger.info("HighestScoreStrategy selected %s with score %.6f", best, best_score)
        return {"id": best, "score": best_score}

