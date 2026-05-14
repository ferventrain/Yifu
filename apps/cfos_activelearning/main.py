import logging
import os
import sys
from typing import Dict

import monailabel
from monailabel.config import settings
from monailabel.interfaces.datastore import Datastore
from monailabel.interfaces.app import MONAILabelApp
from monailabel.interfaces.tasks.infer_v2 import InferTask
from monailabel.interfaces.tasks.scoring import ScoringMethod
from monailabel.interfaces.tasks.strategy import Strategy
from monailabel.tasks.activelearning.first import First
from monailabel.tasks.activelearning.random import Random

APP_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(APP_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from lib.activelearning import HighestScoreStrategy
from lib.datastore import CFOSLocalDatastore
from lib.infers import CFOSActiveLearningInfer
from lib.scoring import CFOSUncertaintyScoring

logger = logging.getLogger(__name__)


class CFOSActiveLearningApp(MONAILabelApp):
    def __init__(self, app_dir: str, studies: str, conf: Dict[str, str]):
        self.model_name = conf.get("model_name", "cfos_unet")
        super().__init__(
            app_dir=app_dir,
            studies=studies,
            conf=conf,
            name=f"cFos Active Learning ({monailabel.__version__})",
            description="MONAI Label app for 3D cFos segmentation active learning.",
            version=monailabel.__version__,
            labels=["background", "cfos"],
        )

    def init_infers(self) -> Dict[str, InferTask]:
        infer_task = CFOSActiveLearningInfer(self.conf)
        logger.info("+++ Adding Inferer:: %s => %s", self.model_name, infer_task)
        return {self.model_name: infer_task}

    def init_datastore(self) -> Datastore:
        logger.info("Init CFOS Datastore for: %s", self.studies)
        return CFOSLocalDatastore(
            self.studies,
            extensions=settings.MONAI_LABEL_DATASTORE_FILE_EXT,
            auto_reload=settings.MONAI_LABEL_DATASTORE_AUTO_RELOAD,
            read_only=settings.MONAI_LABEL_DATASTORE_READ_ONLY,
        )

    def init_trainers(self):
        return {}

    def init_strategies(self) -> Dict[str, Strategy]:
        strategies: Dict[str, Strategy] = {
            "random": Random(),
            "first": First(),
            "highest_entropy": HighestScoreStrategy(score_key=self.model_name),
        }
        logger.info("Active Learning Strategies:: %s", list(strategies.keys()))
        return strategies

    def init_scoring_methods(self) -> Dict[str, ScoringMethod]:
        scorer = CFOSUncertaintyScoring(self.conf, infer_model=self.model_name)
        logger.info("+++ Adding Scoring Method:: %s => %s", self.model_name, scorer)
        return {self.model_name: scorer}


def main():
    from monailabel.interfaces.utils.app import run_main

    run_main()


if __name__ == "__main__":
    main()
