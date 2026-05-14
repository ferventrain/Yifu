from __future__ import annotations

import logging
import os
import shutil
import time
from typing import Any, Dict

from filelock import FileLock
from monailabel.datastore.local import DataModel, LocalDatastore
from monailabel.interfaces.datastore import DefaultLabelTag
from monailabel.interfaces.exception import ImageNotFoundException

logger = logging.getLogger(__name__)


class CFOSLocalDatastore(LocalDatastore):
    def _label_filename(self, image_id: str, ext: str, label_tag: str) -> str:
        if label_tag == DefaultLabelTag.FINAL or str(label_tag).lower() == "final":
            return image_id.replace("image", "mask") + ext
        return image_id + ext

    def save_label(self, image_id: str, label_filename: str, label_tag: str, label_info: Dict[str, Any]) -> str:
        logger.info("Saving Label for Image: %s; Tag: %s; Info: %s", image_id, label_tag, label_info)
        obj = self._datastore.objects.get(image_id)
        if not obj:
            raise ImageNotFoundException(f"Image {image_id} not found")

        _, label_ext = self._to_id(os.path.basename(label_filename))
        label_id = image_id

        logger.info("Adding Label: %s => %s => %s", image_id, label_tag, label_filename)
        label_path = self._datastore.label_path(label_tag)
        name = self._label_filename(image_id, label_ext, label_tag)
        dest = os.path.join(label_path, name)

        with FileLock(self._lock_file):
            logger.debug("Acquired the lock!")
            os.makedirs(label_path, exist_ok=True)
            shutil.copy(label_filename, dest)

            label_info = label_info if label_info else {}
            label_info["ts"] = int(time.time())
            label_info["name"] = name

            obj.labels[label_tag] = DataModel(info=label_info, ext=label_ext)
            logger.info("Label Info: %s", label_info)
            self._update_datastore_file(lock=False)
        logger.debug("Release the lock!")
        return label_id

