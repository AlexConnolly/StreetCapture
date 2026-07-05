"""Multi-group label taxonomy.

In v1 only the labels that are *deterministic* from a YOLO/COCO class are
populated: ``object`` and ``subtype`` always, ``function`` where defensible.
The emergent label types — ``company`` (DPD, Amazon…), ``energy``, and other
``attributes`` — require clustering / identity resolution and are left empty
until v2; the schema and query filters already accept them.
"""

from __future__ import annotations

VEHICLE_CLASSES = {"car", "truck", "bus", "motorbike", "motorcycle", "bicycle", "train"}
BIKE_CLASSES = {"bicycle", "motorbike", "motorcycle"}

# subtype (physical) — from COCO class
SUBTYPE = {
    "car": "car", "truck": "truck", "bus": "bus", "train": "train",
    "bicycle": "bicycle", "motorcycle": "motorbike", "motorbike": "motorbike",
    "person": "person",
}
# function — only where it follows from the class alone
FUNCTION = {"bus": "passenger transport", "train": "passenger transport"}


def category(cls_name: str) -> str:
    """Coarse dashboard bucket: person / vehicle / other."""
    if cls_name == "person":
        return "person"
    if cls_name in VEHICLE_CLASSES:
        return "vehicle"
    return "other"


def object_label(cls_name: str) -> str:
    if cls_name == "person":
        return "person"
    if cls_name in BIKE_CLASSES:
        return "bike"
    if cls_name in VEHICLE_CLASSES:
        return "vehicle"
    return cls_name


def labels_for(cls_name: str) -> list[dict]:
    """Return the multi-label list for an artifact's primary class."""
    labels = [{"type": "object", "value": object_label(cls_name)}]
    if cls_name in SUBTYPE:
        labels.append({"type": "subtype", "value": SUBTYPE[cls_name]})
    if cls_name in FUNCTION:
        labels.append({"type": "function", "value": FUNCTION[cls_name]})
    return labels
