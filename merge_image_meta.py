"""After Colab: put leaf_cnn.keras, landcover_cnn.keras, image_meta.json into models/, then run: python merge_image_meta.py
Merges the image class lists/metrics into models/meta.json without touching the yield/crop settings."""
import json
from common import MODELS
m = json.loads((MODELS / "meta.json").read_text()) if (MODELS / "meta.json").exists() else {}
im = json.loads((MODELS / "image_meta.json").read_text())
m["leaf_classes"], m["landcover_classes"] = im["leaf_classes"], im["landcover_classes"]
m.setdefault("metrics", {})["leaf"] = {"accuracy": im["leaf_val_accuracy"]}
m["metrics"]["landcover"] = {"accuracy": im["landcover_val_accuracy"]}
(MODELS / "meta.json").write_text(json.dumps(m, indent=2)); print("meta.json updated with real image-model results")
