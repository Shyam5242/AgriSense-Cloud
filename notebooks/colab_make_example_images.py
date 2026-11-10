"""Create demonstration images from PlantVillage and EuroSAT in Google Colab.

Run this file in a Colab notebook cell with:
    %run /content/agrisense-cloud/notebooks/colab_make_example_images.py

The script downloads public datasets through kagglehub, writes examples_images.zip,
then offers a Colab download. It is not an accuracy/evaluation dataset.
"""

from __future__ import annotations

import glob
import os
import random
import subprocess
import sys
import zipfile
from pathlib import Path


def main() -> None:
    # Install optional packages in the Colab runtime, not in the main app environment.
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "kagglehub", "tifffile", "pillow", "numpy"]
    )

    import kagglehub
    import numpy as np
    import tifffile
    from PIL import Image

    random.seed(1)
    output_dir = Path("/content/examples")
    leaf_dir = output_dir / "leaf"
    landcover_dir = output_dir / "landcover"
    leaf_dir.mkdir(parents=True, exist_ok=True)
    landcover_dir.mkdir(parents=True, exist_ok=True)

    # PlantVillage leaf photos.
    root = kagglehub.dataset_download("abdallahalidev/plantvillage-dataset")
    data_dir = next(
        (Path(dp) for dp, _, _ in os.walk(root) if Path(dp).name == "color"), None
    )
    if data_dir is None:
        raise FileNotFoundError("Could not find the PlantVillage 'color' directory.")

    classes = sorted(p.name for p in data_dir.iterdir() if p.is_dir())
    wanted = [
        ("Tomato", "healthy"), ("Tomato", "Early_blight"),
        ("Tomato", "Late_blight"), ("Potato", "Late_blight"),
        ("Apple", "scab"), ("Corn", "rust"),
        ("Grape", "Black_rot"), ("Strawberry", "scorch"),
    ]
    for crop, disease in wanted:
        matches = [name for name in classes if crop.lower() in name.lower() and disease.lower() in name.lower()]
        if not matches:
            print(f"PlantVillage class not found: {crop} / {disease}")
            continue
        class_name = matches[0]
        image_files = sorted(glob.glob(str(data_dir / class_name / "*")))
        for index, filename in enumerate(random.sample(image_files, min(3, len(image_files))), 1):
            with Image.open(filename) as image_file:
                image_file.convert("RGB").save(leaf_dir / f"{class_name}__{index}.jpg", quality=92)
        print("Leaf examples:", class_name)

    # EuroSAT 13-band tiles converted to RGB (B04, B03, B02).
    source = Path(kagglehub.dataset_download("apollo2506/eurosat-dataset"))
    base_dir = source / "EuroSATallBands"
    if not base_dir.is_dir():
        raise FileNotFoundError(f"Could not find EuroSATallBands under {source}")

    for class_dir in sorted(p for p in base_dir.iterdir() if p.is_dir()):
        files = sorted(class_dir.glob("*.tif"))
        if not files:
            continue
        destination = landcover_dir / class_dir.name
        destination.mkdir(parents=True, exist_ok=True)
        for index, filename in enumerate(random.sample(files, min(3, len(files))), 1):
            array = tifffile.imread(filename).astype("float32")
            if array.ndim == 3 and array.shape[0] == 13:
                array = array.transpose(1, 2, 0)
            if array.ndim != 3 or array.shape[-1] < 4:
                print("Skipping unexpected image shape:", filename, array.shape)
                continue
            rgb = np.clip(array[:, :, [3, 2, 1]] / 3000.0, 0, 1) * 255
            Image.fromarray(rgb.astype("uint8")).resize((128, 128)).save(
                destination / f"{class_dir.name}__{index}.png"
            )
        print("Land-cover examples:", class_dir.name)

    archive = Path("/content/examples_images.zip")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zip_file:
        for filename in output_dir.rglob("*"):
            if filename.is_file():
                zip_file.write(filename, filename.relative_to("/content"))

    try:
        from google.colab import files
        files.download(str(archive))
    except ImportError:
        print(f"Created {archive}. Download it from the runtime file browser.")


if __name__ == "__main__":
    main()
