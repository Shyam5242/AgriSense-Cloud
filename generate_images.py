"""
generate_images.py
---------------------
Creates synthetic RGB image arrays (actual pixel tensors, not feature
vectors) for the leaf-disease and land-cover CNN classifiers. Each class
gets a distinct base color + procedural texture/noise pattern so a real
CNN has genuine (if synthetic) visual signal to learn from.

This exists because the sandbox has no internet access to PlantVillage,
EuroSAT, or the ImageNet weights MobileNetV2 normally loads (confirmed:
storage.googleapis.com is blocked here with HTTP 403). See
keras_reference.py for the real MobileNetV2 + ImageNet-weights pipeline
to run once those are reachable.
"""
import numpy as np
import os

RNG = np.random.default_rng(7)
OUT = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(OUT, exist_ok=True)

IMG_SIZE = 64


def _base_pattern(h, w, base_color, blob_count, blob_color, blob_radius):
    img = np.ones((h, w, 3), dtype=np.float32) * np.array(base_color)
    yy, xx = np.mgrid[0:h, 0:w]
    for _ in range(blob_count):
        cy, cx = RNG.integers(0, h), RNG.integers(0, w)
        r = blob_radius + RNG.integers(-2, 3)
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2
        img[mask] = blob_color
    img += RNG.normal(0, 10, size=img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def make_leaf_images(n_per_class=180):
    """healthy / blight / rust / mildew leaf-style synthetic images."""
    classes = {
        "healthy": ((40, 130, 40), 3, (25, 100, 25), 6),
        "blight":  ((55, 110, 35), 10, (90, 60, 20), 5),
        "rust":    ((60, 120, 40), 12, (170, 100, 30), 4),
        "mildew":  ((50, 115, 45), 14, (220, 220, 200), 5),
    }
    X, y = [], []
    for label, (base, blobs, blob_color, radius) in classes.items():
        for _ in range(n_per_class):
            img = _base_pattern(IMG_SIZE, IMG_SIZE, base, blobs, blob_color, radius)
            # heavier sensor-style noise so classes overlap somewhat,
            # like real-world lighting/camera variation would cause
            img = img.astype(np.float32) + RNG.normal(0, 28, size=img.shape)
            X.append(np.clip(img, 0, 255).astype(np.uint8))
            y.append(label)
    X = np.stack(X).astype("float32") / 255.0
    y = np.array(y)
    np.savez_compressed(f"{OUT}/leaf_images.npz", X=X, y=y)
    return X, y


def make_landcover_images(n_per_class=130):
    """annual_crop / forest / pasture / residential / water style patches."""
    classes = {
        "annual_crop":  ((195, 185, 90), 6, (165, 150, 60), 5),
        "forest":       ((20, 80, 25), 8, (10, 50, 15), 6),
        "pasture":      ((110, 170, 70), 5, (90, 150, 55), 7),
        "residential":  ((150, 150, 150), 16, (190, 60, 50), 3),
        "water":        ((30, 70, 140), 2, (50, 100, 180), 8),
    }
    X, y = [], []
    for label, (base, blobs, blob_color, radius) in classes.items():
        for _ in range(n_per_class):
            img = _base_pattern(IMG_SIZE, IMG_SIZE, base, blobs, blob_color, radius)
            img = img.astype(np.float32) + RNG.normal(0, 28, size=img.shape)
            X.append(np.clip(img, 0, 255).astype(np.uint8))
            y.append(label)
    X = np.stack(X).astype("float32") / 255.0
    y = np.array(y)
    np.savez_compressed(f"{OUT}/landcover_images.npz", X=X, y=y)
    return X, y


if __name__ == "__main__":
    X1, y1 = make_leaf_images()
    X2, y2 = make_landcover_images()
    print("leaf_images.npz      :", X1.shape, "classes:", sorted(set(y1)))
    print("landcover_images.npz :", X2.shape, "classes:", sorted(set(y2)))
