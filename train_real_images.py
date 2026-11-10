"""
train_real_images.py - train the leaf-disease or land-cover model on REAL images.

Folder layout (one sub-folder per class, any number of .jpg/.png inside):

    my_leaves/
        healthy/   img1.jpg img2.jpg ...
        blight/    ...
        rust/      ...

Examples:
    # PlantVillage downloaded from Kaggle -> leaf-disease model (needs internet the first time, for ImageNet weights)
    python train_real_images.py --data path/to/PlantVillage --name leaf --epochs 8

    # EuroSAT -> land-cover model
    python train_real_images.py --data path/to/EuroSAT --name landcover --epochs 8

The trained model replaces models/<name>_cnn.keras and the class list in models/meta.json,
so `streamlit run app.py` / `uvicorn api:app` pick it up automatically.
"""
import argparse
import json

import tensorflow as tf
from tensorflow.keras import layers, models

from common import MODELS

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True, help="folder with one sub-folder per class")
ap.add_argument("--name", required=True, choices=["leaf", "landcover"])
ap.add_argument("--epochs", type=int, default=8)
ap.add_argument("--size", type=int, default=128, help="image size; 96/128/160/192/224 suit MobileNetV2")
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--weights", default="imagenet", choices=["imagenet", "none"],
                help="'imagenet' = transfer learning (downloads weights once); 'none' = train from scratch")
ap.add_argument("--out-dir", default=str(MODELS))
args = ap.parse_args()

size = (args.size, args.size)
train = tf.keras.utils.image_dataset_from_directory(
    args.data, validation_split=0.2, subset="training", seed=42, image_size=size, batch_size=args.batch)
val = tf.keras.utils.image_dataset_from_directory(
    args.data, validation_split=0.2, subset="validation", seed=42, image_size=size, batch_size=args.batch)
classes = train.class_names
print(f"{len(classes)} classes: {classes}")

to_unit = lambda x, y: (x / 255.0, y)           # the app feeds the model values in [0, 1]
train = train.map(to_unit).prefetch(tf.data.AUTOTUNE)
val = val.map(to_unit).prefetch(tf.data.AUTOTUNE)

augment = tf.keras.Sequential([layers.RandomFlip("horizontal"), layers.RandomRotation(0.1), layers.RandomZoom(0.1)])
base = tf.keras.applications.MobileNetV2(
    input_shape=size + (3,), include_top=False, weights=None if args.weights == "none" else "imagenet")
base.trainable = args.weights == "none"

inputs = layers.Input(shape=size + (3,))
x = augment(inputs)
x = layers.Rescaling(2.0, offset=-1.0)(x)        # [0,1] -> [-1,1], what MobileNetV2 expects
x = base(x, training=False if args.weights != "none" else None)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(len(classes), activation="softmax")(x)
model = models.Model(inputs, outputs)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

hist = model.fit(train, validation_data=val, epochs=args.epochs)
loss, acc = model.evaluate(val, verbose=0)
print(f"validation accuracy: {acc:.4f}")

out = MODELS if args.out_dir == str(MODELS) else __import__("pathlib").Path(args.out_dir)
out.mkdir(parents=True, exist_ok=True)
model.save(out / f"{args.name}_cnn.keras")
meta_path = out / "meta.json"
meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
meta[f"{args.name}_classes"] = classes
meta.setdefault("metrics", {}).setdefault(args.name, {})["accuracy"] = round(float(acc), 4)
meta_path.write_text(json.dumps(meta, indent=2))
print(f"Saved {out / (args.name + '_cnn.keras')} and updated meta.json")
