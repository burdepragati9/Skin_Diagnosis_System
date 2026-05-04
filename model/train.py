import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.threading.set_intra_op_parallelism_threads(2)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(__file__).resolve().parent
MODEL_PATH = MODEL_DIR / "skin_model.keras"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"

IMG_SIZE = 224
BATCH_SIZE = 8
SEED = 123
MOBILENET_ALPHA = 0.5
SHUFFLE_BUFFER_SIZE = 128
PREFETCH_BUFFER_SIZE = 1


def image_count(data_dir: Path) -> int:
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
    return sum(1 for path in data_dir.rglob("*") if path.suffix.lower() in extensions)


def default_data_dir() -> Path:
    cropped = PROJECT_ROOT / "CroppedData"
    if cropped.exists() and image_count(cropped) > 0:
        return cropped
    return PROJECT_ROOT / "MySkinData"


def build_model(num_classes: int, use_pretrained: bool, alpha: float) -> keras.Model:
    weights = "imagenet" if use_pretrained else None
    try:
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(IMG_SIZE, IMG_SIZE, 3),
            alpha=alpha,
            include_top=False,
            weights=weights,
        )
    except Exception as exc:
        print(f"Could not load pretrained weights ({exc}). Training without them.")
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(IMG_SIZE, IMG_SIZE, 3),
            alpha=alpha,
            include_top=False,
            weights=None,
        )

    base_model.trainable = False

    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.10),
        ],
        name="data_augmentation",
    )

    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = data_augmentation(inputs)
    x = layers.Rescaling(1.0 / 127.5, offset=-1)(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.35)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the skin disease classifier.")
    parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--mobilenet-alpha", type=float, default=MOBILENET_ALPHA)
    parser.add_argument("--cache", action="store_true", help="Cache datasets in RAM.")
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()

    if args.batch_size < 1:
        raise SystemExit("--batch-size must be at least 1")
    if args.mobilenet_alpha <= 0:
        raise SystemExit("--mobilenet-alpha must be greater than 0")

    data_dir = args.data_dir.resolve()
    if not data_dir.exists() or image_count(data_dir) == 0:
        raise SystemExit(f"No training images found in: {data_dir}")

    print(f"Training data: {data_dir}")
    print(f"Images found: {image_count(data_dir)}")
    print(f"Batch size: {args.batch_size}")
    print(f"MobileNetV2 alpha: {args.mobilenet_alpha}")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=SEED,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=args.batch_size,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=SEED,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=args.batch_size,
    )

    class_names = train_ds.class_names
    CLASS_NAMES_PATH.write_text(json.dumps(class_names, indent=2), encoding="utf-8")
    print(f"Classes: {class_names}")

    if args.cache:
        train_ds = train_ds.cache()
        val_ds = val_ds.cache()

    train_ds = train_ds.shuffle(SHUFFLE_BUFFER_SIZE, seed=SEED).prefetch(PREFETCH_BUFFER_SIZE)
    val_ds = val_ds.prefetch(PREFETCH_BUFFER_SIZE)

    model = build_model(
        num_classes=len(class_names),
        use_pretrained=not args.no_pretrained,
        alpha=args.mobilenet_alpha,
    )

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    model.save(MODEL_PATH)
    best_val_accuracy = max(history.history.get("val_accuracy", [0.0]))
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Class names saved to: {CLASS_NAMES_PATH}")
    print(f"Best validation accuracy: {best_val_accuracy:.4f}")


if __name__ == "__main__":
    main()
