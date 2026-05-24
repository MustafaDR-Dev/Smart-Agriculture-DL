from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from sklearn.metrics import classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers, models
from tensorflow.keras.applications import mobilenet_v2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam


# Paramètres
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
FINE_TUNE_EPOCHS = 10
LEARNING_RATE = 0.001
FINE_TUNE_LR = 1e-5
RANDOM_STATE = 42

np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


def get_model_output_path() -> Path:
    project_root = Path(__file__).resolve().parents[1]
    return project_root / "models" / "mobilenetv2_plant_disease.keras"


def load_dataset():
    (ds_train, ds_val, ds_test), info = tfds.load(
        "plant_village",
        split=["train[:70%]", "train[70%:85%]", "train[85%:]"],
        with_info=True,
        as_supervised=True,
    )
    class_names = info.features["label"].names
    num_classes = info.features["label"].num_classes

    print("[OK] Dataset PlantVillage chargé")
    print(f"  Split: 70/15/15 (train/val/test)")
    print(f"  Nombre de classes: {num_classes}")

    return ds_train, ds_val, ds_test, class_names, num_classes


def build_data_pipelines(ds_train, ds_val, ds_test):
    autotune = tf.data.AUTOTUNE

    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ])

    def resize(image, label):
        image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
        image = tf.cast(image, tf.float32)
        return image, label

    # Préparer les pipelines de base
    # cache() avant l'augmentation : les images redimensionnées sont mises en cache,
    # l'augmentation est ré-appliquée aléatoirement à chaque époque
    train_base = ds_train.map(resize, num_parallel_calls=autotune).cache()
    train_base = train_base.shuffle(10000).map(
        lambda x, y: (data_augmentation(x, training=True), y),
        num_parallel_calls=autotune,
    ).batch(BATCH_SIZE).prefetch(autotune)

    val_base = ds_val.map(resize, num_parallel_calls=autotune).batch(BATCH_SIZE).cache().prefetch(autotune)
    test_base = ds_test.map(resize, num_parallel_calls=autotune).batch(BATCH_SIZE).cache().prefetch(autotune)

    # Appliquer preprocess_input de MobileNetV2
    train_ds = train_base.map(lambda x, y: (mobilenet_v2.preprocess_input(x), y), num_parallel_calls=autotune)
    val_ds = val_base.map(lambda x, y: (mobilenet_v2.preprocess_input(x), y), num_parallel_calls=autotune)
    test_ds = test_base.map(lambda x, y: (mobilenet_v2.preprocess_input(x), y), num_parallel_calls=autotune)

    print("[OK] Pipelines créés (preprocess_input MobileNetV2, 224x224)")
    return train_ds, val_ds, test_ds


def build_mobilenetv2_model(num_classes):
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    print("[OK] Modèle MobileNetV2 construit (base gelée)")
    return model, base_model


def train_transfer_learning(model, train_ds, val_ds, checkpoint_path, class_weights):
    early_stop = EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True,
    )
    checkpoint = ModelCheckpoint(
        checkpoint_path, monitor="val_loss", save_best_only=True, verbose=1,
    )
    print("\n>>> Phase 1 — Transfer Learning (base gelée)...")
    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
        callbacks=[early_stop, checkpoint],
        class_weight=class_weights,
        verbose=1,
    )
    loss, accuracy = model.evaluate(val_ds, verbose=0)
    print(f"[OK] Transfer Learning — Val Accuracy: {accuracy:.4f} | Val Loss: {loss:.4f}")
    return history


def fine_tune_model(model, base_model, train_ds, val_ds, checkpoint_path, class_weights):
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=FINE_TUNE_LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    early_stop = EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True,
    )
    checkpoint = ModelCheckpoint(
        checkpoint_path, monitor="val_loss", save_best_only=True, verbose=1,
    )
    print(f"\n>>> Phase 2 — Fine-tuning ({FINE_TUNE_EPOCHS} époques)...")
    history = model.fit(
        train_ds,
        epochs=FINE_TUNE_EPOCHS,
        validation_data=val_ds,
        callbacks=[early_stop, checkpoint],
        class_weight=class_weights,
        verbose=1,
    )
    loss, accuracy = model.evaluate(val_ds, verbose=0)
    print(f"[OK] Fine-tuning — Val Accuracy: {accuracy:.4f} | Val Loss: {loss:.4f}")
    return history


def evaluate_on_test(model, test_ds, class_names):
    y_true, y_pred = [], []
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(preds, axis=1))
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    loss, acc = model.evaluate(test_ds, verbose=0)
    f1 = f1_score(y_true, y_pred, average="weighted")

    print("=" * 60)
    print("  MobileNetV2 — Résultats sur le jeu de TEST")
    print("=" * 60)
    print(f"  Test Accuracy : {acc:.4f}")
    print(f"  Test Loss     : {loss:.4f}")
    print(f"  F1-score      : {f1:.4f}")
    print(classification_report(y_true, y_pred, target_names=class_names))

    return acc, loss, f1


def plot_training_history(history_tl, history_ft):
    acc_train = history_tl.history["accuracy"] + history_ft.history["accuracy"]
    acc_val = history_tl.history["val_accuracy"] + history_ft.history["val_accuracy"]
    loss_train = history_tl.history["loss"] + history_ft.history["loss"]
    loss_val = history_tl.history["val_loss"] + history_ft.history["val_loss"]
    epochs_range = range(1, len(acc_train) + 1)
    cut = len(history_tl.history["accuracy"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs_range, acc_train, label="Train", marker="o", markersize=4)
    ax1.plot(epochs_range, acc_val, label="Validation", marker="s", markersize=4)
    ax1.axvline(x=cut, color="gray", linestyle="--", alpha=0.5, label="Début fine-tuning")
    ax1.set_title("MobileNetV2 — Accuracy", fontweight="bold")
    ax1.set_xlabel("Époque")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs_range, loss_train, label="Train", marker="o", markersize=4)
    ax2.plot(epochs_range, loss_val, label="Validation", marker="s", markersize=4)
    ax2.axvline(x=cut, color="gray", linestyle="--", alpha=0.5, label="Début fine-tuning")
    ax2.set_title("MobileNetV2 — Loss", fontweight="bold")
    ax2.set_xlabel("Époque")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def save_model(model, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(output_path)
    print(f"\n[OK] Modèle sauvegardé: {output_path}")


def main() -> None:
    model_output_path = get_model_output_path()

    # Chargement des données
    ds_train, ds_val, ds_test, class_names, num_classes = load_dataset()
    train_ds, val_ds, test_ds = build_data_pipelines(ds_train, ds_val, ds_test)

    # Calcul des poids de classes pour compenser le déséquilibre
    train_labels = np.concatenate([y.numpy() for y in ds_train.map(lambda x, y: y).batch(10000)])
    all_classes = np.arange(num_classes)
    weights = compute_class_weight("balanced", classes=all_classes, y=train_labels)
    class_weights = dict(zip(all_classes, weights))
    print(f"[OK] Class weights calculés ({len(class_weights)} classes)")

    # Construction et entraînement
    model, base_model = build_mobilenetv2_model(num_classes)
    checkpoint_path = str(model_output_path.with_name("mobilenetv2_best.keras"))
    history_tl = train_transfer_learning(model, train_ds, val_ds, checkpoint_path, class_weights)
    history_ft = fine_tune_model(model, base_model, train_ds, val_ds, checkpoint_path, class_weights)

    # Évaluation sur le jeu de test
    evaluate_on_test(model, test_ds, class_names)

    # Visualisation
    plot_training_history(history_tl, history_ft)

    # Sauvegarde
    save_model(model, model_output_path)


if __name__ == "__main__":
    main()
