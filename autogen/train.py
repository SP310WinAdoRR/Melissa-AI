"""
train.py — Bucle de entrenamiento y evaluación para el modelo ChordCNN.

Pipeline completo:
    1. Descubre automáticamente todas las pistas de GuitarSet
    2. Divide los datos por guitarrista (player) para evitar data leakage
    3. Crea DataLoaders de entrenamiento y validación
    4. Entrena la CNN con CrossEntropyLoss + Adam
    5. Evalúa en validación cada época y guarda el mejor modelo
"""

import os
import glob
import time

# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn as nn
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
from torch.utils.data import DataLoader

from dataset import GuitarSetDataset, IDX_TO_CHORD
from model import ChordCNN

# =====================================================================
# 1. CONFIGURACIÓN GENERAL
# =====================================================================

CONFIG = {
    # Rutas
    "base_dir": "GuitarSet/3371780",
    "audio_type": "audio_mono-pickup_mix",
    "save_path": "best_model.pth",

    # Parámetros de audio / dataset
    "sr": 44100,
    "frame_duration": 0.5,
    "overlap": 0.5,

    # Hiperparámetros de entrenamiento
    "batch_size": 32,
    "learning_rate": 1e-3,
    "epochs": 30,
    "dropout_rate": 0.3,
    "num_classes": 25,

    # División de datos por guitarrista (player ID)
    # GuitarSet tiene 6 guitarristas: 00, 01, 02, 03, 04, 05
    # Usamos 4 para Train, 1 para Validación, 1 para Test
    "train_players": ["00", "01", "02", "03"],
    "val_players": ["04"],
    "test_players": ["05"],
}


# =====================================================================
# 2. FUNCIONES AUXILIARES
# =====================================================================

def discover_tracks(base_dir, players):
    """
    Descubre automáticamente los nombres de pistas (sin extensión)
    disponibles en la carpeta de anotaciones, filtrados por guitarrista.
    """
    annotation_dir = os.path.join(base_dir, "annotation")
    all_jams = glob.glob(os.path.join(annotation_dir, "*.jams"))

    tracks = []
    for jams_path in sorted(all_jams):
        track_name = os.path.basename(jams_path).replace(".jams", "")
        player_id = track_name.split("_")[0]  # "00", "01", etc.
        if player_id in players:
            tracks.append(track_name)

    return tracks


def compute_accuracy(model, dataloader, device):
    """
    Calcula la precisión (exact-match accuracy) sobre un DataLoader completo.
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, dim=1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total if total > 0 else 0.0


def compute_per_class_accuracy(model, dataloader, device, num_classes=25):
    """
    Calcula la precisión desglosada por cada clase de acorde.
    Útil para detectar si el modelo ignora clases minoritarias.
    """
    model.eval()
    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, dim=1)

            for i in range(labels.size(0)):
                label = labels[i].item()
                class_total[label] += 1
                if predicted[i].item() == label:
                    class_correct[label] += 1

    return class_correct, class_total


# =====================================================================
# 3. BUCLE DE ENTRENAMIENTO PRINCIPAL
# =====================================================================

def train():
    # ─── Selección de dispositivo (GPU si está disponible) ───
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  🎸 Melissa AI — Entrenamiento de ChordCNN")
    print(f"{'='*60}")
    print(f"  Dispositivo: {device}")

    # ─── Descubrir pistas por guitarrista ───
    train_tracks = discover_tracks(CONFIG["base_dir"], CONFIG["train_players"])
    val_tracks = discover_tracks(CONFIG["base_dir"], CONFIG["val_players"])

    print(f"\n  División de datos (por guitarrista, sin data leakage):")
    print(f"    Train: guitarristas {CONFIG['train_players']} → {len(train_tracks)} pistas")
    print(f"    Val:   guitarristas {CONFIG['val_players']} → {len(val_tracks)} pistas")
    print(f"    Test:  guitarristas {CONFIG['test_players']} → (reservado para evaluación final)")

    # ─── Crear Datasets ───
    print(f"\n{'─'*60}")
    train_dataset = GuitarSetDataset(
        base_dir=CONFIG["base_dir"],
        track_names=train_tracks,
        audio_type=CONFIG["audio_type"],
        sr=CONFIG["sr"],
        frame_duration=CONFIG["frame_duration"],
        overlap=CONFIG["overlap"]
    )

    val_dataset = GuitarSetDataset(
        base_dir=CONFIG["base_dir"],
        track_names=val_tracks,
        audio_type=CONFIG["audio_type"],
        sr=CONFIG["sr"],
        frame_duration=CONFIG["frame_duration"],
        overlap=CONFIG["overlap"]
    )

    # ─── Crear DataLoaders ───
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False)

    print(f"\n  Resumen del dataset:")
    print(f"    Muestras de entrenamiento: {len(train_dataset)}")
    print(f"    Muestras de validación:    {len(val_dataset)}")
    print(f"    Batches por época (train): {len(train_loader)}")

    # ─── Inicializar modelo, pérdida y optimizador ───
    model = ChordCNN(num_classes=CONFIG["num_classes"], dropout_rate=CONFIG["dropout_rate"])
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])

    # Scheduler para reducir el learning rate cuando la validación se estanque
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )

    # ─── Tracking del mejor modelo ───
    best_val_accuracy = 0.0
    best_epoch = 0

    print(f"\n{'='*60}")
    print(f"  Iniciando entrenamiento por {CONFIG['epochs']} épocas...")
    print(f"{'='*60}\n")

    # ─── Bucle de entrenamiento ───
    for epoch in range(1, CONFIG["epochs"] + 1):
        model.train()
        running_loss = 0.0
        epoch_start = time.time()

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass + optimización
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # ─── Métricas de la época ───
        avg_loss = running_loss / len(train_loader)
        train_acc = compute_accuracy(model, train_loader, device)
        val_acc = compute_accuracy(model, val_loader, device)
        epoch_time = time.time() - epoch_start

        # Actualizar scheduler
        scheduler.step(val_acc)

        # ─── Guardar el mejor modelo ───
        improved = ""
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), CONFIG["save_path"])
            improved = " ★ BEST"

        # ─── Imprimir progreso ───
        current_lr = optimizer.param_groups[0]['lr']
        print(
            f"  Época {epoch:3d}/{CONFIG['epochs']} │ "
            f"Loss: {avg_loss:.4f} │ "
            f"Train Acc: {train_acc:.1%} │ "
            f"Val Acc: {val_acc:.1%} │ "
            f"LR: {current_lr:.1e} │ "
            f"{epoch_time:.1f}s{improved}"
        )

    # ─── Resumen final ───
    print(f"\n{'='*60}")
    print(f"  ✅ Entrenamiento finalizado.")
    print(f"  Mejor validación: {best_val_accuracy:.1%} (época {best_epoch})")
    print(f"  Modelo guardado en: {CONFIG['save_path']}")
    print(f"{'='*60}")

    # ─── Evaluación por clase del mejor modelo ───
    print(f"\n  📊 Precisión por clase de acorde (mejor modelo):")
    model.load_state_dict(torch.load(CONFIG["save_path"], weights_only=True))
    model.eval()

    class_correct, class_total = compute_per_class_accuracy(model, val_loader, device)

    print(f"  {'Acorde':<8} {'Correctos':>10} {'Total':>8} {'Precisión':>10}")
    print(f"  {'─'*40}")
    for i in range(CONFIG["num_classes"]):
        if class_total[i] > 0:
            acc = class_correct[i] / class_total[i]
            print(f"  {IDX_TO_CHORD[i]:<8} {class_correct[i]:>10} {class_total[i]:>8} {acc:>10.1%}")

    print(f"\n🎉 ¡Proceso completo!")


if __name__ == "__main__":
    train()
