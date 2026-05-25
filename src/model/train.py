"""
train.py — Bucle de entrenamiento para el modelo de clasificación de acordes.

Ejecutar desde la terminal:
    cd src/model
    ../../venv/bin/python train.py
"""

import os
import time
import torch

from config import build_model, DEVICE, BATCH_SIZE, EPOCHS, SAVE_EVERY, OUTPUT_DIR
from dataset import get_loaders


def train():
    print("Cargando datasets...")
    train_loader, test_loader = get_loaders(batch_size=BATCH_SIZE)
    print(f"  Train: {len(train_loader.dataset)} muestras ({len(train_loader)} batches)")
    print(f"  Test:  {len(test_loader.dataset)} muestras ({len(test_loader)} batches)")

    model, criterion, optimizer = build_model()

    checkpoint_path = os.path.join(OUTPUT_DIR, "checkpoint.pth")
    best_path = os.path.join(OUTPUT_DIR, "mejor_modelo.pth")
    last_path = os.path.join(OUTPUT_DIR, "ultimo_modelo.pth")
    log_path = os.path.join(OUTPUT_DIR, "training_log.txt")

    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("epoca,train_loss,train_acc,test_acc,tiempo_seg,mejor\n")

    best_acc = 0.0

    print(f"\nIniciando entrenamiento por {EPOCHS} épocas en {DEVICE}...\n")

    for epoch in range(EPOCHS):
        t0 = time.time()

        # Fase de entrenamiento
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for spec, aux, label in train_loader:
            spec = spec.to(DEVICE)
            aux = aux.to(DEVICE)
            label = label.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(spec, aux)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total

        # Fase de evaluación
        model.eval()
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for spec, aux, label in test_loader:
                spec = spec.to(DEVICE)
                aux = aux.to(DEVICE)
                label = label.to(DEVICE)

                outputs = model(spec, aux)
                _, predicted = torch.max(outputs, 1)
                test_total += label.size(0)
                test_correct += (predicted == label).sum().item()

        test_acc = 100 * test_correct / test_total
        elapsed = time.time() - t0

        is_best = test_acc > best_acc
        if is_best:
            best_acc = test_acc
            torch.save(model.state_dict(), best_path)

        if (epoch + 1) % SAVE_EVERY == 0:
            torch.save(model.state_dict(), checkpoint_path)

        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(f"{epoch+1},{train_loss:.4f},{train_acc:.2f},{test_acc:.2f},{elapsed:.1f},{'si' if is_best else 'no'}\n")

        marker = " << mejor" if is_best else ""
        print(f"Época {epoch+1:>2}/{EPOCHS} | "
              f"Loss: {train_loss:.4f} | "
              f"Train: {train_acc:.1f}% | "
              f"Test: {test_acc:.1f}% | "
              f"{elapsed:.1f}s{marker}")

    torch.save(model.state_dict(), last_path)

    print(f"\nEntrenamiento finalizado.")
    print(f"  Mejor precisión en Test: {best_acc:.1f}%")
    print(f"  Archivos guardados en: {os.path.abspath(OUTPUT_DIR)}/")


if __name__ == '__main__':
    train()
