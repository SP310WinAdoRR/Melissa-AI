"""
train.py — Bucle de entrenamiento para el modelo de clasificación de acordes.

Ejecutar desde la terminal:
    cd src/model
    ../../venv/bin/python train.py
"""

import time
import torch

from config import build_model, DEVICE, BATCH_SIZE, EPOCHS
from dataset import get_loaders


def train():
    # 1. Cargar datos
    print("Cargando datasets...")
    train_loader, test_loader = get_loaders(batch_size=BATCH_SIZE)
    print(f"  Train: {len(train_loader.dataset)} muestras ({len(train_loader)} batches)")
    print(f"  Test:  {len(test_loader.dataset)} muestras ({len(test_loader)} batches)")

    # 2. Construir modelo, pérdida y optimizador
    model, criterion, optimizer = build_model()

    best_acc = 0.0

    # 3. Bucle de épocas
    print(f"\nIniciando entrenamiento por {EPOCHS} épocas en {DEVICE}...\n")

    for epoch in range(EPOCHS):
        t0 = time.time()

        # --- FASE DE ENTRENAMIENTO ---
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for spec, aux, label in train_loader:
            # Enviar los tensores al dispositivo (GPU o CPU)
            spec = spec.to(DEVICE)
            aux = aux.to(DEVICE)
            label = label.to(DEVICE)

            # Limpiar gradientes de la iteración anterior
            optimizer.zero_grad()

            # Propagación hacia adelante
            outputs = model(spec, aux)

            # Calcular el error
            loss = criterion(outputs, label)

            # Retropropagación y actualización de pesos
            loss.backward()
            optimizer.step()

            # Acumular métricas
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total

        # --- FASE DE EVALUACIÓN ---
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

        # --- REPORTE ---
        marker = ""
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), "mejor_modelo.pth")
            marker = " << mejor modelo guardado"

        print(f"Época {epoch+1:>2}/{EPOCHS} | "
              f"Loss: {train_loss:.4f} | "
              f"Train: {train_acc:.1f}% | "
              f"Test: {test_acc:.1f}% | "
              f"{elapsed:.1f}s{marker}")

    print(f"\nEntrenamiento finalizado. Mejor precisión en Test: {best_acc:.1f}%")


if __name__ == '__main__':
    train()
