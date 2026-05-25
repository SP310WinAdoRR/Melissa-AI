"""
config.py — Hiperparámetros y configuración del entrenamiento.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim

from model import ChordCNN

# Configuración de hardware
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f"GPU detectada: {torch.cuda.get_device_name(0)}")
    print(f"Memoria disponible: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
else:
    DEVICE = torch.device("cpu")
    print("No se detectó GPU. Entrenando en CPU.")

# Hiperparámetros del modelo y entrenamiento
BATCH_SIZE = 64       # Tamaño del lote de datos (batch size)
EPOCHS = 50           # Cantidad total de épocas de entrenamiento
LEARNING_RATE = 0.002 # Tasa de aprendizaje inicial para el optimizador Adam
NUM_CLASSES = 25      # Número de clases de acordes (12 mayores, 12 menores, 1 sin acorde)
SEED = 42             # Semilla aleatoria para asegurar la reproducibilidad
SAVE_EVERY = 5        # Frecuencia de épocas para guardar checkpoints

# Directorio de salida
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def build_model():
    """Construye el modelo, la función de pérdida y el optimizador."""
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)

    model = ChordCNN(num_classes=NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    return model, criterion, optimizer
