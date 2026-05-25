"""
config.py — Hiperparámetros y configuración del entrenamiento.

Centraliza todos los parámetros ajustables del modelo en un solo lugar
para facilitar la experimentación y la reproducibilidad.
"""

import torch
import torch.nn as nn
import torch.optim as optim

from model import ChordCNN


# ==========================================================
# DISPOSITIVO DE CÓMPUTO (GPU / CPU)
# ==========================================================
# PyTorch detecta automáticamente si hay una tarjeta gráfica NVIDIA
# con drivers CUDA instalados. Si existe, entrena en GPU (20x más rápido).
# Si no, usa la CPU normalmente.
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f"🔥 GPU detectada: {torch.cuda.get_device_name(0)}")
    print(f"   Memoria disponible: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
else:
    DEVICE = torch.device("cpu")
    print("⚠️  No se detectó GPU. Entrenando en CPU (será más lento).")


# ==========================================================
# HIPERPARÁMETROS DEL ENTRENAMIENTO
# ==========================================================
BATCH_SIZE = 64       # Fragmentos por lote (64 es un buen balance velocidad/estabilidad)
EPOCHS = 30           # Cantidad de veces que la red verá todo el dataset
LEARNING_RATE = 0.001 # Velocidad de aprendizaje del optimizador Adam
NUM_CLASSES = 25      # 12 mayores + 12 menores + 1 sin acorde
SEED = 42             # Semilla para reproducibilidad


# ==========================================================
# INSTANCIACIÓN DEL MODELO, PÉRDIDA Y OPTIMIZADOR
# ==========================================================
def build_model():
    """
    Construye e inicializa el modelo, la función de pérdida y el optimizador.

    Returns:
        model:     Red neuronal ChordCNN enviada al dispositivo (GPU/CPU)
        criterion: Función de pérdida CrossEntropyLoss
        optimizer: Optimizador Adam con los parámetros del modelo
    """
    # Fijar semilla para reproducibilidad
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)

    # 1. Modelo → enviarlo al dispositivo
    model = ChordCNN(num_classes=NUM_CLASSES).to(DEVICE)

    # 2. Función de Pérdida: CrossEntropyLoss
    #    Combina LogSoftmax + NLLLoss internamente.
    #    Recibe logits crudos (sin Softmax) y etiquetas enteras.
    criterion = nn.CrossEntropyLoss()

    # 3. Optimizador: Adam
    #    Ajusta los pesos de la red usando gradientes adaptativos.
    #    lr=0.001 es el valor por defecto recomendado por el paper original de Adam.
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    return model, criterion, optimizer
