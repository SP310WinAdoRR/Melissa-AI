"""
model.py — Arquitectura CNN para clasificación de acordes de guitarra.

La red fusiona el espectrograma CQT (1, 84, 43) con el flujo auxiliar
Chroma + Energía Armónica (2, 12, 43) en una sola entrada unificada (3, 84, 43),
permitiendo que las capas convolucionales aprendan simultáneamente
la relación entre timbre acústico y armonía teórica.

Salida: 25 clases (12 mayores, 12 menores, 1 sin acorde).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChordCNN(nn.Module):
    def __init__(self, num_classes=25):
        super(ChordCNN, self).__init__()

        # ==============================================================
        # CAPA DE ENTRADA: Fusión CQT + Chroma
        # El aux_stream (2, 12, 43) se interpola a (2, 84, 43) y se
        # concatena con el CQT (1, 84, 43) → entrada unificada (3, 84, 43)
        # ==============================================================

        # Conv2D 32: Detecta patrones de timbre y armonía combinados
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # MaxPooling: Reduce dimensionalidad espacial a la mitad
        self.pool = nn.MaxPool2d(kernel_size=2)

        # 2x Conv2D 64: Extracción de progresiones armónicas y filtrado de ruido
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        # Conv2D 128: Extracción de características de alto nivel
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        # Global Average Pooling: Colapsa cada mapa de características a un solo valor
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Capa densa de 128 neuronas
        self.fc1 = nn.Linear(128, 128)

        # Capa de salida: 25 neuronas (una por clase de acorde)
        # Se entregan logits crudos, ya que CrossEntropyLoss aplica Softmax internamente.
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, spec, aux):
        """
        Args:
            spec: Tensor CQT de forma (batch, 1, 84, 43)
            aux:  Tensor Chroma+Energía de forma (batch, 2, 12, 43)
        Returns:
            Tensor de forma (batch, 25) con las predicciones por clase
        """
        # --- Capa de Entrada: Fusión ---
        # Interpolar el aux_stream de (2, 12, 43) a (2, 84, 43)
        # para que coincida espacialmente con el CQT
        aux_upsampled = F.interpolate(aux, size=(84, 43), mode='bilinear', align_corners=False)

        # Concatenar por el eje de canales: (1+2, 84, 43) = (3, 84, 43)
        x = torch.cat((spec, aux_upsampled), dim=1)

        # --- Conv2D 32 + LeakyReLU ---
        x = F.leaky_relu(self.bn1(self.conv1(x)))

        # --- MaxPooling2D ---
        x = self.pool(x)  # (32, 42, 21)

        # --- 2x Conv2D 64 + LeakyReLU ---
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = F.leaky_relu(self.bn3(self.conv3(x)))

        # --- MaxPooling2D ---
        x = self.pool(x)  # (64, 21, 10)

        # --- Conv2D 128 ---
        x = F.leaky_relu(self.bn4(self.conv4(x)))  # (128, 21, 10)

        # --- Global Average Pooling ---
        x = self.gap(x)            # (128, 1, 1)
        x = torch.flatten(x, 1)    # (128,)

        # --- Densa 128 + LeakyReLU ---
        x = F.leaky_relu(self.fc1(x))

        # --- Densa de Salida (Logits crudos) ---
        x = self.fc2(x)

        return x
