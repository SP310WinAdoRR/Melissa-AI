"""
model.py — Arquitectura CNN para clasificación de acordes de guitarra.

Fusiona el espectrograma CQT (1, 84, 43) con el flujo auxiliar
Chroma + Energía Armónica (2, 12, 43) en una entrada unificada (3, 84, 43).

Salida: 25 clases (12 mayores, 12 menores, 1 sin acorde).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChordCNN(nn.Module):
    def __init__(self, num_classes=25):
        super(ChordCNN, self).__init__()

        # Fusión CQT + Chroma: el aux_stream se interpola y concatena con el CQT
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.pool = nn.MaxPool2d(kernel_size=2)

        # Extracción de progresiones armónicas y filtrado de ruido
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        # Extracción de características de alto nivel
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, spec, aux):
        """
        Args:
            spec: Tensor CQT (batch, 1, 84, 43)
            aux:  Tensor Chroma+Energía (batch, 2, 12, 43)
        Returns:
            Logits de forma (batch, 25)
        """
        aux_upsampled = F.interpolate(aux, size=(84, 43), mode='bilinear', align_corners=False)
        x = torch.cat((spec, aux_upsampled), dim=1)

        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = self.pool(x)

        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        x = self.pool(x)

        x = F.leaky_relu(self.bn4(self.conv4(x)))

        x = self.gap(x)
        x = torch.flatten(x, 1)

        x = self.dropout(x)
        x = F.leaky_relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x
