"""
model.py — Arquitectura CNN 2D para detección de acordes de guitarra acústica.

Entrada: Espectrograma CQT de 0.5 segundos con dimensiones (1, 84, 43)
         - 1 canal (escala de grises / magnitud dB)
         - 84 bins de frecuencia (C2 a B8, 12 bins/octava × 7 octavas)
         - 43 frames temporales (~0.5s con hop_length=512 a 44100 Hz)

Salida: Vector de 25 logits (probabilidades sin normalizar) correspondientes a:
        - 12 acordes mayores (C, C#, D, ..., B)
        - 12 acordes menores (Cm, C#m, Dm, ..., Bm)
        - 1 clase "sin acorde" (N)
"""

# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn as nn


class ChordCNN(nn.Module):
    """
    Red Neuronal Convolucional 2D para clasificación de acordes.

    Arquitectura:
        - 3 bloques convolucionales (Conv2d → BatchNorm → ReLU → MaxPool)
        - 1 capa de aplanamiento (Flatten)
        - 2 capas lineales (Fully Connected) con Dropout
        - Salida: 25 clases
    """

    def __init__(self, num_classes=25, dropout_rate=0.3):
        super(ChordCNN, self).__init__()

        # ─── Bloque Convolucional 1 ───
        # Entrada: (batch, 1, 84, 43)
        # Salida:  (batch, 32, 42, 21)
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        # ─── Bloque Convolucional 2 ───
        # Entrada: (batch, 32, 42, 21)
        # Salida:  (batch, 64, 21, 10)
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        # ─── Bloque Convolucional 3 ───
        # Entrada: (batch, 64, 21, 10)
        # Salida:  (batch, 128, 10, 5)
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        # ─── Clasificador (Fully Connected) ───
        # Después de conv_block3: (batch, 128, 10, 5) → aplanado: 128 * 10 * 5 = 6400
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 10 * 5, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        """
        Propagación hacia adelante.

        Args:
            x: Tensor de forma (batch_size, 1, 84, 43) — espectrograma CQT

        Returns:
            Tensor de forma (batch_size, 25) — logits por clase de acorde
        """
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.classifier(x)
        return x


# =====================================================================
# TEST LOCAL: Verificar que las dimensiones fluyen correctamente
# =====================================================================

if __name__ == "__main__":
    print("🔬 Verificando arquitectura ChordCNN...")

    model = ChordCNN(num_classes=25)

    # Crear un tensor sintético con las dimensiones exactas de nuestro dataset
    # Batch=4, Canales=1, Frecuencia=84, Tiempo=43
    dummy_input = torch.randn(4, 1, 84, 43)

    print(f"\n    Entrada: {dummy_input.shape}")
    output = model(dummy_input)
    print(f"    Salida:  {output.shape}")

    # Mostrar resumen de parámetros entrenables
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n    Parámetros totales entrenables: {total_params:,}")

    # Mostrar la arquitectura completa
    print(f"\n    Arquitectura completa:\n{model}")
    print("\n🎉 ¡Arquitectura verificada con éxito!")
