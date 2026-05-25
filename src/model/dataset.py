"""
dataset.py — Cargador de datos para el modelo de clasificación de acordes.

Lee los archivos .pt preprocesados desde disco y los empaqueta en una clase
Dataset compatible con el DataLoader de PyTorch.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader

# ==========================================================
# RUTA GLOBAL AL DIRECTORIO DE DATOS PREPROCESADOS
# Estructura esperada:
#   PROCESSED_PATH/
#   ├── train/
#   │   └── train_data.pt
#   └── test/
#       └── test_data.pt
# ==========================================================
PATH = os.path.join(os.path.dirname(__file__), "../../GuitarSet/3371780/processed")


class MelissaDataset(Dataset):
    def __init__(self, cache_file_path):
        """
        Carga un archivo .pt preprocesado a la memoria RAM.

        Args:
            cache_file_path (str): Ruta absoluta o relativa al archivo .pt
        """
        data = torch.load(cache_file_path, weights_only=False)

        self.samples = data['samples']
        self.labels = data['labels']
        self.metadata = data['metadata']

    def __len__(self):
        """Devuelve la cantidad total de fragmentos de audio."""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Extrae un fragmento individual para el entrenamiento.

        Returns:
            spec:  Tensor CQT (1, 84, 43)
            aux:   Tensor Chroma + Energía (2, 12, 43)
            label: Tensor entero con la clase del acorde
        """
        spec = self.samples[idx]['spectrogram']
        aux = self.samples[idx]['aux_stream']
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return spec, aux, label


def get_loaders(batch_size=64):
    """
    Crea los DataLoaders de Train y Test listos para inyectar al modelo.

    Args:
        batch_size (int): Cantidad de fragmentos por lote.

    Returns:
        train_loader, test_loader: DataLoaders de PyTorch.
    """
    train_path = os.path.join(PATH, 'train', 'train_data.pt')
    test_path = os.path.join(PATH, 'test', 'test_data.pt')

    train_dataset = MelissaDataset(train_path)
    test_dataset = MelissaDataset(test_path)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

