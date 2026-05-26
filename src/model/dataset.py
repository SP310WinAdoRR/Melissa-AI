"""
dataset.py — Cargador de datos para el modelo de clasificación de acordes.

Lee los archivos .pt preprocesados desde disco y los empaqueta
en una clase Dataset compatible con el DataLoader de PyTorch.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader

# Ruta al directorio de datos preprocesados
PATH = os.path.join(os.path.dirname(__file__), "../../GuitarSet/processed")


class MelissaDataset(Dataset):
    def __init__(self, cache_file_path):
        """Carga un archivo .pt preprocesado a memoria."""
        data = torch.load(cache_file_path, weights_only=False)
        self.samples = data['samples']
        self.labels = data['labels']
        self.metadata = data['metadata']

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
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
    """Crea los DataLoaders de Train y Test."""
    train_path = os.path.join(PATH, 'train', 'train_data.pt')
    test_path = os.path.join(PATH, 'test', 'test_data.pt')

    train_dataset = MelissaDataset(train_path)
    test_dataset = MelissaDataset(test_path)

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)

    return train_loader, test_loader
