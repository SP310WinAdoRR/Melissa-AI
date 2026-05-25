"""
test_dataset.py — Prueba unitaria para verificar la estructura de los archivos .pt

Valida existencia, dimensiones, alineación y compatibilidad con DataLoader.
"""

import os
import sys
import random
import torch

sys.path.insert(0, os.path.dirname(__file__))

from dataset import PATH, MelissaDataset, get_loaders


def check_pt_file(name, filepath):
    """Audita un archivo .pt individual."""
    print(f"\n{'='*60}")
    print(f"  AUDITANDO: {name}")
    print(f"{'='*60}")

    if not os.path.exists(filepath):
        print(f"  [FALLO] Archivo no encontrado: {filepath}")
        return False

    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"  Archivo: {filepath}")
    print(f"  Tamano en disco: {size_mb:.2f} MB")

    data = torch.load(filepath, weights_only=False)

    keys = list(data.keys())
    print(f"  Claves: {keys}")
    expected_keys = {'samples', 'labels', 'metadata'}
    if set(keys) != expected_keys:
        print(f"  [FALLO] Claves incorrectas. Esperadas: {expected_keys}")
        return False

    samples = data['samples']
    labels = data['labels']
    metadata = data['metadata']

    aligned = len(samples) == len(labels) == len(metadata)
    print(f"\n  Total de fragmentos: {len(samples)}")
    print(f"  Listas alineadas: {'OK' if aligned else 'FALLO'}")

    if not aligned:
        print(f"  samples={len(samples)}, labels={len(labels)}, metadata={len(metadata)}")
        return False

    if len(samples) == 0:
        print(f"  [FALLO] Archivo vacio (0 fragmentos)")
        return False

    idx = random.randint(0, len(samples) - 1)
    sample = samples[idx]
    label = labels[idx]
    meta = metadata[idx]

    print(f"\n  Muestra aleatoria (indice {idx}):")
    print(f"    Espectrograma CQT: {sample['spectrogram'].shape}  (esperado: [1, 84, 43])")
    print(f"    Flujo Auxiliar:    {sample['aux_stream'].shape}  (esperado: [2, 12, 43])")
    print(f"    Etiqueta:          {label}")
    print(f"    Metadata:")
    for k, v in meta.items():
        print(f"      {k}: {v}")

    spec_ok = sample['spectrogram'].shape == torch.Size([1, 84, 43])
    aux_ok = sample['aux_stream'].shape == torch.Size([2, 12, 43])

    if not spec_ok:
        print(f"  [FALLO] Dimension del espectrograma incorrecta")
    if not aux_ok:
        print(f"  [FALLO] Dimension del flujo auxiliar incorrecta")

    unique_labels = set(labels)
    print(f"\n  Clases unicas: {len(unique_labels)} de 25")
    print(f"  Rango de etiquetas: [{min(labels)}, {max(labels)}]")

    return spec_ok and aux_ok and aligned


def check_dataloader():
    """Verifica que el DataLoader empaquete correctamente los lotes."""
    print(f"\n{'='*60}")
    print(f"  PROBANDO DATALOADER")
    print(f"{'='*60}")

    try:
        train_loader, test_loader = get_loaders(batch_size=16)
        spec, aux, label = next(iter(train_loader))

        print(f"  Batch espectrogramas: {spec.shape}  (esperado: [16, 1, 84, 43])")
        print(f"  Batch auxiliares:     {aux.shape}  (esperado: [16, 2, 12, 43])")
        print(f"  Batch etiquetas:      {label.shape}  (esperado: [16])")
        print(f"  Etiquetas del batch:  {label.tolist()}")
        print(f"\n  [OK] DataLoader funciona correctamente")
        return True
    except Exception as e:
        print(f"  [FALLO] Error en DataLoader: {e}")
        return False


if __name__ == '__main__':
    print(f"Ruta configurada: {os.path.abspath(PATH)}")

    train_file = os.path.join(PATH, 'train', 'train_data.pt')
    test_file = os.path.join(PATH, 'test', 'test_data.pt')

    train_ok = check_pt_file("TRAIN", train_file)
    test_ok = check_pt_file("TEST", test_file)

    if train_ok and test_ok:
        loader_ok = check_dataloader()
    else:
        print("\n  No se puede probar el DataLoader: fallo la auditoria de archivos.")
        loader_ok = False

    print(f"\n{'='*60}")
    print(f"  RESUMEN")
    print(f"{'='*60}")
    print(f"  Train:      {'OK' if train_ok else 'FALLO'}")
    print(f"  Test:       {'OK' if test_ok else 'FALLO'}")
    print(f"  DataLoader: {'OK' if loader_ok else 'FALLO'}")
