import os
import json
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import librosa
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
from torch.utils.data import Dataset, DataLoader

# =====================================================================
# 1. MAPEADOR DE ACORDES A 25 CLASES
# =====================================================================

# Notas estándar de la escala cromática en sostenidos
CHROMATIC_NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Mapeo para normalizar notas bemoles a sostenidos
FLAT_TO_SHARP = {
    'Db': 'C#', 'Eb': 'D#', 'Gb': 'F#', 'Ab': 'G#', 'Bb': 'A#'
}

def get_chord_mapping():
    """
    Crea un diccionario inverso para mapear nombres de clases a índices enteros.
    Clases:
      0 a 11: Mayores (C, C#, D, ..., B)
      12 a 23: Menores (Cm, C#m, Dm, ..., Bm)
      24: Sin Acorde / Silencio (N)
    """
    mapping = {}
    
    # 12 acordes mayores
    for idx, note in enumerate(CHROMATIC_NOTES):
        mapping[note] = idx
        
    # 12 acordes menores
    for idx, note in enumerate(CHROMATIC_NOTES):
        mapping[note + 'm'] = idx + 12
        
    # Clase para 'Sin acorde' (No Chord)
    mapping['N'] = 24
    
    # Lista invertida para mapear de índice a string legible
    idx_to_chord = CHROMATIC_NOTES + [note + 'm' for note in CHROMATIC_NOTES] + ['N']
    
    return mapping, idx_to_chord

CHORD_MAP, IDX_TO_CHORD = get_chord_mapping()

def simplify_jams_chord(chord_str):
    """
    Simplifica un acorde del formato JAMS (ej. 'C:maj', 'A:min7', 'Db:7') 
    a una de nuestras 25 clases simplificadas.
    """
    if not chord_str or chord_str == 'N' or chord_str.startswith('silence'):
        return 'N'
        
    # Separar la raíz de la calidad del acorde
    parts = chord_str.split(':')
    root = parts[0]
    
    # Normalizar bemoles a sostenidos
    root = FLAT_TO_SHARP.get(root, root)
    
    # Si la raíz no está en la escala cromática, la tratamos como sin acorde
    if root not in CHROMATIC_NOTES:
        return 'N'
        
    # Determinar si es mayor o menor
    quality = parts[1] if len(parts) > 1 else 'maj'
    
    if 'min' in quality or 'm' in quality:
        return root + 'm'
    else:
        return root

# =====================================================================
# 2. CLASE GUITARSET DATASET EN PYTORCH
# =====================================================================

class GuitarSetDataset(Dataset):
    def __init__(self, base_dir, track_names, audio_type="audio_mono-pickup_mix", 
                 sr=44100, frame_duration=0.5, overlap=0.5):
        """
        Dataset de PyTorch para GuitarSet que realiza pre-cómputo y segmentación en RAM.
        
        Args:
            base_dir (str): Ruta base del dataset (ej. "GuitarSet/3371780")
            track_names (list): Lista de nombres de pistas a incluir (para Train/Val/Test splits)
            audio_type (str): "audio_mono-pickup_mix" o "audio_mono-mic"
            sr (int): Tasa de muestreo
            frame_duration (float): Duración de cada ventana/frame de análisis (0.5s por defecto)
            overlap (float): Porcentaje de solapamiento entre ventanas (50% por defecto)
        """
        self.base_dir = base_dir
        self.track_names = track_names
        self.audio_type = audio_type
        self.sr = sr
        self.hop_length = 512
        self.fmin = librosa.note_to_hz('C2')
        self.n_bins = 84
        
        # Parámetros de segmentación
        # Un frame de CQT tiene una duración de hop_length / sr segundos
        self.frame_len_sec = self.hop_length / self.sr
        self.segment_size = int(frame_duration / self.frame_len_sec) # ~43 frames
        self.stride = int(self.segment_size * (1 - overlap))         # ~21 frames
        
        # Listas para almacenar los datos segmentados en RAM
        self.samples = []
        self.labels = []
        
        print(f"\n[Dataset] Cargando e indexando {len(track_names)} pistas desde '{audio_type}'...")
        self._load_and_segment_dataset()
        print(f"[Dataset] ¡Carga completada! Total de muestras segmentadas de 0.5s: {len(self.samples)}")

    def _load_and_segment_dataset(self):
        """
        Itera sobre todas las pistas, extrae sus CQTs completas, lee las anotaciones JAMS,
        y divide el audio y etiquetas en segmentos de 0.5 segundos.
        """
        for i, track in enumerate(self.track_names):
            audio_file = f"{track}_mix.wav" if "mix" in self.audio_type else f"{track}_mic.wav"
            audio_path = os.path.join(self.base_dir, self.audio_type, audio_file)
            jams_path = os.path.join(self.base_dir, "annotation", f"{track}.jams")
            
            if not os.path.exists(audio_path) or not os.path.exists(jams_path):
                print(f"      ⚠️ Advertencia: Omitiendo pista {track} por archivos faltantes.")
                continue
                
            # 1. Cargar acordes desde JAMS
            with open(jams_path, 'r', encoding='utf-8') as f:
                jams_data = json.load(f)
                
            chords_timeline = []
            for annotation in jams_data.get('annotations', []):
                if annotation.get('namespace') == 'chord':
                    for observation in annotation.get('data', []):
                        chords_timeline.append({
                            'start': observation.get('time'),
                            'end': observation.get('time') + observation.get('duration'),
                            'chord': simplify_jams_chord(observation.get('value'))
                        })
                    break
            
            # 2. Cargar audio y calcular CQT completa para la pista
            y, _ = librosa.load(audio_path, sr=self.sr)
            cqt = librosa.cqt(y, sr=self.sr, fmin=self.fmin, n_bins=self.n_bins, 
                               bins_per_octave=12, hop_length=self.hop_length)
            
            # Convertir CQT a escala logarítmica de decibelios
            cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
            
            # 3. Segmentación en ventanas de 0.5s
            num_cqt_frames = cqt_db.shape[1]
            
            # Deslizar ventana temporal
            start_frame = 0
            while start_frame + self.segment_size <= num_cqt_frames:
                end_frame = start_frame + self.segment_size
                segment_cqt = cqt_db[:, start_frame:end_frame]
                
                # Obtener el tiempo del frame central del segmento
                center_frame = start_frame + (self.segment_size // 2)
                center_time = center_frame * self.frame_len_sec
                
                # Buscar qué acorde corresponde en la línea de tiempo
                active_chord = 'N'
                for c in chords_timeline:
                    if c['start'] <= center_time <= c['end']:
                        active_chord = c['chord']
                        break
                
                # Convertir etiqueta de acorde string a entero
                label_idx = CHORD_MAP.get(active_chord, CHORD_MAP['N'])
                
                # Guardar el fragmento espectral y su etiqueta
                # Convertimos a tensor float32 para PyTorch
                self.samples.append(torch.tensor(segment_cqt, dtype=torch.float32))
                self.labels.append(label_idx)
                
                # Avanzar con traslape
                start_frame += self.stride
                
            if (i + 1) % 10 == 0 or (i + 1) == len(self.track_names):
                print(f"      * Procesadas {i + 1}/{len(self.track_names)} pistas...")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Retorna el espectrograma 2D CQT de 0.5s y su etiqueta de acorde
        # Añadimos una dimensión de canal (1, Frecuencias, Tiempo) para que actúe como imagen 2D en CNN
        sample = self.samples[idx].unsqueeze(0)
        label = self.labels[idx]
        return sample, label

# =====================================================================
# 3. PRUEBA DE FUNCIONAMIENTO LOCAL
# =====================================================================

if __name__ == "__main__":
    # Test rápido de dataset con 3 pistas de ejemplo
    test_tracks = [
        "00_BN3-119-G_comp",
        "00_BN3-154-E_solo",
        "00_Funk1-97-C_solo"
    ]
    
    print("🚀 Probando creación de la clase GuitarSetDataset...")
    dataset = GuitarSetDataset(
        base_dir="GuitarSet/3371780",
        track_names=test_tracks,
        audio_type="audio_mono-pickup_mix"
    )
    
    print("\n[Test] Propiedades de una muestra extraída:")
    sample_tensor, label_idx = dataset[0]
    print(f"    - Tensor del Espectrograma shape (C, F, T): {sample_tensor.shape}")
    print(f"    - Índice del acorde clasificado: {label_idx}")
    print(f"    - Acorde real: {IDX_TO_CHORD[label_idx]}")
    
    # Probar DataLoader de PyTorch
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    batch_samples, batch_labels = next(iter(dataloader))
    print(f"\n[Test] Propiedades de un Batch del DataLoader:")
    print(f"    - Batch de CQTs shape (BatchSize, Channels, FreqBins, TimeFrames): {batch_samples.shape}")
    print(f"    - Batch de etiquetas shape: {batch_labels.shape}")
    print(f"    - Etiquetas del primer batch: {batch_labels.tolist()}")
    print("🎉 ¡Prueba de PyTorch Dataset y DataLoader exitosa!")
