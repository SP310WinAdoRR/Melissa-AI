import os
import json
# pyrefly: ignore [missing-import]
import librosa
# pyrefly: ignore [missing-import]
import numpy as np

def load_chords_from_jams(jams_path):
    """
    Carga un archivo .jams y extrae los acordes junto con sus tiempos de inicio y fin
    utilizando únicamente la librería nativa json de Python.
    """
    print(f"\n[1] Cargando anotaciones de: {os.path.basename(jams_path)}...")
    with open(jams_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    chords = []
    # Buscar la capa (namespace) que corresponde a acordes
    for annotation in data.get('annotations', []):
        if annotation.get('namespace') == 'chord':
            for observation in annotation.get('data', []):
                start_time = observation.get('time')
                duration = observation.get('duration')
                chord_val = observation.get('value')
                chords.append({
                    'start': start_time,
                    'end': start_time + duration,
                    'chord': chord_val
                })
            break  # Una vez encontrada la capa de acordes, salimos
            
    return chords

def explore_audio_and_cqt(audio_path):
    """
    Carga los primeros 10 segundos de un audio de GuitarSet y calcula su CQT.
    """
    print(f"\n[2] Cargando audio (primeros 20s): {os.path.basename(audio_path)}...")
    # Cargamos a la tasa de muestreo nativa del dataset (44100 Hz)
    y, sr = librosa.load(audio_path, sr=44100, duration=20)
    print(f"    - Audio cargado correctamente.")
    print(f"    - Duración en muestras: {len(y)}")
    print(f"    - Tasa de muestreo (Sample Rate): {sr} Hz")
    
    # Calcular la Constant-Q Transform (CQT)
    print("\n[3] Calculando la Transformada Constant-Q (CQT)...")
    # Configuramos CQT para guitarra:
    # fmin = C2 (~65.4 Hz), 12 bins por octava (un bin por semitono), 7 octavas (84 bins totales)
    fmin = librosa.note_to_hz('C2')
    cqt = librosa.cqt(y, sr=sr, fmin=fmin, n_bins=84, bins_per_octave=12, hop_length=512)
    
    # Obtenemos la magnitud del CQT (espectrograma)
    cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
    
    print("    - CQT calculada correctamente.")
    print(f"    - Dimensiones de la matriz CQT (bins de frecuencia, frames temporales): {cqt_db.shape}")
    print(f"      * {cqt_db.shape[0]} bins de frecuencia (representando 84 notas desde C2 en adelante)")
    print(f"      * {cqt_db.shape[1]} frames de tiempo en 10 segundos.")
    
    return cqt_db, sr

def main():
    # Definimos rutas relativas basados en la estructura existente
    base_dir = "GuitarSet/3371780"
    
    # Usaremos una pista de ejemplo (ej. 00_BN3-119-G_comp)
    track_name = "00_BN3-119-G_comp"
    
    jams_path = os.path.join(base_dir, "annotation", f"{track_name}.jams")
    audio_path = os.path.join(base_dir, "audio_mono-pickup_mix", f"{track_name}_mix.wav")
    
    # Validar que los archivos existan
    if not os.path.exists(jams_path) or not os.path.exists(audio_path):
        print("❌ Error: No se encontraron los archivos de ejemplo en las rutas especificadas.")
        print(f"   Por favor, verifica que {jams_path} y {audio_path} existan.")
        return
        
    # 1. Cargar acordes
    chords = load_chords_from_jams(jams_path)
    print(f"    - Se encontraron {len(chords)} cambios de acorde en total.")
    print("    - Primeros 10 acordes:")
    for i, c in enumerate(chords[:10]):
        print(f"      * Acorde {i+1}: {c['chord']:<8} (Desde {c['start']:6.2f}s hasta {c['end']:6.2f}s)")
        
    # 2. Cargar audio y calcular CQT
    cqt_matrix, sr = explore_audio_and_cqt(audio_path)
    
    # 3. Mapear un frame a su acorde como demostración
    hop_length = 512
    # Tomemos el frame 100 de CQT y veamos a qué tiempo y acorde corresponde
    frame_idx = 100
    frame_time = librosa.frames_to_time(frame_idx, sr=sr, hop_length=hop_length)
    
    # Buscar el acorde activo en este frame_time
    active_chord = "N/A"
    for c in chords:
        if c['start'] <= frame_time <= c['end']:
            active_chord = c['chord']
            break
            
    print(f"\n[4] Sincronización de ejemplo:")
    print(f"    - El Frame temporal número {frame_idx} corresponde a {frame_time:.3f} segundos.")
    print(f"    - El acorde que está sonando en ese instante exacto es: {active_chord}")
    print("\n🎉 ¡Exploración completada con éxito! Tu entorno y pipeline base están listos.")

if __name__ == "__main__":
    main()
