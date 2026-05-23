
"""
GuitarSet Dataset Downloader & Extractor
-----------------------------------------
Este script automatiza la descarga y extracción de la última versión disponible del dataset
GuitarSet desde Zenodo (https://zenodo.org/records/3371780).

Permite elegir qué componentes descargar mediante un diccionario de configuración, sin interfaz.
Usa únicamente la biblioteca estándar de Python (sin dependencias externas).
"""

import os
import sys
import json
import zipfile
import urllib.request
from pathlib import Path

# ==============================================================================
# --- CONFIGURACIÓN DE DESCARGA (Modifica manualmente True o False) ---
# ==============================================================================
# True  = Descargar y extraer el componente.
# False = Omitir y no descargar.
DOWNLOAD_CONFIG = {
    "annotation": True,                       # Anotaciones en formato .jams (~39 MB)
    "audio_mono-pickup_mix": True,             # Audio mezclas mono de pastilla (~683 MB)
    "audio_mono-mic": True,                  # Audio de micrófono mono (~657 MB)
    "audio_hex-pickup_original": True,       # Audio pastilla hexafónica original (~3.2 GB)
    "audio_hex-pickup_debleeded": True       # Audio pastilla hexafónica sin sangrado (~3.6 GB)
}

# Borrar el archivo .zip original tras extraerlo correctamente para ahorrar espacio
ELIMINAR_ZIP_TRAS_EXTRACCION = True
# ==============================================================================

# URL base de la API de Zenodo para resolver la última versión del registro
ZENODO_RECORD_ID = "3371780"
API_LATEST_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}/versions/latest"


def get_latest_dataset_metadata():
    """
    Consulta la API de Zenodo para obtener la información de la última versión
    disponible del dataset y los enlaces directos de descarga de los archivos.
    """
    print("Consultando la API de Zenodo para obtener la última versión disponible...")
    req = urllib.request.Request(
        API_LATEST_URL,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    )
    try:
        with urllib.request.urlopen(req) as response:
            metadata = json.loads(response.read().decode("utf-8"))
            record_id = metadata.get("id", ZENODO_RECORD_ID)
            version = metadata.get("metadata", {}).get("version", "desconocida")
            files_info = metadata.get("files", [])
            
            print(f"Versión encontrada: {version} (Registro ID: {record_id})")
            return record_id, version, files_info
    except Exception as e:
        print(f"Error al consultar la API de Zenodo: {e}")
        print("Se intentará proceder con los valores por defecto del registro original.")
        return ZENODO_RECORD_ID, "1.1.0", []


def download_file(url, dest_path, file_size=None):
    """
    Descarga un archivo desde una URL mostrando una barra de progreso en consola.
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_dest = dest_path.with_suffix(".tmp")
    
    print(f"Descargando: {dest_path.name}")
    print(f"Desde: {url}")
    
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    )
    
    try:
        with urllib.request.urlopen(req) as response:
            if not file_size:
                file_size = int(response.headers.get("Content-Length", 0))
            
            chunk_size = 1024 * 1024  # Buffer de 1 MB
            downloaded = 0
            
            with open(temp_dest, "wb") as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    if file_size > 0:
                        percent = (downloaded / file_size) * 100
                        mb_downloaded = downloaded / (1024 * 1024)
                        mb_total = file_size / (1024 * 1024)
                        
                        bar_length = 30
                        filled_length = int(bar_length * downloaded // file_size)
                        bar = "=" * filled_length + "-" * (bar_length - filled_length)
                        
                        sys.stdout.write(
                            f"\r   [{bar}] {percent:.1f}% ({mb_downloaded:.1f} MB / {mb_total:.1f} MB)"
                        )
                    else:
                        mb_downloaded = downloaded / (1024 * 1024)
                        sys.stdout.write(f"\r   Descargado: {mb_downloaded:.1f} MB")
                    sys.stdout.flush()
            
            # Renombrar el archivo temporal al nombre final tras completar con éxito
            if os.path.exists(dest_path):
                os.remove(dest_path)
            os.rename(temp_dest, dest_path)
            print("\n ¡Descarga finalizada con éxito!")
            return True
            
    except Exception as e:
        if os.path.exists(temp_dest):
            os.remove(temp_dest)
        print(f"\nError al descargar {dest_path.name}: {e}")
        return False


def extract_zip(zip_path, extract_dir, config_key):
    """
    Extrae un archivo ZIP clasificando todos los archivos de forma ordenada
    dentro de su subcarpeta correspondiente (extract_dir / config_key).
    Si el ZIP contiene una carpeta raíz con el mismo nombre, elimina ese nivel
    de anidamiento para que los archivos no queden duplicados bajo otra subcarpeta.
    """
    dest_folder = Path(extract_dir) / config_key
    dest_folder.mkdir(parents=True, exist_ok=True)
    
    print(f"Extrayendo {zip_path.name} en {dest_folder}...")
    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            members = zip_ref.infolist()
            # Filtrar solo archivos válidos (descartar directorios puros)
            file_members = [m for m in members if not m.is_dir() and not m.filename.endswith('/')]
            total_files = len(file_members)
            
            if total_files == 0:
                print("   El archivo ZIP está vacío.")
                return True
                
            # Determinar si hay un directorio raíz común en el ZIP
            common_prefix = ""
            first_parts = file_members[0].filename.split('/')
            if len(first_parts) > 1:
                candidate = first_parts[0] + '/'
                if all(m.filename.startswith(candidate) for m in file_members):
                    common_prefix = candidate
            
            for i, member in enumerate(file_members, 1):
                # Eliminar el prefijo raíz si existe
                rel_path = member.filename
                if common_prefix and rel_path.startswith(common_prefix):
                    rel_path = rel_path[len(common_prefix):]
                
                rel_path = rel_path.lstrip('/')
                if not rel_path:
                    continue
                    
                target_path = dest_folder / rel_path
                target_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Extraer el archivo leyendo en bloques de 1 MB para eficiencia de memoria
                with zip_ref.open(member) as source, open(target_path, "wb") as target:
                    while True:
                        chunk = source.read(1024 * 1024)
                        if not chunk:
                            break
                        target.write(chunk)
                
                if i % max(1, total_files // 20) == 0 or i == total_files:
                    percent = (i / total_files) * 100
                    sys.stdout.write(f"\r   Progreso de extracción: {percent:.1f}% ({i}/{total_files} archivos)")
                    sys.stdout.flush()
            print(f"\n   ¡Extracción completada en: {dest_folder}!")
            return True
    except Exception as e:
        print(f"\nError al extraer {zip_path.name}: {e}")
        return False


def main():
    # Obtener el directorio donde reside este script de forma dinámica
    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    
    print("====================================================================")
    print("               GuitarSet Dataset Downloader                         ")
    print("====================================================================")
    
    # 1. Obtener metadatos actualizados de Zenodo
    record_id, version, files_info = get_latest_dataset_metadata()
    
    # Definir la ruta de destino base como GuitarSet/<record_id>
    # Dado que el script está en GuitarSet/, apuntamos a script_dir / record_id
    base_dest_dir = script_dir / str(record_id)
    base_dest_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Carpeta de destino del dataset: {base_dest_dir}")
    print("--------------------------------------------------------------------")
    
    # Si la API falló o no devolvió archivos, construimos la lista estándar manualmente
    if not files_info:
        print("Usando enlaces y tamaños predeterminados para la versión 1.1.0...")
        files_info = [
            {
                "key": "annotation.zip",
                "size": 39132574,
                "links": {"self": f"https://zenodo.org/api/records/{record_id}/files/annotation.zip/content"}
            },
            {
                "key": "audio_mono-pickup_mix.zip",
                "size": 683145360,
                "links": {"self": f"https://zenodo.org/api/records/{record_id}/files/audio_mono-pickup_mix.zip/content"}
            },
            {
                "key": "audio_mono-mic.zip",
                "size": 656927981,
                "links": {"self": f"https://zenodo.org/api/records/{record_id}/files/audio_mono-mic.zip/content"}
            },
            {
                "key": "audio_hex-pickup_original.zip",
                "size": 3210735945,
                "links": {"self": f"https://zenodo.org/api/records/{record_id}/files/audio_hex-pickup_original.zip/content"}
            },
            {
                "key": "audio_hex-pickup_debleeded.zip",
                "size": 3607349578,
                "links": {"self": f"https://zenodo.org/api/records/{record_id}/files/audio_hex-pickup_debleeded.zip/content"}
            }
        ]
    
    # Procesar cada uno de los archivos según la configuración
    for file_item in files_info:
        file_key = file_item.get("key", "")
        file_size = file_item.get("size")
        download_url = file_item.get("links", {}).get("self")
        
        # Obtener la clave de configuración sin el ".zip"
        config_key = file_key.replace(".zip", "")
        
        # Validar si el usuario quiere descargar este componente
        should_download = DOWNLOAD_CONFIG.get(config_key, False) or DOWNLOAD_CONFIG.get(file_key, False)
        
        if not should_download:
            print(f"Componente '{config_key}' marcado como False en CONFIG. Omitiendo...")
            continue
            
        print(f"Procesando componente: {config_key}")
        
        # Comprobar si el contenido ya fue extraído para evitar descargas innecesarias
        extracted_folder_path = base_dest_dir / config_key
        if extracted_folder_path.exists() and any(extracted_folder_path.iterdir()):
            print(f"El componente ya parece estar extraído en: {extracted_folder_path}")
            print("Omitiendo descarga para ahorrar ancho de banda.")
            continue
            
        # Ruta local temporal para guardar el zip
        local_zip_path = base_dest_dir / file_key
        
        # Descargar el archivo ZIP si no existe localmente
        download_success = True
        if not local_zip_path.exists():
            download_success = download_file(download_url, local_zip_path, file_size)
            
        if download_success:
            # Extraer el archivo zip clasificando el contenido en su subcarpeta
            extraction_success = extract_zip(local_zip_path, base_dest_dir, config_key)
            
            # Limpieza del zip para ahorrar espacio si fue configurado y la extracción fue exitosa
            if extraction_success and ELIMINAR_ZIP_TRAS_EXTRACCION:
                try:
                    os.remove(local_zip_path)
                    print(f"   Cleaned up temporary ZIP: {local_zip_path.name}")
                except Exception as e:
                    print(f"   No se pudo borrar el archivo zip temporal: {e}")
        else:
            print(f"Falló el procesamiento del componente {config_key} debido a un error de descarga.")

    print("\n====================================================================")
    print("¡Proceso finalizado!")
    print(f"El dataset está listo en: {base_dest_dir}")
    print("Puedes modificar el diccionario DOWNLOAD_CONFIG en el código de este")
    print("script para habilitar/deshabilitar componentes en el futuro.")
    print("====================================================================")


if __name__ == "__main__":
    main()
