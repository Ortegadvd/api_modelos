import os
import gdown

# Enlace directo de Google Drive para gdown
url = 'https://drive.google.com/uc?id=1cdv4LmPVxgbvybpj-DJwj62-btTkWZzh'
output = 'api_modelos/modelo_descripcion.pkl'

# Crea la carpeta si no existe
os.makedirs(os.path.dirname(output), exist_ok=True)

if not os.path.exists(output):
    print("Descargando modelo...")
    gdown.download(url, output, quiet=False)
    print("Descarga completada.")
else:
    print("El modelo ya existe.")