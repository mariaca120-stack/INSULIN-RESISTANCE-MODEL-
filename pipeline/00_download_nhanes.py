"""Paso 00: descarga los archivos de NHANES 2021-2023 desde el sitio de los CDC a data/raw/."""
import urllib.request

from config import DATA_RAW, NHANES_MODULOS, NHANES_URL

for modulo in NHANES_MODULOS:
    destino = DATA_RAW / f"{modulo}.xpt"
    if destino.exists() and destino.stat().st_size > 0:
        print(f"{modulo}: already downloaded")
        continue
    url = NHANES_URL.format(modulo=modulo)
    print(f"{modulo}: downloading {url}", flush=True)
    peticion = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research code)"})
    with urllib.request.urlopen(peticion, timeout=300) as r, open(destino, "wb") as f:
        f.write(r.read())
print(f"Listo: {len(NHANES_MODULOS)} módulos en {DATA_RAW}")
