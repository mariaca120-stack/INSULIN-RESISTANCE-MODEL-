"""Rutas y constantes que usan todos los scripts (todas las rutas son relativas a la carpeta del repositorio)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA = ROOT / "data" / "processed"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
for _d in (DATA_RAW, DATA, MODELS, RESULTS, FIGURES):
    _d.mkdir(parents=True, exist_ok=True)

SEED = 42
TARGET = "RESISTENCIA_INSULINA"

# archivos de NHANES 2021-2023
NHANES_URL = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/{modulo}.xpt"
NHANES_MODULOS = ["DEMO_L", "BMX_L", "BPXO_L", "GLU_L", "INS_L", "HDL_L", "TRIGLY_L", "TCHOL_L", "HSCRP_L",
                  "FERTIN_L", "BIOPRO_L", "VID_L", "BPQ_L", "DIQ_L", "DR1TOT_L", "PAQ_L", "SMQ_L", "ALQ_L", "WHQ_L"]

# TabPFN requiere un token gratuito: defínelo en TABPFN_TOKEN o guárdalo en un archivo .tabpfn_token
TABPFN_TOKEN_FILE = ROOT / ".tabpfn_token"


def configurar_token_tabpfn():
    if "TABPFN_TOKEN" not in os.environ and TABPFN_TOKEN_FILE.exists():
        os.environ["TABPFN_TOKEN"] = TABPFN_TOKEN_FILE.read_text().strip()


def dispositivo_tabpfn():
    """GPU if available (Apple 'mps' or CUDA), otherwise CPU."""
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"
