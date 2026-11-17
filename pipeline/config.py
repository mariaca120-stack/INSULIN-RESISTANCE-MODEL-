"""Shared paths and constants. Every script in this folder imports this module.

All paths are relative to the repository root, so the code runs from any computer
without Google Drive or any other external storage.

    data/raw/        NHANES .xpt files downloaded from the CDC (step 00)      [not versioned]
    data/processed/  analysis datasets (steps 01-02)                          [versioned]
    models/          trained models (step 03)                                 [versioned]
    results/         tables and arrays produced by the analysis               [versioned]
    figures/         figures                                                  [not versioned]
"""
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

# NHANES 2021-2023 cycle ("_L" files)
NHANES_URL = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/{modulo}.xpt"
NHANES_MODULOS = ["DEMO_L", "BMX_L", "BPXO_L", "GLU_L", "INS_L", "HDL_L", "TRIGLY_L", "TCHOL_L", "HSCRP_L",
                  "FERTIN_L", "BIOPRO_L", "VID_L", "BPQ_L", "DIQ_L", "DR1TOT_L", "PAQ_L", "SMQ_L", "ALQ_L", "WHQ_L"]

# TabPFN (v2.5) needs a free token from https://ux.priorlabs.ai . Provide it as the environment variable
# TABPFN_TOKEN or in a file named .tabpfn_token at the repository root (the file is git-ignored).
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
