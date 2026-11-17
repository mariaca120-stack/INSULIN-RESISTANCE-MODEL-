"""Paso 01: une los archivos de NHANES por SEQN en una sola tabla (data/processed/nhanes_variables.csv).
La presión arterial y el pulso usan la primera de las tres lecturas."""
import pandas as pd

from config import DATA, DATA_RAW

# variable de NHANES -> variable del estudio, por archivo
VARIABLES = {
    "DEMO_L": {"RIAGENDR": "GENERO", "RIDAGEYR": "EDAD", "RIDRETH3": "ETNIA", "DMDEDUC2": "NIVEL_EDUCATIVO"},
    "BMX_L": {"BMXWT": "PESO", "BMXHT": "ALTURA", "BMXBMI": "IMC", "BMXWAIST": "CINTURA", "BMXARMC": "BRAZO",
              "BMXHIP": "CADERA"},
    "BPXO_L": {"BPXODI1": "PAD", "BPXOSY1": "PAS", "BPXOPLS1": "PULSO"},   # orden usado en los archivos de análisis
    "GLU_L": {"LBXGLU": "GLUCOSA"},
    "INS_L": {"LBXIN": "INSULINA"},
    "HDL_L": {"LBDHDD": "COLESTEROL_HDL"},
    "TRIGLY_L": {"LBXTLG": "TRIGLICERIDOS", "LBDLDL": "COLESTEROL_LDL"},
    "TCHOL_L": {"LBXTC": "COLESTEROL_TOTAL"},
    "HSCRP_L": {"LBXHSCRP": "PROTEINA_C_REACTIVA"},
    "FERTIN_L": {"LBXFER": "FERRITINA"},
    "BIOPRO_L": {"LBXSATSI": "ALT", "LBXSASSI": "AST", "LBDSBUSI": "BUN", "LBXSCR": "CREATININA", "LBXSCK": "CPK",
                 "LBXSIR": "HIERRO", "LBXMAGN": "MAGNESIO"},
    "VID_L": {"LBXVIDMS": "VITAMINA_D"},
    "BPQ_L": {"BPQ020": "DX_HIPERTENSION", "BPQ030": "DX_HTA", "BPQ150": "USO_ANTIHIPERTENSIVOS",
              "BPQ080": "DX_HIPERCOLESTEROLEMIA", "BPQ101D": "USO_HIPOLIPEMIANTES"},
    "DIQ_L": {"DIQ160": "DX_PREDIABETES"},
    "DR1TOT_L": {"DR1TKCAL": "INGESTA_CALORIAS", "DR1TPROT": "INGESTA_PROTEINA", "DR1TCARB": "INGESTA_CARBOHIDRATOS",
                 "DR1TSUGR": "INGESTA_AZUCAR", "DR1TFIBE": "INGESTA_FIBRA", "DR1TTFAT": "INGESTA_GRASAS"},
    "PAQ_L": {"PAD790Q": "FRECUENCIA_ACTIVIDAD_MODERADA", "PAD790U": "UNIDAD_ACTIVIDAD_MODERADA",
              "PAD800": "DURACION_ACTIVIDAD_MODERADA", "PAD810Q": "FRECUENCIA_ACTIVIDAD_VIGOROSA",
              "PAD810U": "UNIDAD_ACTIVIDAD_VIGOROSA", "PAD820": "DURACION_ACTIVIDAD_VIGOROSA",
              "PAD680": "TIEMPO_SENTADO"},
    "SMQ_L": {"SMQ020": "MAS_100_CIGARRILLOS", "SMQ040": "FRECUENCIA_CONSUMO"},
    "ALQ_L": {"ALQ111": "ALCOHOL_VIDA", "ALQ121": "ALCOHOL_12_MESES", "ALQ130": "NUMERO_BEBIDAS"},
    "WHQ_L": {"WHQ070": "INTENTO_PERDIDA_PESO"},
}


def leer(modulo):
    return pd.read_sas(DATA_RAW / f"{modulo}.xpt", format="xport", encoding="utf-8")


base = leer("DEMO_L")[["SEQN"]]
for modulo, columnas in VARIABLES.items():
    tabla = leer(modulo)[["SEQN"] + list(columnas)].rename(columns=columnas)
    assert tabla["SEQN"].is_unique, f"{modulo}: duplicated SEQN"
    base = base.merge(tabla, on="SEQN", how="left")

assert base["SEQN"].is_unique
n_ins_glu = int(base[["INSULINA", "GLUCOSA"]].notna().all(axis=1).sum())
print(f"nhanes_variables.csv: {base.shape[0]} participants x {base.shape[1]} columns")
print(f"Participantes con insulina y glucosa en ayunas: {n_ins_glu} (esperado: 3483)")
assert n_ins_glu == 3483, "Número inesperado de participantes con insulina y glucosa en ayunas"
base.to_csv(DATA / "nhanes_variables.csv", index=False)
