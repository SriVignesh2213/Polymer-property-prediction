"""Application configuration and constants."""

from __future__ import annotations

from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = APP_DIR / "artifacts"
TRAIN_DEFAULT_PATH = Path(r"D:\Amalgam\train.csv")
TEST_DEFAULT_PATH = Path(r"D:\Amalgam\test.csv")
SUBMISSION_PATH = APP_DIR / "submission.csv"

TARGET_LABELS = {
    "tg": "Glass transition temperature (Tg)",
    "egc": "Electrochemical bandgap (Egc)",
    "egb": "Optoelectronic bandgap (Egb)",
    "nc": "Refractive index (nc)",
    "eps": "Dielectric constant (eps)",
    "ei": "Ionization energy (Ei)",
    "eea": "Electron affinity (Eea)",
}

PHYSICAL_BOUNDS = {
    "tg": (-120.0, 520.0),
    "egc": (0.10, 10.50),
    "egb": (0.10, 10.50),
    "nc": (1.35, 2.95),
    "eps": (1.60, 15.00),
    "ei": (3.50, 11.50),
    "eea": (0.10, 6.50),
}

ACCENT = "#2aa198"
PHYSICS_LAYERS = ["Level 1: tg and egc", "Level 2: minority targets", "Level 3: eps Maxwell transfer"]
RANDOM_SEED = 42