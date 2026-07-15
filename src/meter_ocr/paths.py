from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT_DIR / "config"
ASSETS_DIR = ROOT_DIR / "assets"
TEMPLATE_DIR = ASSETS_DIR / "digit_templates"
OUTPUT_DIR = ROOT_DIR / "output"

ROI_CONFIG = CONFIG_DIR / "rois.json"
COMPASS_CONFIG = CONFIG_DIR / "compass.json"
DIGIT_SLOTS_CONFIG = CONFIG_DIR / "digit_slots.json"
