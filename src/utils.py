"""
Utility functions for the Smart Agriculture Decision Support System
"""
import os
import json
import logging
from pathlib import Path
from datetime import datetime


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
def setup_logging(log_dir='results', level=logging.INFO):
    """Configure root logger to write to console and a rotating log file."""
    Path(log_dir).mkdir(exist_ok=True)
    log_file = Path(log_dir) / 'app.log'

    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding='utf-8'),
        ],
    )
    return logging.getLogger('agriculture_dss')


# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------
def create_directories():
    """Ensure all required project directories exist."""
    for directory in ['data', 'models', 'results']:
        Path(directory).mkdir(exist_ok=True)


def get_project_root() -> Path:
    """Return the project root directory (parent of src/)."""
    return Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Metrics persistence
# ---------------------------------------------------------------------------
def save_metrics(metrics_dict, filename='results/metrics.json'):
    """Persist model metrics to a JSON file with a timestamp."""
    Path('results').mkdir(exist_ok=True)
    metrics_dict['_saved_at'] = datetime.now().isoformat()
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(metrics_dict, f, indent=4, default=str)
    print(f"Metrics saved to {filename}")


def load_metrics(filename='results/metrics.json'):
    """Load model metrics from a JSON file. Returns empty dict if not found."""
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------
def format_metrics_display(metrics: dict) -> str:
    """Format a metrics dictionary into a human-readable string."""
    lines = []
    for key, value in metrics.items():
        if key.startswith('_'):
            continue  # skip internal keys like _saved_at
        if isinstance(value, float):
            lines.append(f"{key}: {value:.4f}")
        else:
            lines.append(f"{key}: {value}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------
# Valid ranges for each input feature (min, max)
FEATURE_RANGES = {
    'Nitrogen':    (0, 140),
    'Phosphorus':  (5, 145),
    'Potassium':   (5, 205),
    'Temperature': (8.8, 43.7),
    'Humidity':    (14, 99),
    'pH':          (3.5, 9.9),
    'Rainfall':    (20, 298),
}


def validate_input(field_name: str, value: float) -> tuple[bool, str]:
    """Validate a single input value against known agronomic ranges.

    Returns:
        (is_valid, warning_message)
        warning_message is empty string when valid.
    """
    if field_name not in FEATURE_RANGES:
        return True, ''  # unknown field — pass through

    lo, hi = FEATURE_RANGES[field_name]
    if value < lo or value > hi:
        return False, (
            f"{field_name} value {value} is outside the expected range "
            f"[{lo}, {hi}]. Predictions may be unreliable."
        )
    return True, ''
