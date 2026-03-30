import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

# Define all directories
list_of_dirs = [
    "data",
    "data/raw",
    "data/custom",
    "data/processed",
    "data/processed/images",
    "data/processed/images/train",
    "data/processed/images/val",
    "data/processed/images/test",
    "data/processed/labels",
    "data/processed/labels/train",
    "data/processed/labels/val",
    "data/processed/labels/test",
    "src",
    "notebooks",
    "models",
    "outputs",
    "outputs/sample_predictions",
    "outputs/reports",
    "docs"
]

# Optional: add placeholder files if needed
list_of_files = [
    "src/__init__.py",
    "notebooks/.gitkeep",
    "models/.gitkeep",
    "outputs/.gitkeep",
    "docs/.gitkeep",
    "requirements.txt"
]

# Create directories
for dir_path in list_of_dirs:
    dir_path = Path(dir_path)
    os.makedirs(dir_path, exist_ok=True)
    logging.info(f"Created directory: {dir_path}")

# Create files
for file_path in list_of_files:
    file_path = Path(file_path)
    filedir, filename = os.path.split(file_path)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)

    if (not os.path.exists(file_path)) or (os.path.getsize(file_path) == 0):
        with open(file_path, "w") as f:
            pass
        logging.info(f"Created empty file: {file_path}")
    else:
        logging.info(f"File already exists: {file_path}")