#!/bin/bash
#SBATCH --job-name=be-lora
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --partition=AMD9554-H200NVL-common

set -euo pipefail

PROJECT_DIR="/slowhome/common/barisenok.av/mini_gpt"
VENV_DIR="$PROJECT_DIR/.venv"

mkdir -p "$PROJECT_DIR/logs"
cd "$PROJECT_DIR"
source "$VENV_DIR/bin/activate"

python -m pip install -r requirements.txt
export PYTHONUNBUFFERED=1

python -u llm_adaptation/train_lora.py --config llm_adaptation/config.yaml
python -u llm_adaptation/eval.py --config llm_adaptation/config.yaml
