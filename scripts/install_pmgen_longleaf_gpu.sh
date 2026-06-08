#!/bin/bash
# Longleaf-compatible PMGen install (run on a GPU compute node).
# Uses CPU conda spec + pip JAX CUDA to avoid strict PMGen.yml solve failures on HPC.

set -euo pipefail

PMGEN_ROOT="${PMGEN_ROOT:-$(pwd)}"
cd "$PMGEN_ROOT"

module load cuda 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"

CONDA_CMD="mamba"
command -v mamba &>/dev/null || CONDA_CMD="conda"

echo "=== Longleaf PMGen install ==="
echo "Host: $(hostname)"
echo "PMGEN_ROOT: $PMGEN_ROOT"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"

if conda env list | grep -q "^PMGen "; then
    echo "Removing existing PMGen env..."
    conda env remove -n PMGen -y
fi

echo "Creating PMGen env from PMGen-cpuonly.yml (flexible channels)..."
$CONDA_CMD env create -f PMGen-cpuonly.yml --channel-priority flexible

conda activate PMGen

echo "Installing pip requirements (JAX CUDA + AFfine deps)..."
pip install --upgrade pip
pip install -r pip_requirements.txt

echo "Installing PANDORA..."
cd PANDORA
pip install -e .
cd ..

echo "Fetching PANDORA data..."
pandora-fetch

echo "Downloading AFfine parameters..."
AFFINE_ZIP_URL="https://owncloud.gwdg.de/index.php/s/M1YQOgKxLbVjO0G/download"
AFFINE_ZIP_NAME="AFfine/AFfine.zip"
mkdir -p AFfine
if [[ ! -f AFfine/af_params/params_original/.done 2>/dev/null ]]; then
    wget -O "$AFFINE_ZIP_NAME" "$AFFINE_ZIP_URL"
    unzip -o "$AFFINE_ZIP_NAME" -d AFfine
fi

if [[ ! -d ProteinMPNN ]]; then
    echo "Cloning ProteinMPNN..."
    git clone https://github.com/dauparas/ProteinMPNN.git
else
    echo "ProteinMPNN already present — skipping clone."
fi

python -c "import jax; import torch; print('jax devices:', jax.devices()); print('torch cuda:', torch.cuda.is_available())"

echo "=== Longleaf PMGen install complete ==="
