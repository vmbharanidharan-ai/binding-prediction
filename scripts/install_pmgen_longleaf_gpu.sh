#!/bin/bash
# Longleaf-compatible PMGen install (run on a GPU compute node).

set -euo pipefail

PMGEN_ROOT="${PMGEN_ROOT:-$(pwd)}"
cd "$PMGEN_ROOT"

module load cuda 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"

echo "=== Longleaf PMGen install ==="
echo "Host: $(hostname)"
echo "PMGEN_ROOT: $PMGEN_ROOT"
echo "Conda: $(conda info --base)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"

if conda env list | awk '{print $1}' | grep -qx "PMGen"; then
    echo "Removing existing PMGen env..."
    conda env remove -n PMGen -y || true
fi

echo "Step 1/7: Creating PMGen env (python 3.9)..."
conda create -n PMGen python=3.9 pip -y --override-channels -c conda-forge

echo "Step 2/7: Activating PMGen..."
conda activate PMGen

echo "Step 3/7: Installing core conda packages (conda-forge only)..."
conda install -y --override-channels -c conda-forge \
    biopython pandas numpy scipy pyyaml requests tqdm matplotlib seaborn wget unzip git

echo "Step 4/7: Installing pip requirements (JAX CUDA + AFfine)..."
pip install --upgrade pip "typing-extensions>=4.9"
pip install -r pip_requirements.txt || {
    echo "WARN: pip_requirements had conflicts — retrying with upgraded typing-extensions..."
    pip install --upgrade "typing-extensions>=4.9"
    pip install -r pip_requirements.txt --no-deps
    pip install absl-py chex dm-haiku dm-tree flatbuffers jmp optax tabulate toolz protobuf urllib3 python-Levenshtein
}
pip install --upgrade "typing-extensions>=4.9"

echo "Step 5/7: Installing PANDORA..."
cd PANDORA
pip install -e .
cd ..

echo "Step 6/7: Fetching PANDORA data + AFfine weights..."
pandora-fetch

AFFINE_ZIP_URL="https://owncloud.gwdg.de/index.php/s/M1YQOgKxLbVjO0G/download"
AFFINE_ZIP_NAME="AFfine/AFfine.zip"
mkdir -p AFfine
if [[ ! -d AFfine/af_params ]]; then
    wget -O "$AFFINE_ZIP_NAME" "$AFFINE_ZIP_URL"
    unzip -o "$AFFINE_ZIP_NAME" -d AFfine
fi

if [[ ! -d ProteinMPNN ]]; then
    git clone https://github.com/dauparas/ProteinMPNN.git
fi

echo "Step 7/7: Verifying GPU + JAX..."
python -c "import jax; import torch; print('jax devices:', jax.devices()); print('torch cuda:', torch.cuda.is_available())"

echo "=== Longleaf PMGen install complete ==="
