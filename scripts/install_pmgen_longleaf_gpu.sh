#!/bin/bash
# Longleaf HPC port of official PMGen install.sh when `bash install.sh` fails.
#
# Replicates the upstream install.sh outcome on GPU nodes:
#   - conda env: PMGen
#   - pip install -e PANDORA  (import: PANDORA, pip name: CSB-PANDORA)
#   - pandora-fetch
#   - AFfine weights under AFfine/af_params/
#   - ProteinMPNN clone
#   - pip install -r pip_requirements.txt (JAX/CUDA for AlphaFold)
#
# Skips PMGen.yml / mamba env update (strict channel priority on Longleaf).
# Run on a GPU compute node, not the login node.

set -euo pipefail

PMGEN_ROOT="${PMGEN_ROOT:-$(pwd)}"
cd "$PMGEN_ROOT"

module load cuda 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"

echo "=== Longleaf PMGen install (official-equivalent) ==="
echo "Host: $(hostname)"
echo "PMGEN_ROOT: $PMGEN_ROOT"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"

RESUME=0
if conda env list | awk '{print $1}' | grep -qx "PMGen" \
    && [[ -d AFfine/af_params ]] \
    && python -c "import PANDORA" 2>/dev/null; then
    echo "PMGen env + PANDORA + AFfine weights found — resume/verify mode."
    RESUME=1
    conda activate PMGen
elif conda env list | awk '{print $1}' | grep -qx "PMGen"; then
    echo "Removing partial PMGen env..."
    conda env remove -n PMGen -y || true
fi

if [[ $RESUME -eq 0 ]]; then
    echo "[official step] Create conda env PMGen..."
    conda create -n PMGen python=3.9 pip -y --override-channels -c conda-forge
    conda activate PMGen

    echo "[official step] Base dependencies..."
    conda install -y --override-channels -c conda-forge \
        biopython pandas numpy scipy pyyaml requests tqdm matplotlib seaborn wget unzip git

    echo "[official step] pip install -r pip_requirements.txt (JAX/CUDA)..."
    pip install --upgrade pip "typing-extensions>=4.9"
    pip install -r pip_requirements.txt || {
        echo "WARN: pip_requirements conflicts — retrying with typing-extensions fix..."
        pip install --upgrade "typing-extensions>=4.9"
        pip install -r pip_requirements.txt --no-deps
        pip install absl-py chex dm-haiku dm-tree flatbuffers jmp optax tabulate toolz protobuf urllib3 python-Levenshtein
    }
    pip install --upgrade "typing-extensions>=4.9"

    echo "[official step] pip install -e PANDORA..."
    cd PANDORA
    pip install -e .
    cd ..

    echo "[official step] pandora-fetch..."
    pandora-fetch

    echo "[official step] Download AFfine weights..."
    AFFINE_ZIP_URL="https://owncloud.gwdg.de/index.php/s/M1YQOgKxLbVjO0G/download"
    AFFINE_ZIP_NAME="AFfine/AFfine.zip"
    mkdir -p AFfine
    if [[ ! -d AFfine/af_params ]]; then
        wget -O "$AFFINE_ZIP_NAME" "$AFFINE_ZIP_URL"
        unzip -o "$AFFINE_ZIP_NAME" -d AFfine
    fi

    echo "[official step] Clone ProteinMPNN..."
    if [[ ! -d ProteinMPNN ]]; then
        git clone https://github.com/dauparas/ProteinMPNN.git
    fi
else
    echo "Install steps skipped (already present)."
fi

echo "[verify] PANDORA + JAX + GPU..."
_SAVED_LD="${LD_LIBRARY_PATH:-}"
unset LD_LIBRARY_PATH
set +e
python -c "import PANDORA, jax, torch; print('PANDORA OK'); print('jax', jax.__version__); print('jax devices:', jax.devices()); print('torch cuda:', torch.cuda.is_available())"
VERIFY_RC=$?
set -e
export LD_LIBRARY_PATH="$_SAVED_LD"

if [[ $VERIFY_RC -ne 0 ]]; then
    echo "WARN: GPU verify exited $VERIFY_RC — use source scripts/pmgen_env.sh before running PMGen."
else
    echo "GPU verify OK"
fi

echo "=== Longleaf PMGen install complete ==="
echo "Run structure prediction (official CLI):"
echo "  conda activate PMGen"
echo "  cd $PMGEN_ROOT"
echo "  python run_PMGen.py --mode wrapper --run single --df input.tsv --output_dir output/ --initial_guess"
