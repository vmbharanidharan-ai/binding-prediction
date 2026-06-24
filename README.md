# neo-binder: Minibinder Design for Splicing-Factor Neoepitopes

**Peptide–HLA structure prediction → RFdiffusion minibinder design → AlphaFold validation → XGBoost ranking**

A modular computational immunoinformatics pipeline for prioritizing neoantigen peptides and designing protein binders against peptide–MHC complexes. Built for UNC Longleaf HPC with restart-safe, manifest-driven execution.

**In one sentence:** Given RNA-seq–derived neoantigen candidates, this pipeline models each peptide on its presenting HLA allele, designs de novo protein minibinders against the complex, validates them structurally, and ranks designs for experimental follow-up.

**Who might use this:** Computational biologists and immunology researchers working on tumor neoantigens — especially splice-derived epitopes — who need a reproducible, HPC-ready workflow from cohort TSV to ranked binder candidates without hand-wiring multiple ML tools.

---

## Table of contents

1. [Overview](#overview)
2. [What this does](#what-this-does)
3. [Biological interpretation](#biological-interpretation)
4. [Architecture](#architecture)
5. [Technical stack](#technical-stack)
6. [Key contributions](#key-contributions)
7. [RFdiffusion on HPC (Apptainer)](#rfdiffusion-on-hpc-apptainer)
8. [Quick start (Longleaf)](#quick-start-longleaf)
9. [Reproducing in a new directory](#reproducing-in-a-new-directory)
10. [Usage](#usage)
11. [Pipeline details](#pipeline-details)
12. [Input, outputs, and scoring](#input-format)
13. [Configuration](#configuration)
14. [Repository and work directories](#repository-layout)
15. [Future work](#future-work)
16. [Development and troubleshooting](#development-and-troubleshooting)

---

## Overview

Immunotherapy against tumor-specific neoantigens requires not only identifying presented peptides, but also developing molecular tools to engage them. This project targets **novel neoepitopes arising from aberrant splicing in cancer** — particularly cryptic exon inclusion and mis-splicing events in **splicing-factor pathways** — as candidates for targeted immunotherapy.

Splicing-factor mutations and mis-splicing events can produce **tumor-specific peptide sequences** presented on MHC class I. These neoepitopes are attractive immunotherapy targets, but most lack structural characterization, known binders, or experimental validation.

The pipeline closes that gap: given a cohort of neoantigen candidates (peptide sequence, HLA allele, presentation metrics, source gene), it produces a **ranked table of peptide–allele–binder triplets** ready for experimental follow-up.

| Question | Approach |
|----------|----------|
| What does the peptide look like on HLA? | AlphaFold-Multimer (ColabFold) structural ensemble |
| Which structural models are reliable? | Interface metrics, RMSD clustering, confidence scoring |
| Can we design a binder to the exposed peptide? | RFdiffusion backbone generation + ProteinMPNN sequence design |
| Does the binder engage the target? | AlphaFold-Multimer re-folding and structural scoring |
| Which designs to test first? | XGBoost learning-to-rank over biological, structural, and embedding features |

---

## What this does

Given input from RNA-seq–derived neoantigen discovery (e.g., neoJunction calls, MHCflurry/NetMHCpan presentation scores), the pipeline:

1. Predicts **peptide–HLA complex structures** (AlphaFold-Multimer via ColabFold; ensemble of conformations)
2. Scores and **clusters structural hypotheses** to identify reliable models
3. Optionally **truncates HLA to the binding groove** to reduce generative-design compute while preserving the presented peptide
4. Generates **minibinder backbones** (~50–80 aa) conditioned on the exposed peptide surface (RFdiffusion)
5. Assigns **binder sequences** to those backbones while keeping peptide + HLA fixed (ProteinMPNN)
6. **Validates** designs by re-folding binder + target with AlphaFold-Multimer
7. **Ranks** candidates with an XGBoost learning-to-rank model over biological, structural, and embedding features

The final output (`step5_ranked/final_rankings.tsv`) integrates all evidence modalities into a single prioritization for wet-lab screening.

---

## Biological interpretation

### Design principle

This is not a single end-to-end predictor of immunogenicity or clinical response. Each stage measures something different:

| Evidence type | Source | Role in the pipeline |
|---------------|--------|----------------------|
| **Biological priors** | MHCflurry, NetMHCpan, cohort prevalence, tumor expression (PSR) | How likely the peptide is presented and shared across patients |
| **Structural confidence** | pLDDT, PAE, interface metrics, cluster agreement | How reliable the *computational models* are — not binding affinity |
| **Generative design** | RFdiffusion + ProteinMPNN | Whether a plausible binder scaffold and sequence can be built against the target |
| **Structural validation** | AlphaFold re-folding scores | Whether the designed binder appears to engage the peptide–HLA complex in silico |
| **Integrated ranking** | XGBoost (`rank:pairwise`) | Combines all modalities; no hand-tuned weighted score before Step 5 |

**Biological presentation scores are treated as priors.** Structural scores measure **model reliability and engagement under uncertainty.** The ML ranker integrates everything at the end.

### How to read the scores

| Score | Stage | Biological meaning |
|-------|-------|-------------------|
| `plddt_mean`, `pae_mean` | 1 | Per-residue confidence in the peptide–HLA structural model |
| `structure_confidence_score` | 2 | Overall reliability of the peptide–HLA hypothesis (pLDDT, PAE, cluster agreement) |
| `binder_structural_score` | 4 | Whether the designed binder appears to structurally engage the target in the validation fold |
| `binder_score` | 5 | ML-integrated rank across biological, structural, and embedding features |

**None of these scores are binding affinity, T-cell recognition, or clinical immunogenicity directly.** They inform prioritization when experimental data are scarce — which splice-derived neoepitopes and binder designs to test first.

### ML features (Step 5)

The ranker combines three feature groups (see `config/config.yaml` → `features`):

- **Biological:** MHCflurry/NetMHCpan presentation, cohort carrier count, tumor PSR, frameshift flag — captures *presentation likelihood and tumor relevance*
- **Structural:** confidence scores from Steps 2 and 4, interface metrics, optional Rosetta energies — captures *model quality and predicted engagement*
- **Embedding:** ESM-2 1280D → PCA 50D, optional ProtT5, cosine similarity to IEDB known binders — captures *sequence-level similarity to characterized antigens*

Training uses `rank:pairwise` with **gene-level train/test split** to reduce leakage across multiple junctions from the same splicing-factor gene.

---

## Architecture

This pipeline **orchestrates multiple interdependent ML systems** with different CUDA/Python dependencies. It processes **RNA-seq–derived neoantigen cohorts through multiple inference stages** — structure prediction, generative design, sequence assignment, validation, and ranking — without collapsing them into a single fragile environment.

```
Cohort TSV (peptide, HLA, presentation metrics, gene)
        │
        ▼
┌──────────────────────────────────────────────────────────────┐
│  0   Embeddings          ESM-2 (+ optional ProtT5)           │
├──────────────────────────────────────────────────────────────┤
│  1   Structure           ColabFold multimer → peptide:HLA    │
│                          → parsed_structures.tsv             │
├──────────────────────────────────────────────────────────────┤
│  1.5 Truncation (opt.)   HLA α1/α2 groove (res 25–180)     │
│                          → step1_5_truncated/                │
├──────────────────────────────────────────────────────────────┤
│  2   Scoring             Interface metrics, RMSD clusters    │
│                          → ranked_structures.tsv             │
├──────────────────────────────────────────────────────────────┤
│  3   RFdiffusion         Minibinder backbones (~50–80 aa)    │
│         (Apptainer)      conditioned on peptide hotspots     │
├──────────────────────────────────────────────────────────────┤
│  3.5 ProteinMPNN         Binder sequences on backbones       │
│                          → designed_binders.tsv              │
├──────────────────────────────────────────────────────────────┤
│  4   Validation          AlphaFold-Multimer re-folding       │
├──────────────────────────────────────────────────────────────┤
│  5   ML ranking          XGBoost over multi-modal features   │
│                          → final_rankings.tsv                │
└──────────────────────────────────────────────────────────────┘
```

| Stage | Tool | Runtime | Role |
|-------|------|---------|------|
| 0 | ESM-2 | `neo_binder` | Peptide/allele embeddings for downstream ranking |
| 1 | ColabFold / AlphaFold-Multimer v3 | `alphafoldenv` | Peptide–HLA complex modeling |
| 1.5 | BioPython | `neo_binder` | HLA groove truncation (reduces diffusion compute) |
| 2 | Custom metrics | `neo_binder` | Structure confidence, clustering, top-N selection |
| 3 | RFdiffusion | **Apptainer** (`rfdiffusion.sif`) | De novo binder backbone generation |
| 3.5 | ProteinMPNN | `proteinmpnn` | Sequence assignment to binder backbone |
| 4 | ColabFold multimer | `alphafoldenv` | Binder + target re-folding validation |
| 5 | XGBoost (`rank:pairwise`) | `neo_binder` | Multi-modal candidate prioritization |

**Apptainer at Step 3:** RFdiffusion is the only stage that requires the legacy PyTorch 1.9 / CUDA 11.1 / DGL stack. Containerizing it keeps that stack isolated while ColabFold (JAX) and ProteinMPNN (modern PyTorch) run in their own environments. See [RFdiffusion on HPC](#rfdiffusion-on-hpc-apptainer).

---

## Technical stack

| Layer | Tool | Notes |
|-------|------|-------|
| **Step 0** — Embeddings | ESM-2 (650M), optional ProtT5 | Peptide/allele sequence embeddings |
| **Step 1** — Structure | ColabFold (AlphaFold2-Multimer v3) | Peptide–HLA complex modeling; JAX/CUDA |
| **Step 1.5** — Target prep | BioPython | HLA groove truncation for diffusion |
| **Step 2** — Scoring | Custom metrics + clustering | Structure confidence, top-N selection |
| **Step 3** — Backbone design | RFdiffusion in **Apptainer** | PyTorch 1.9.1 + CUDA 11.1 + DGL 1.0.0 |
| **Step 3.5** — Sequence design | ProteinMPNN | Binder sequence on RFdiffusion backbone |
| **Step 4** — Validation | ColabFold multimer | Re-fold binder + target; engagement scoring |
| **Step 5** — Ranking | XGBoost (`rank:pairwise`) | Multi-modal learning-to-rank |
| **Orchestration** | Python, YAML, SLURM | Manifest-driven, restart-safe |
| **Container runtime** | Apptainer/Singularity `--nv` | GPU passthrough for Step 3 only |

---

## Key contributions

This is **research/production computational work** — not a tutorial or class exercise. Specific technical contributions:

- **End-to-end immunoinformatics workflow** — processes RNA-seq–derived neoantigen cohorts through structure prediction, generative binder design, validation, and ML ranking with restart-safe SLURM execution on Longleaf
- **Solved DGL CUDA runtime compatibility** — diagnosed `DGLError: Operator Range does not support cuda device` across heterogeneous GPU nodes; replaced fragile conda `SE3nv` with a reproducible Apptainer image
- **Multi-environment orchestration** — manages ColabFold (JAX), RFdiffusion (legacy PyTorch 1.9), ProteinMPNN, and XGBoost in isolated runtimes without PATH or library conflicts
- **Apptainer container ensures identical execution across GPU nodes** — pinned torch/DGL/SE3Transformer stack, pre-flight CUDA validation, bind-mounted weights
- **Target-aware binder design** — automatic peptide hotspot selection, HLA groove truncation, PPI-conditioned RFdiffusion contigs for splice-derived neoepitopes
- **Leakage-aware ML ranking** — gene-level train/test splits; biological priors separated from structural confidence scores

---

## RFdiffusion on HPC (Apptainer)

### The problem

RFdiffusion depends on a **fixed, fragile stack** — PyTorch 1.9.1 + CUDA 11.1, DGL 1.0.0, SE3Transformer, Hydra — that failed under conda on Longleaf:

```
DGLError: Operator Range does not support cuda device
```

DGL's CUDA libraries failed to resolve consistently across GPU node types. Beyond RFdiffusion, the pipeline orchestrates **multiple interdependent ML systems** with incompatible Python/CUDA dependencies; a single conda environment caused PATH and library conflicts.

### The solution

RFdiffusion runs inside an **Apptainer container** with a pinned, immutable stack:

| Component | Version |
|-----------|---------|
| PyTorch | 1.9.1+cu111 |
| DGL | 1.0.0 (CUDA 11.1) |
| NumPy / SciPy | 1.23.5 / 1.10.1 |
| RFdiffusion + SE3Transformer | upstream, editable install |

The container **isolates** the RFdiffusion stack, **eliminates** conda drift, **pre-validates** DGL CUDA before inference, and **bind-mounts** weights from `$PROJECT_ROOT/RFdiffusion/models`. Orchestration uses `neo_binder`; ColabFold uses `alphafoldenv`; ProteinMPNN uses `proteinmpnn`. **Only Step 3 enters the container.**

### Performance and reproducibility

| Aspect | Before (conda) | After (Apptainer) |
|--------|----------------|-------------------|
| Setup time | ~1 hour (with debugging) | ~40 minutes (one-time build) |
| Runtime issues | DGL CUDA failures | Zero (pre-flight validated) |
| Reproducibility | Node-dependent | Guaranteed across GPU nodes |
| Maintenance | High (conda drift) | None (immutable image) |
| Disk | ~500 MB per env | ~2–3 GB `.sif` (shareable across datasets) |

### Build and verify

```bash
module load apptainer
sbatch slurm/build_rfdiffusion_container.sbatch   # ~40 min on GPU node
bash scripts/verify_rfdiffusion_container.sh
```

| Artifact | Path |
|----------|------|
| Container image | `$PROJECT_ROOT/rfdiffusion.sif` |
| Model weights | `$PROJECT_ROOT/RFdiffusion/models/*.pt` |
| Inference wrapper | `scripts/run_rfdiffusion_container.sh` |
| Definition file | `containers/rfdiffusion.def` |

Full build instructions: [`containers/APPTAINER_BUILD_INSTRUCTIONS.md`](containers/APPTAINER_BUILD_INSTRUCTIONS.md)

---

## Quick start (Longleaf)

### 1. One-time setup (per user)

```bash
conda env create -f environment.yml -n neo_binder
conda activate neo_binder
```

### 2. First project — full tool install

Pick a project root on `/work` (not `$HOME`). All large outputs and external tools live here:

```bash
export PROJECT_ROOT=/work/users/$USER/minibinder_prediction
export NEO_BINDER_WORK_ROOT=$PROJECT_ROOT/work
mkdir -p "$PROJECT_ROOT"
cd "$PROJECT_ROOT"

git clone https://github.com/vmbharanidharan-ai/binding-prediction.git
cd binding-prediction

bash scripts/prefetch_colabfold_weights.sh          # ColabFold (Steps 1, 4)
bash scripts/setup_colabfold_longleaf.sh
bash scripts/setup_rfdiffusion_longleaf.sh          # clone RFdiffusion + weights
sbatch slurm/build_rfdiffusion_container.sbatch     # build rfdiffusion.sif (~40 min)
bash scripts/setup_proteinmpnn_longleaf.sh          # ProteinMPNN (Step 3.5)
```

After this, `$PROJECT_ROOT` contains `rfdiffusion.sif`, `RFdiffusion/`, `ProteinMPNN/`, `alphafoldenv/`, and `colabfold_params/`. You can **reuse these across cohorts** — see [Reproducing in a new directory](#reproducing-in-a-new-directory).

### 3. Set paths and run

```bash
export PROJECT_ROOT=/work/users/$USER/minibinder_prediction
export NEO_BINDER_WORK_ROOT=$PROJECT_ROOT/work
cd $PROJECT_ROOT/binding-prediction

# Single pair (interactive) — saves input for later steps
./slurm/run_pair.sh -i --step 1

# Or point at a cohort TSV
export INPUT_TSV=data/generated/your_cohort.tsv

./slurm/submit_step.sh 0      # embeddings
./slurm/submit_step.sh 1      # peptide–HLA structures
./slurm/submit_step.sh 1.5    # optional: HLA groove truncation
./slurm/submit_step.sh 2      # scoring + clustering
./slurm/submit_step.sh 3      # RFdiffusion backbones
./slurm/submit_step.sh 3.5    # ProteinMPNN sequences
./slurm/submit_step.sh 4      # binder validation
./slurm/submit_step.sh 5      # ML ranking
```

**Important:** `PROJECT_ROOT` or `NEO_BINDER_WORK_ROOT` must be set before every `submit_step.sh` call. If unset, jobs default to `/work/users/$USER/neo_binder` and outputs will appear in the wrong place.

Monitor:

```bash
squeue -u $USER
python utils/step_summary.py --step step1
tail -f logs/step3_*.out
```

---

## Reproducing in a new directory

For each new cohort or peptide set, you do **not** need to re-download weights or rebuild the Apptainer image. Use `scripts/init_project.sh` to bootstrap a fresh project directory.

### What gets created

```
NEW_PROJECT_ROOT/
├── .env                    # source this for path exports
├── work/                   # NEO_BINDER_WORK_ROOT (all pipeline outputs)
├── binding-prediction/     # cloned repo
├── rfdiffusion.sif         # symlinked from old install (if --old-root)
├── RFdiffusion/            # symlinked
├── ProteinMPNN/            # symlinked
├── alphafoldenv/           # symlinked
└── colabfold_params/       # symlinked
```

Pipeline outputs always live under `$NEO_BINDER_WORK_ROOT` (`$PROJECT_ROOT/work`), not inside the git repo.

### Initialize a new project (recommended)

From any checkout of this repo:

```bash
bash scripts/init_project.sh \
  --new-root /work/users/$USER/cohort_2/minibinder_prediction \
  --old-root /work/users/$USER/cohort_1/minibinder_prediction
```

Options:

| Flag | Purpose |
|------|---------|
| `--new-root` | New project directory (required) |
| `--old-root` | Symlink heavy assets from an existing install |
| `--branch` | Git branch to checkout after clone (default: `main`) |

Symlinked assets: `rfdiffusion.sif`, `RFdiffusion`, `ProteinMPNN`, `alphafoldenv`, `colabfold_params`, `.cache`.

Without `--old-root`, run the [first-project tool install](#2-first-project--full-tool-install) under the new root.

### Load environment and run

```bash
source /work/users/$USER/cohort_2/minibinder_prediction/.env
cd $PROJECT_ROOT/binding-prediction

# Create input (one pair)
./slurm/run_pair.sh -p AIMDLVMMV -a HLA_A0201 --gene GENE --step 1 --make-only

# Or reuse saved input from a prior interactive run
source data/generated/current_pair.env

# Submit steps in order
./slurm/submit_step.sh 0
./slurm/submit_step.sh 1
# ... through 5
```

### One-liner: init + submit first step

```bash
bash scripts/run_cohort.sh \
  --root /work/users/$USER/cohort_2/minibinder_prediction \
  --old-root /work/users/$USER/cohort_1/minibinder_prediction \
  --input data/generated/cohort.tsv \
  --start-step 0 \
  --submit
```

### Copy an existing run instead of starting fresh

To resume or duplicate results, copy the `work/` directory and input TSV:

```bash
rsync -a "$OLD_PROJECT_ROOT/work/" "$NEW_PROJECT_ROOT/work/"
rsync -a "$OLD_PROJECT_ROOT/binding-prediction/data/generated/" \
          "$NEW_PROJECT_ROOT/binding-prediction/data/generated/"
source "$NEW_PROJECT_ROOT/.env"
python run_pipeline.py --from-step step4   # example resume point
```

### Key output paths (under `$NEO_BINDER_WORK_ROOT`)

| Step | Main output |
|------|-------------|
| 3 | `step3_binders/binder_designs.tsv` + backbone PDBs |
| 3.5 | `step3_5_sequences/designed_binders.tsv` (sequences; no new PDB) |
| 4 | `step4_validated/binder_scores.tsv` + multimer PDBs in `step4_validated/multimer/` |
| 5 | `step5_ranked/final_rankings.tsv` |

To view the **full binder + peptide + HLA complex**, open the Step 4 multimer PDB (`binder_scores.tsv` → `pdb_path` column).

---

## Usage

**Environment (every session):**

```bash
source $PROJECT_ROOT/.env          # if created by init_project.sh
conda activate neo_binder
cd $PROJECT_ROOT/binding-prediction
```

**Submit individual steps:**

```bash
./slurm/submit_step.sh 3      # RFdiffusion (GPU + Apptainer)
./slurm/submit_step.sh 3.5    # ProteinMPNN
./slurm/submit_step.sh 4      # validation
./slurm/submit_step.sh 5      # ML ranking
```

**Run locally or resume:**

```bash
python run_pipeline.py --input data/step5_input.tsv
python run_pipeline.py --step step3
python run_pipeline.py --from-step step3
python run_pipeline.py --dry-run
```

Each step writes status TSVs and `done.flag` files. Re-submitting skips completed work.

---

## Pipeline details

### Step 1: Peptide–HLA multimer

ColabFold folds peptide and HLA as a **single multimer** (`PEPTIDE:HLA` FASTA). Outputs: `work/step1_structures/`; `parsed_structures.tsv` for downstream steps.

### Step 1.5: HLA groove truncation (optional)

Keeps peptide (~9 aa) + HLA residues 25–180 (α1/α2 groove). Step 3 prefers truncated PDBs when `step1_5.enabled: true`.

### Hotspot selection (Step 3)

`step3_rfdesign/select_peptide_hotspots.py` selects 5–6 exposed peptide residues for `ppi.hotspot_res` — skipping MHC anchors, preferring charged/bulky residues in P4–P8.

### Step 3.5: ProteinMPNN

Assigns binder sequences while peptide + HLA remain fixed. Output: `work/step3_5_sequences/designed_binders.tsv`.

### Step 5: ML ranking

XGBoost `rank:pairwise` with gene-level split. See [ML features](#ml-features-step-5) under Biological interpretation.

---

## Input format

`data/step5_input.tsv` — one row per peptide–allele pair:

| Column | Description |
|--------|-------------|
| `peptide` | Neoantigen peptide sequence |
| `allele` | HLA allele ID (e.g. `HLA_A0201`) |
| `gene` | Source gene (leakage-safe ML split) |
| `junction` | Junction / splice-event identifier |
| `mhcflurry_presentation_percentile` | MHCflurry presentation percentile |
| `netmhcpan_EL_rank` | NetMHCpan EL rank |
| `n_carriers_in_cohort` | Patients carrying this neoantigen |
| `PSR_tumor` | Tumor cell expression proportion |
| `frameshift_flag` | 1 if frameshift-derived |

HLA sequences resolve automatically from IMGT on first run.

---

## Key outputs

| File | Stage | Description |
|------|-------|-------------|
| `step2_scored/parsed_structures.tsv` | 1 | Parsed multimer complexes |
| `step2_scored/ranked_structures.tsv` | 2 | Top structures per peptide |
| `step3_binders/binder_designs.tsv` | 3 | RFdiffusion backbone PDBs |
| `step3_5_sequences/designed_binders.tsv` | 3.5 | ProteinMPNN binder sequences |
| `step4_validated/binder_scores.tsv` | 4 | Structural engagement scores |
| `step5_ranked/final_rankings.tsv` | 5 | **Final ranked candidates** |

---

## Scoring semantics

Quick reference — see [Biological interpretation](#biological-interpretation) for full context.

| Score | Stage | Meaning |
|-------|-------|---------|
| `plddt_mean`, `pae_mean` | 1 | ColabFold per-residue confidence |
| `structure_confidence_score` | 2 | Reliability of peptide–HLA model |
| `binder_structural_score` | 4 | Predicted binder engagement |
| `binder_score` | 5 | ML-integrated rank |

These scores inform prioritization under uncertainty — **not** binding affinity or immunogenicity.

---

## Configuration

Edit `config/config.yaml`:

| Section | Controls |
|---------|----------|
| `runtime.profile` | `optimized` vs `full` accuracy |
| `step1` | ColabFold settings, model count |
| `step1_5` | Truncation range, RFdiffusion target selection |
| `step2` | Clustering threshold, top-N structures |
| `step3` | Binder length, diffusion steps, hotspots, container path |
| `step3_5` | ProteinMPNN model, sampling temperature |
| `step4` | Validation model count |
| `step5` | XGBoost hyperparameters, PCA dimensions |
| `slurm` | Partition, memory, time limits |

---

## Repository layout

```
binding-prediction/
├── run_pipeline.py              Orchestrator
├── config/config.yaml           Pipeline configuration
├── containers/                  Apptainer definition + build docs
├── slurm/                       SLURM batch scripts + submit_step.sh
├── step1_structure_generation/  ColabFold inputs + parsing
├── step1_5_structure_prep/      HLA groove truncation
├── step2_structure_scoring/     Metrics, clustering
├── step3_rfdesign/              RFdiffusion contigs + container inference
├── step3_5_sequence_design/     ProteinMPNN
├── step4_binder_validation/     Multimer validation
├── step5_ml_model/              Feature engineering + XGBoost
├── embeddings/                  ESM-2 / ProtT5
├── hla/                         IMGT allele resolution
├── utils/                       Shared helpers, step summaries
└── scripts/                     HPC setup, init_project.sh, run_cohort.sh
```

### Project root (`$PROJECT_ROOT/`)

```
├── binding-prediction/          Git repo (code, slurm/, config/)
├── work/                        Pipeline outputs (NEO_BINDER_WORK_ROOT)
├── rfdiffusion.sif              Apptainer image (Step 3)
├── RFdiffusion/                 Weights + inference code
├── ProteinMPNN/
├── alphafoldenv/                ColabFold venv
├── colabfold_params/            AlphaFold model weights
├── .env                         Path exports (from init_project.sh)
└── .cache/                      ColabFold cache (optional)
```

### Work directory (`$NEO_BINDER_WORK_ROOT/`)

```
├── step1_structures/          ColabFold multimer outputs
├── step1_5_truncated/         Groove-truncated PDBs (optional)
├── step2_scored/              Metrics, clusters, rankings
├── step3_binders/             RFdiffusion backbones
├── step3_5_sequences/         ProteinMPNN sequences
├── step4_validated/           Validation scores
├── step5_ranked/              Final rankings
└── embeddings/                ESM-2 vectors
```

### External dependencies

| Tool | Stages | Setup |
|------|--------|-------|
| **ColabFold** | 1, 4 | `scripts/setup_colabfold_longleaf.sh` |
| **RFdiffusion** | 3 | `scripts/setup_rfdiffusion_longleaf.sh` + `slurm/build_rfdiffusion_container.sbatch` |
| **ProteinMPNN** | 3.5 | `scripts/setup_proteinmpnn_longleaf.sh` |
| **Apptainer** | 3 | `module load apptainer` |

### Storage estimates (~100 peptides)

| Data | Size |
|------|------|
| AlphaFold structures | 50–200 GB |
| RFdiffusion designs | 10–50 GB |
| Embeddings | 1–5 GB |
| Container image | 2–3 GB (one-time, shareable) |

Use `/work/users/<onyen>/`, not `$HOME`.

---

## Future work

- **New cohorts** — `bash scripts/init_project.sh --new-root ... --old-root ...`; reuse `rfdiffusion.sif` and weights across datasets
- **Alternative structure backends** — PMGen for Step 1 (`step1.backend`)
- **Rosetta interface scoring** — optional in Step 4 (`step4.use_rosetta`)
- **Additional embeddings** — ProtT5 toggle in `config/config.yaml`
- **Accuracy vs. cost** — `runtime.profile: full` for higher diffusion steps and ColabFold recycles

---

## Development and troubleshooting

| Topic | Documentation |
|-------|---------------|
| New project bootstrap | `scripts/init_project.sh`, `scripts/run_cohort.sh` |
| Apptainer build & rebuild | [`containers/APPTAINER_BUILD_INSTRUCTIONS.md`](containers/APPTAINER_BUILD_INSTRUCTIONS.md) |
| Container definition | [`containers/rfdiffusion.def`](containers/rfdiffusion.def) |
| RFdiffusion setup | `scripts/setup_rfdiffusion_longleaf.sh` |
| ColabFold / JAX issues | `scripts/repair_colabfold_jax_haiku.sh` |
| Environment isolation | `scripts/check_env_isolation.sh` |
| Pipeline configuration | `config/config.yaml` |
| Per-step summaries | `python utils/step_summary.py --step <step>` |

HPC notes: GPU jobs use `--qos=gpu_access` on Longleaf. ColabFold cache redirects to `/work` via `scripts/colabfold_work_paths.sh`. If pipeline outputs are missing, verify `echo $NEO_BINDER_WORK_ROOT` matches where you expect before submitting jobs.

---

## License

Academic / research use. See individual tool licenses for ColabFold, RFdiffusion, ProteinMPNN, ESM-2, and XGBoost.
