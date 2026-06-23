# neo-binder: Minibinder Design for Splicing-Factor Neoepitopes

**Peptide–HLA structure prediction → RFdiffusion minibinder design → AlphaFold validation → XGBoost ranking**

A modular computational immunoinformatics pipeline for prioritizing neoantigen peptides and designing protein binders against peptide–MHC complexes. Built for UNC Longleaf HPC with restart-safe, manifest-driven execution.

---

## Overview

Immunotherapy against tumor-specific neoantigens requires not only identifying presented peptides, but also developing molecular tools to engage them. This project targets **novel neoepitopes arising from aberrant splicing in cancer** — particularly cryptic exon inclusion and mis-splicing events in **splicing-factor pathways** — as candidates for targeted immunotherapy. Most of these peptides lack structural characterization, known binders, or experimental validation.

The pipeline closes that gap: given a cohort of neoantigen candidates (peptide sequence, HLA allele, presentation metrics, source gene), it produces a **ranked table of peptide–allele–binder triplets** ready for experimental follow-up.

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

## Scientific motivation

Splicing-factor mutations and mis-splicing events can produce **tumor-specific peptide sequences** presented on MHC class I. These neoepitopes are attractive immunotherapy targets, but most lack structural characterization or existing binders.

This pipeline addresses that gap computationally:

| Question | Approach |
|----------|----------|
| What does the peptide look like on HLA? | AlphaFold-Multimer (ColabFold) structural ensemble |
| Which structural models are reliable? | Interface metrics, RMSD clustering, confidence scoring |
| Can we design a binder to the exposed peptide? | RFdiffusion backbone generation + ProteinMPNN sequence design |
| Does the binder engage the target? | AlphaFold-Multimer re-folding and structural scoring |
| Which designs to test first? | XGBoost learning-to-rank over biological, structural, and embedding features |

See [Biological interpretation](#biological-interpretation) for how to read scores and feature modalities.

---

## The challenge: RFdiffusion on heterogeneous GPU clusters

RFdiffusion is a state-of-the-art SE(3)-equivariant diffusion model for protein binder design. It has a **fixed, fragile dependency stack**:

- PyTorch 1.9.1 + CUDA 11.1
- DGL 1.0.0 with GPU graph operations
- SE3Transformer + Hydra configuration

On Longleaf, a conda-based `SE3nv` environment produced persistent runtime failures:

```
DGLError: Operator Range does not support cuda device
```

The root cause was DGL's CUDA libraries not resolving consistently across GPU node types — `LD_LIBRARY_PATH` workarounds were node-dependent and brittle. RFdiffusion inference never completed successfully under conda.

---

## The solution: containerized RFdiffusion

RFdiffusion now runs inside an **Apptainer container** with a pinned, immutable stack:

| Component | Version |
|-----------|---------|
| PyTorch | 1.9.1+cu111 |
| DGL | 1.0.0 (CUDA 11.1) |
| NumPy / SciPy | 1.23.5 / 1.10.1 |
| RFdiffusion + SE3Transformer | upstream, editable install |

The container:

- **Isolates** the full RFdiffusion runtime from host conda environments
- **Bind-mounts** model weights from `$PROJECT_ROOT/RFdiffusion/models` at runtime
- **Pre-validates** DGL CUDA graph operations before each inference job
- **Runs identically** on A100, L40, and Volta GPU nodes on Longleaf

Orchestration stays in a lightweight `neo_binder` conda env; only Step 3 inference enters the container.

### Performance and reproducibility

| Aspect | Conda SE3nv (before) | Apptainer (after) |
|--------|----------------------|-------------------|
| One-time setup | ~1 h + ongoing debugging | ~40 min build, then stable |
| Runtime GPU errors | DGL CUDA failures on inference | Zero (pre-flight validated) |
| Cross-node behavior | Node-dependent | Identical |
| Maintenance | High (conda drift, numpy conflicts) | None (immutable image) |
| Disk per dataset | ~500 MB env | ~2–3 GB `.sif` (shareable across projects) |

See [`containers/APPTAINER_BUILD_INSTRUCTIONS.md`](containers/APPTAINER_BUILD_INSTRUCTIONS.md) for build details.

---

## Architecture

This is not a single end-to-end model. Four ML systems are orchestrated with explicit separation of concerns:

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

| Stage | Tool | Role |
|-------|------|------|
| 0 | ESM-2 | Peptide/allele embeddings for downstream ranking |
| 1 | ColabFold / AlphaFold-Multimer v3 | Peptide–HLA complex modeling |
| 1.5 | BioPython | HLA groove truncation (reduces diffusion compute) |
| 2 | Custom metrics | Structure confidence, clustering, top-N selection |
| 3 | RFdiffusion (Apptainer) | De novo binder backbone generation |
| 3.5 | ProteinMPNN | Sequence assignment to binder backbone |
| 4 | ColabFold multimer | Binder + target re-folding validation |
| 5 | XGBoost (`rank:pairwise`) | Multi-modal candidate prioritization |

---

## Key contributions

- **End-to-end immunoinformatics workflow** from RNA-seq–derived neoepitope lists to ranked minibinder candidates, with restart-safe HPC execution
- **Solved RFdiffusion GPU integration** on Longleaf by replacing conda with a reproducible Apptainer image and bind-mounted weights
- **Environment isolation architecture** — orchestration (`neo_binder`), structure prediction (`alphafoldenv`), sequence design (`proteinmpnn`), and diffusion (container) each run in isolated runtimes without PATH conflicts
- **Target-aware binder design** — automatic peptide hotspot selection, HLA groove truncation, and PPI-conditioned RFdiffusion contigs
- **Leakage-aware ML ranking** — gene-level train/test splits, biological priors separated from structural confidence scores

---

## Technical stack

| Layer | Components |
|-------|------------|
| Structure prediction | ColabFold, AlphaFold2-Multimer v3, JAX/CUDA |
| Generative design | RFdiffusion in Apptainer (SE(3)-equivariant diffusion) |
| Sequence design | ProteinMPNN |
| Protein language models | ESM-2 (650M), optional ProtT5 |
| ML ranking | XGBoost learning-to-rank, PCA feature compression |
| Orchestration | Python, YAML config, SLURM, manifest-driven restart |
| Container runtime | Apptainer/Singularity with `--nv` GPU passthrough |

---

## Quick start (Longleaf)

### 1. Environment

```bash
conda env create -f environment.yml
conda activate neo_binder
```

### 2. Set paths

Large outputs belong on `/work`, not `$HOME`:

```bash
export PROJECT_ROOT=/work/users/$USER/minibinder_prediction
export NEO_BINDER_WORK_ROOT=$PROJECT_ROOT/work
cd $PROJECT_ROOT/binding-prediction
```

### 3. One-time external tool setup

```bash
bash scripts/prefetch_colabfold_weights.sh          # ColabFold (Steps 1, 4)
bash scripts/setup_colabfold_longleaf.sh            # if not already installed
bash scripts/setup_rfdiffusion_longleaf.sh          # clone RFdiffusion + weights
sbatch slurm/build_rfdiffusion_container.sbatch     # build rfdiffusion.sif (~40 min)
bash scripts/setup_proteinmpnn_longleaf.sh          # ProteinMPNN (Step 3.5)
```

### 4. Prepare input and run

Tab-separated cohort file at `data/step5_input.tsv` (see [Input format](#input-format)):

```bash
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

Monitor and inspect:

```bash
squeue -u $USER
python utils/step_summary.py --step step1
tail -f logs/step3_*.out
```

---

## RFdiffusion container (Step 3)

Step 3 requires a built Apptainer image and downloaded model weights:

```bash
# On a GPU node
module load apptainer
bash scripts/build_rfdiffusion_container.sh
bash scripts/verify_rfdiffusion_container.sh
```

Artifacts:

| Path | Description |
|------|-------------|
| `$PROJECT_ROOT/rfdiffusion.sif` | Container image (~2–3 GB); shareable across datasets |
| `$PROJECT_ROOT/RFdiffusion/models/*.pt` | Checkpoint weights (bind-mounted read-only) |
| `scripts/run_rfdiffusion_container.sh` | Wrapper invoked by the pipeline |

The same `.sif` can be copied or symlinked into other project roots; only weights and input PDBs are dataset-specific.

---

## Pipeline details

### Step 1: Peptide–HLA multimer

ColabFold folds peptide and HLA as a **single multimer** (`PEPTIDE:HLA` FASTA). Outputs land in `work/step1_structures/`; `parsed_structures.tsv` is written for downstream steps.

### Step 1.5: HLA groove truncation (optional)

RFdiffusion does not need the full HLA heavy chain. Step 1.5 keeps peptide (~9 aa) + HLA residues 25–180 (α1/α2 groove) in separate truncated PDBs. Step 3 prefers these when `step1_5.enabled: true`.

### Hotspot selection (Step 3)

`step3_rfdesign/select_peptide_hotspots.py` selects 5–6 exposed peptide residues for `ppi.hotspot_res` — skipping MHC anchor positions and preferring charged/bulky residues in the P4–P8 stretch.

### Step 3.5: ProteinMPNN

RFdiffusion produces backbone only. ProteinMPNN assigns binder sequences while peptide + HLA remain fixed. Output: `work/step3_5_sequences/designed_binders.tsv`.

### Step 5: ML ranking

XGBoost `rank:pairwise` over biological, structural, and embedding features. Gene-level split reduces leakage across junctions from the same source gene. See [ML features](#ml-features-step-5) under Biological interpretation.

---

## Input format

`data/step5_input.tsv` — one row per peptide–allele pair:

| Column | Description |
|--------|-------------|
| `peptide` | Neoantigen peptide sequence |
| `allele` | HLA allele ID (e.g. `HLA_A0201`) |
| `gene` | Source gene (used for leakage-safe ML split) |
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
| `plddt_mean`, `pae_mean` | 1 | ColabFold per-residue confidence in peptide–HLA model |
| `structure_confidence_score` | 2 | Reliability of peptide–HLA structural hypothesis |
| `binder_structural_score` | 4 | Predicted structural engagement of designed binder |
| `binder_score` | 5 | ML-integrated rank across all feature modalities |

These scores inform prioritization under uncertainty — they are **not** direct measures of binding affinity or immunogenicity.

---

## Local execution

```bash
python run_pipeline.py --input data/step5_input.tsv     # full pipeline
python run_pipeline.py --step step3                       # single step
python run_pipeline.py --from-step step3                  # resume
python run_pipeline.py --dry-run
```

Each step writes status TSVs and `done.flag` files. Re-submitting a SLURM job skips completed work.

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

## Work directory layout

```
$NEO_BINDER_WORK_ROOT/
├── step1_structures/          ColabFold multimer outputs
├── step1_5_truncated/         Groove-truncated PDBs (optional)
├── step2_scored/              Metrics, clusters, rankings
├── step3_binders/             RFdiffusion backbones
├── step3_5_sequences/         ProteinMPNN sequences
├── step4_validated/           Validation scores
├── step5_ranked/              Final rankings
└── embeddings/                ESM-2 vectors
```

---

## External dependencies

| Tool | Stages | Setup |
|------|--------|-------|
| **ColabFold** | 1, 4 | `scripts/setup_colabfold_longleaf.sh` |
| **RFdiffusion** | 3 | `scripts/setup_rfdiffusion_longleaf.sh` + `slurm/build_rfdiffusion_container.sbatch` |
| **ProteinMPNN** | 3.5 | `scripts/setup_proteinmpnn_longleaf.sh` |
| **Apptainer** | 3 | `module load apptainer` on Longleaf GPU nodes |

Longleaf notes: GPU jobs use `--qos=gpu_access`; ColabFold cache is redirected to `/work` via `scripts/colabfold_work_paths.sh`.

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
└── scripts/                     HPC setup, container wrappers
```

---

## Storage estimates (~100 peptides)

| Data | Size |
|------|------|
| AlphaFold structures | 50–200 GB |
| RFdiffusion designs | 10–50 GB |
| Embeddings | 1–5 GB |
| Container image | 2–3 GB (one-time, shareable) |

Use `/work/users/<onyen>/`, not `$HOME`.

---

## License

Academic / research use. See individual tool licenses for ColabFold, RFdiffusion, ProteinMPNN, ESM-2, and XGBoost.
