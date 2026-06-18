# Neo Binder Pipeline

**Peptide–HLA structure prediction → RFdiffusion minibinder design → AlphaFold validation → XGBoost ranking**

A modular computational immunoinformatics pipeline for prioritizing neoantigen peptides and designing protein binders against peptide–MHC complexes. Built for UNC Longleaf HPC with restart-safe, manifest-driven execution.

---

## What this does

Given a cohort of neoantigen candidates (peptide, HLA allele, presentation metrics), the pipeline:

1. Predicts **peptide–HLA complex structures** (AlphaFold-Multimer via ColabFold)
2. Scores and clusters structural hypotheses
3. Optionally **truncates HLA to the binding groove** to reduce RFdiffusion compute
4. Generates **minibinder backbones** conditioned on the target (RFdiffusion)
5. Validates designs with **AlphaFold-Multimer**
6. Ranks candidates with an **XGBoost learning-to-rank** model over biological, structural, and embedding features

The output is a ranked table of peptide–allele–binder triplets ready for experimental follow-up.

---

## Architecture

This is not a single end-to-end model. It integrates four systems with explicit separation of concerns:

| System | Stage | Tool | Role |
|--------|-------|------|------|
| Structural uncertainty | 1 | ColabFold / AlphaFold-Multimer v3 | Peptide–HLA complex modeling; ensemble of conformations |
| Target preparation | 1.5 *(optional)* | BioPython | HLA groove truncation for diffusion (preserves full Step 1 PDBs) |
| Generative design | 3 | RFdiffusion | Novel binder backbone generation on truncated target |
| Structural validation | 4 | ColabFold multimer | Re-fold binder + target; score engagement |
| Prioritization | 5 | XGBoost (`rank:pairwise`) | Multi-modal ranking over biological + structural + embedding features |

**Design principle:** biological presentation scores (MHCflurry, NetMHCpan) are treated as *priors*, structural scores measure *model reliability*, and the ML ranker integrates everything at the end. No hand-tuned weighted score before Step 5.

```
neoJunction / cohort TSV
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
├──────────────────────────────────────────────────────────────┤
│  4   Validation          AlphaFold-Multimer re-folding       │
├──────────────────────────────────────────────────────────────┤
│  5   ML ranking          XGBoost over multi-modal features   │
│                          → final_rankings.tsv                │
└──────────────────────────────────────────────────────────────┘
```

---

## Tech stack

| Layer | Components |
|-------|------------|
| Structure prediction | ColabFold, AlphaFold2-Multimer v3, JAX/CUDA |
| Generative design | RFdiffusion (SE(3)-equivariant diffusion) |
| Protein language models | ESM-2 (650M), optional ProtT5 |
| ML ranking | XGBoost learning-to-rank, PCA feature compression, gene-level train/test split |
| Orchestration | Python, YAML config, SLURM, restart-safe manifests |
| Structure I/O | BioPython, custom PDB parsing utilities |

---

## Quick start (Longleaf)

### 1. Environment

```bash
conda env create -f environment.yml
conda activate neo_binder
```

Install external tools separately (see [External dependencies](#external-dependencies)).

### 2. Set paths

All large outputs go on `/work`, not `$HOME`:

```bash
export PROJECT_ROOT=/work/users/$USER/minibinder_prediction   # or your project root
export NEO_BINDER_WORK_ROOT=$PROJECT_ROOT/work
cd $PROJECT_ROOT/binding-prediction
```

### 3. Prefetch ColabFold weights (once, login node)

```bash
bash scripts/prefetch_colabfold_weights.sh
```

Weights and cache are redirected to `/work` via `scripts/colabfold_work_paths.sh` to avoid home quota issues.

### 4. Prepare input

Tab-separated file at `data/step5_input.tsv` (see [Input format](#input-format)). Point to your cohort:

```bash
export INPUT_TSV=data/generated/AIMDLVMMV_HLA_A0201.tsv   # example single pair
```

### 5. Submit steps

```bash
./slurm/submit_step.sh 0      # embeddings (GPU; can run parallel with step 1)
./slurm/submit_step.sh 1      # ColabFold peptide–HLA multimer
./slurm/submit_step.sh 1.5    # optional: truncate HLA groove for RFdiffusion
./slurm/submit_step.sh 2      # scoring + clustering
./slurm/submit_step.sh 3      # RFdiffusion
./slurm/submit_step.sh 4      # binder validation
./slurm/submit_step.sh 5      # ML ranking
```

After each step completes, inspect outputs:

```bash
python utils/step_summary.py --step step1
python utils/step_summary.py --step step1_5
```

Monitor jobs:

```bash
squeue -u $USER
tail -f logs/step1_*.out
```

---

## Step 1: Peptide–HLA multimer

Step 1 runs ColabFold with **colon-separated multimer FASTA** (`PEPTIDE:HLA`) so peptide and HLA fold as a single complex, not separate monomers.

Key settings in `config/config.yaml`:

```yaml
step1:
  backend: colabfold
  colabfold_model_type: alphafold2_multimer_v3
  colabfold_pair_mode: unpaired_paired
  colabfold_min_chains: 2
```

Outputs:

| Path | Description |
|------|-------------|
| `work/step1_structures/<job_id>/` | Multimer PDBs, PAE plots, score JSON |
| `work/step2_scored/parsed_structures.tsv` | Parsed complex table (written at end of Step 1) |

Inspect the best model in Mol\*: upload `*_rank_001_*multimer*.pdb` (combined complex, chains A/B).

---

## Step 1.5: HLA groove truncation (optional)

RFdiffusion does not need the full HLA heavy chain or β2-microglobulin. Step 1.5 writes **separate truncated PDBs** without modifying Step 1 outputs.

**Keep:**
- All of peptide chain (~9 residues)
- HLA chain residues **25–180** (α1/α2 binding groove; skips disordered N-terminal low-pLDDT region)

```bash
./slurm/submit_step.sh 1.5
# or locally:
python run_pipeline.py --step step1_5
```

Outputs:

```
work/step1_5_truncated/
  <job_id>/
    <job_id>_truncated.pdb
  truncated_structures.tsv    # includes pLDDT stats for kept residues
  truncation_status.tsv
```

Step 3 automatically prefers truncated PDBs when `step1_5.enabled: true` and the manifest exists. Set `step1_5.enabled: false` in config to skip entirely.

### Hotspot selection (Step 3)

Before RFdiffusion runs, `step3_rfdesign/select_peptide_hotspots.py` picks **5–6 peptide hotspots** for `ppi.hotspot_res`:

- **Skip** MHC-I anchors (P2, P9) — buried in the groove
- **Skip** Pro/Gly and N-terminal Ala/Gly
- **Prefer** charged (Asp/Glu/Lys/Arg) and bulky hydrophobics (Trp/Phe/Tyr/Met/Leu) in the exposed P4–P8 stretch
- **Require** ≥3 hydrophobic hotspots in the final set

For `AIMDLVMMV` this yields `A4,A5,A6,A7,A8` (Asp + Leu/Val/Met/Met).

Override in `config/config.yaml`:

```yaml
step3:
  hotspots:
    auto_select: true
    manual_hotspots: [A4,A5,A6,A7,A8]   # optional override
```

---

## Work directory layout

```
$NEO_BINDER_WORK_ROOT/
├── inputs/step1/              FASTA + manifest
├── step1_structures/          Full ColabFold multimer outputs
├── step1_5_truncated/         Groove-truncated PDBs (optional)
├── step2_scored/              Metrics, clusters, rankings
├── step3_binders/             RFdiffusion designs + contigs
├── step4_validated/           Multimer validation scores
├── step5_ranked/              Feature matrix + final rankings
└── embeddings/                ESM-2 / ProtT5 vectors
```

---

## Input format

`data/step5_input.tsv` — one row per peptide–allele pair:

| Column | Description |
|--------|-------------|
| `peptide` | Neoantigen peptide sequence |
| `allele` | HLA allele ID (e.g. `HLA_A0201`) |
| `gene` | Source gene (used for leakage-safe ML split) |
| `junction` | Junction identifier |
| `mhcflurry_presentation_percentile` | MHCflurry presentation percentile |
| `netmhcpan_EL_rank` | NetMHCpan EL rank |
| `n_carriers_in_cohort` | Patients carrying this neoantigen |
| `PSR_tumor` | Tumor cell expression proportion |
| `frameshift_flag` | 1 if frameshift-derived |

HLA sequences are resolved automatically from IMGT (`hla/setup_hla.py` runs on first job if the index is missing).

---

## Key outputs

| File | Stage | Description |
|------|-------|-------------|
| `step2_scored/parsed_structures.tsv` | 1 | Parsed multimer complexes with pLDDT/PAE |
| `step1_5_truncated/truncated_structures.tsv` | 1.5 | Truncated PDB paths + pLDDT summary |
| `step2_scored/ranked_structures.tsv` | 2 | Top structures per peptide with `structure_confidence_score` |
| `step3_binders/binder_designs.tsv` | 3 | RFdiffusion backbone PDBs + sequences |
| `step4_validated/binder_scores.tsv` | 4 | `binder_structural_score` per design |
| `step5_ranked/final_rankings.tsv` | 5 | **Final ranked candidates** |
| `step5_ml_model/model.pkl` | 5 | Trained XGBoost ranker |

---

## Scoring semantics

| Score | Stage | Meaning |
|-------|-------|---------|
| `plddt_mean`, `pae_mean` | 1 | Per-residue confidence from ColabFold |
| `structure_confidence_score` | 2 | Reliability of peptide–HLA model (pLDDT, PAE, cluster agreement) |
| `binder_structural_score` | 4 | Structural engagement of designed binder with target |
| `binder_score` | 5 | ML-integrated rank across all feature modalities |

**None of these scores are binding affinity or immunogenicity directly.** They inform prioritization under uncertainty.

---

## ML features (Step 5)

The ranker combines three feature groups (see `config/config.yaml` → `features`):

- **Biological:** MHCflurry/NetMHCpan presentation, cohort prevalence, PSR, frameshift flag
- **Structural:** confidence scores from Steps 2 and 4, interface metrics, optional Rosetta energies
- **Embedding:** ESM-2 1280D → PCA 50D, optional ProtT5, cosine similarity to IEDB known binders

Training uses `rank:pairwise` with gene-level train/test split to reduce leakage across junctions from the same gene.

---

## Pipeline execution (local)

```bash
# Full pipeline (Steps 0–5; Step 1.5 is optional — run separately)
python run_pipeline.py --input data/step5_input.tsv

# Single step
python run_pipeline.py --step step1_5

# Resume from a step
python run_pipeline.py --from-step step3

# Dry run
python run_pipeline.py --dry-run
```

### Restart-safe execution

Each step writes status TSVs and `done.flag` files. Re-submitting a SLURM job skips completed work. To force a rerun, remove the relevant job directory and status entry.

---

## External dependencies

| Tool | Stages | Setup |
|------|--------|-------|
| **ColabFold** | 1, 4 | `scripts/setup_colabfold_longleaf.sh`; env at `$PROJECT_ROOT/alphafoldenv` |
| **RFdiffusion** | 3 | `scripts/setup_rfdiffusion_longleaf.sh`; separate `SE3nv` env |
| **PMGen** *(alternative Step 1)* | 1 | `scripts/setup_pmgen_longleaf.sh` |
| **CUDA** | GPU steps | `module load cuda` on Longleaf |

### Data files

| File | Purpose |
|------|---------|
| `data/iedb_reference.csv` | IEDB positive MHC-I binders for embedding similarity priors |
| `config/hla_sequences.fasta` | Legacy fallback; IMGT auto-download preferred |

### Longleaf notes

- Use `--qos=gpu_access` for GPU jobs (set in `slurm/*.sbatch`)
- Redirect ColabFold cache to `/work`: handled by `scripts/colabfold_work_paths.sh`
- Unset inherited `SBATCH_QOS` before submit: handled by `submit_step.sh`
- ColabFold JAX stack: see `scripts/repair_colabfold_jax_haiku.sh` if import errors occur

---

## Configuration

Edit `config/config.yaml`:

| Section | Controls |
|---------|----------|
| `runtime.profile` | `optimized` (default) vs `full` accuracy settings |
| `step1` | ColabFold vs PMGen, model count, recycle steps |
| `step1_5` | Truncation residue range, RFdiffusion target selection |
| `step2` | Contact distance, RMSD cluster threshold, top-N structures |
| `step3` | Binder length, designs per structure, diffusion steps |
| `step4` | Multimer validation model count, complexes per peptide cap |
| `step5` | XGBoost hyperparameters, PCA dimensions, split strategy |
| `embeddings` | ESM-2 model, optional ProtT5 |
| `slurm` | Partition, memory, time limits |

Paths expand from `$NEO_BINDER_WORK_ROOT` at runtime.

---

## Storage estimates (~100 peptides)

| Data | Size |
|------|------|
| AlphaFold structures | 50–200 GB |
| RFdiffusion designs | 10–50 GB |
| Embeddings | 1–5 GB |
| TSV manifests | < 100 MB |

Use `/work/users/<onyen>/`, not `$HOME`.

---

## Repository layout

```
binding-prediction/
├── run_pipeline.py              Orchestrator
├── config/config.yaml           Pipeline configuration
├── slurm/                       SLURM batch scripts + submit_step.sh
├── step1_structure_generation/  ColabFold / PMGen inputs + parsing
├── step1_5_structure_prep/      HLA groove truncation
├── step2_structure_scoring/     Metrics, clustering, ranking
├── step3_rfdesign/              RFdiffusion contigs + inference
├── step4_binder_validation/     Multimer validation
├── step5_ml_model/              Feature engineering + XGBoost
├── embeddings/                  ESM-2 / ProtT5
├── hla/                         IMGT allele resolution
├── utils/                       Shared helpers, step summaries
└── scripts/                     HPC setup, ColabFold env, prefetch
```

---

## License

Academic / research use. See individual tool licenses for ColabFold, RFdiffusion, ESM-2, and XGBoost.
