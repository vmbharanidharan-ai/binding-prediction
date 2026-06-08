# Neo Binder Pipeline

**Peptide–HLA structure → RFdiffusion binder design → ML ranking**

A production-grade computational immunoinformatics pipeline for neoantigen-derived peptide prioritization and minibinder design. Built for UNC Longleaf HPC with modular, restart-safe execution.

---

## Overview

The pipeline processes neoantigen peptides (from neoJunction Step 5 output) and ranks them for minibinder design using structural modeling, generative protein design, and machine learning.

```
neoJunction Step 5 TSV
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  STAGE 0: Embeddings (ESM-2 + ProtT5)                  │  ← always runs
├─────────────────────────────────────────────────────────┤
│  STAGE 1: Peptide–HLA structure (ColabFold)          │
│           → 5 models per pair, ensemble hypotheses      │
├─────────────────────────────────────────────────────────┤
│  STAGE 2: Structural scoring + clustering             │
│           → interface metrics, RMSD clusters, ranking   │
│           → structure_confidence_score (reliability)    │
├─────────────────────────────────────────────────────────┤
│  STAGE 3: RFdiffusion minibinder design               │
│           → backbone PDBs + sequences                   │
├─────────────────────────────────────────────────────────┤
│  STAGE 4: Binder validation (AlphaFold-Multimer)     │
│           → binder_structural_score                     │
├─────────────────────────────────────────────────────────┤
│  STAGE 5: XGBoost Ranker (rank:pairwise)              │
│           → biological + structural + embedding features│
│           → final ranked candidates                     │
└─────────────────────────────────────────────────────────┘
```

### What this pipeline is

This is **not** a single model. It is four integrated systems:

1. **Structural uncertainty system** — peptide–HLA ensemble modeling via AlphaFold
2. **Generative protein design system** — RFdiffusion minibinders
3. **Structural validation system** — AlphaFold-Multimer re-scoring
4. **Biological prioritization system** — XGBoost ranker over multi-modal features

---

## Biological Interpretation

### Peptide–HLA presentation

Neoantigen peptides must be presented on MHC class I molecules to be visible to CD8+ T cells. The pipeline uses `mhcflurry_presentation_percentile` and `netmhcpan_EL_rank` as biological priors — these estimate whether a peptide is likely presented, not whether a binder will work.

### Binder design concept

Minibinders are small protein scaffolds (~50–80 aa) designed to recognize a specific peptide–HLA complex. RFdiffusion generates novel binder backbones conditioned on the target structure. AlphaFold-Multimer then validates whether the designed binder physically engages the target.

### Immunogenic prioritization

The final XGBoost ranker integrates:
- **Biological features**: presentation scores, tumor prevalence (PSR), cohort carrier count, frameshift flag
- **Structural features**: confidence scores from Stages 2 and 4 (reliability, not affinity)
- **Embedding features**: ESM-2 (1280D → PCA 50D), ProtT5, cosine similarity to IEDB known binders

**Critical design rule**: No arbitrary weighted biological scoring before Step 5. Structure scores measure *reliability*, not binding affinity.

---

## Installation

### Conda environment

```bash
conda env create -f environment.yml
conda activate neo_binder
```

### External dependencies (install separately on HPC)

| Tool | Purpose | Notes |
|------|---------|-------|
| **ColabFold** | Stages 1 & 4 | `pip install colabfold[alphafold]` or module |
| **RFdiffusion** | Stage 3 | Separate env (`SE3nv`), download weights |
| **CUDA** | GPU steps | `module load cuda` on Longleaf |

### Required data files (user-provided)

1. **HLA sequences** — `config/hla_sequences.fasta`
   - Source: [IMGT/HLA](https://www.ebi.ac.uk/ipd/imgt/hla/) or curated allele FASTA
   - Template provided; extend with your cohort alleles

2. **IEDB reference** — `data/iedb_reference.csv`
   - Download from [IEDB](https://www.iedb.org/): positive MHC-I binders, Homo sapiens
   - Used for embedding similarity priors in ML stage

3. **RFdiffusion weights** — set `RFDIFFUSION_WEIGHTS` env var

---

## Cluster Usage (UNC Longleaf)

### Setup

```bash
# On Longleaf login node
module load cuda
conda activate neo_binder

# Set work directory (NOT home — structures are large)
export NEO_BINDER_WORK_ROOT=/work/users/$USER/neo_binder
mkdir -p $NEO_BINDER_WORK_ROOT
```

### Directory layout on `/work`

```
/work/users/<onyen>/neo_binder/
├── inputs/step1/          # ColabFold FASTA inputs
├── step1_structures/      # AlphaFold PDB outputs (~GB per peptide)
├── step2_scored/          # Metrics, clusters, rankings
├── step3_binders/         # RFdiffusion designs
├── step4_validated/       # Multimer validation
├── step5_ranked/          # Final ML rankings
└── embeddings/            # ESM-2 + ProtT5 vectors
```

### Submit jobs

```bash
# Step 1: Structure generation (GPU, ~24h)
sbatch step1_structure_generation/slurm_step1.sbatch

# Step 2: Scoring + clustering (CPU, ~12h)
sbatch step2_structure_scoring/slurm_step2.sbatch

# Step 3: RFdiffusion (GPU, ~24h)
sbatch step3_rfdesign/slurm_step3.sbatch

# Step 4: Binder validation (GPU, ~24h)
sbatch step4_binder_validation/slurm_step4.sbatch
```

### Monitor jobs

```bash
squeue -u $USER
tail -f logs/step1_<jobid>.out
tail -f logs/step1_<jobid>.err
```

### Restart-safe execution

Every step checks for `done.flag` files and completed TSV entries. Re-submitting a SLURM job skips finished work automatically.

---

## Pipeline Execution

### Full pipeline (local or login node orchestration)

```bash
python run_pipeline.py --input data/step5_input.tsv
```

### Run specific steps

```bash
# Embeddings only (GPU recommended)
python run_pipeline.py --steps embeddings

# Resume from Step 3
python run_pipeline.py --from-step step3

# Dry run (print commands)
python run_pipeline.py --dry-run
```

### Individual step commands

```bash
# Step 1
python step1_structure_generation/generate_inputs.py \
    --input data/step5_input.tsv --output-dir work/inputs/step1
python step1_structure_generation/run_colabfold.py \
    --manifest work/inputs/step1/input_manifest.tsv --output-dir work/step1_structures

# Step 5
python step5_ml_model/feature_engineering.py --input data/step5_input.tsv --output work/step5_ranked/features.tsv
python step5_ml_model/train_xgboost_ranker.py --features work/step5_ranked/features.tsv
python step5_ml_model/inference.py --features work/step5_ranked/features.tsv --output work/step5_ranked/final_rankings.tsv
```

---

## Input Format

`data/step5_input.tsv` (tab-separated):

| Column | Description |
|--------|-------------|
| `peptide` | Neoantigen peptide sequence |
| `allele` | HLA allele (e.g. `HLA_A0201`) |
| `gene` | Source gene |
| `junction` | Junction identifier (for leakage-safe ML split) |
| `mhcflurry_presentation_percentile` | MHCflurry presentation score |
| `netmhcpan_EL_rank` | NetMHCpan EL rank |
| `n_carriers_in_cohort` | Number of patients with this neoantigen |
| `PSR_tumor` | Proportion of tumor cells expressing variant |
| `frameshift_flag` | 1 if frameshift-derived, 0 otherwise |

---

## Outputs

| File | Stage | Description |
|------|-------|-------------|
| `step2_scored/ranked_structures.tsv` | 2 | Top 1–3 structures per peptide with `structure_confidence_score` |
| `step3_binders/binder_designs.tsv` | 3 | RFdiffusion backbone PDBs + sequences |
| `step4_validated/binder_scores.tsv` | 4 | `binder_structural_score` per design |
| `step5_ranked/final_rankings.tsv` | 5 | **Final output**: peptide, allele, binder_score, rank, confidence |
| `step5_ml_model/model.pkl` | 5 | Trained XGBoost ranker |

### Final rankings TSV

```
peptide    allele      gene   junction  binder_score  rank  confidence
SIINFEKL   HLA_A0201   EGFR   J1        0.87          1     0.95
GILGFVFTL  HLA_A0201   MART1  J2        0.72          2     0.78
...
```

---

## Scoring Philosophy

| Score | Stage | Meaning |
|-------|-------|---------|
| `structure_confidence_score` | 2 | How reliable is the peptide–HLA model? (pLDDT, PAE, cluster agreement) |
| `binder_structural_score` | 4 | How well does the binder engage the target structurally? |
| `binder_score` | 5 | ML-integrated ranking across all feature modalities |

**No score before Step 5 represents binding affinity or immunogenicity directly.**

---

## Configuration

Edit `config/config.yaml` to adjust:
- Model counts, recycle steps, cluster thresholds
- RFdiffusion binder length and design count
- XGBoost hyperparameters
- SLURM partition/memory/time
- Work directory paths

---

## Storage Planning

| Data type | Estimated size (100 peptides) |
|-----------|--------------------------------|
| AlphaFold structures | 50–200 GB |
| RFdiffusion designs | 10–50 GB |
| Embeddings | 1–5 GB |
| TSV manifests | < 100 MB |

**Always use `/work/users/<onyen>/` on Longleaf, not `$HOME`.**

---

## License

Academic / research use. See individual tool licenses for ColabFold, RFdiffusion, ESM-2, and XGBoost.
