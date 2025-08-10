# vDNAmic Pipeline 
## Overview
vDNAmic is a comprehensive data analysis pipeline for DNA microscopy experiments. It provides tools for processing raw sequencing data, performing spatial reconstruction, and analyzing gene expression profiles.
## Key Features
- Quality control and filtering of raw sequencing reads
- UMI/UEI extraction and clustering to correct sequencing errors
- Consensus sequence assembly and optional genome alignment
- Spatial reconstruction using Geodesic Spectral Embedding (GSE)
- Built-in FASTQ data simulator for end-to-end testing
- Integration with genome alignment tools

---

> **Repo:** https://github.com/wlab-bio/vdnamic  
> **Full docs (PDF):** [vdnamic_documentation.pdf](./vdnamic_documentation.pdf)

## Contents
- [Installation](#installation)
- [Verification (5-Minute Test)](#verification-5-minute-test)
- [Quickstart](#quickstart)
- [Input layout & config](#input-layout--config)
- [Sequence analysis (`lib`)](#sequence-analysis-lib)
- [Spatial reconstruction (GSE)](#spatial-reconstruction-gse)
- [FASTQ simulator](#fastq-simulator)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

## Installation

### Prerequisites
- **Python** 3.8+ (Linux/macOS recommended)
- **System tools**: `awk`, `bioawk`, `sed`
- **Optional aligner**: **STAR** ≥ 2.7.9a (if aligning amplicons to a genome)

### Create environment & install dependencies
```bash
# (Recommended) create and activate a virtual environment
python -m venv .venv && source .venv/bin/activate
# or: conda create -n vdnamic python=3.10 -y && conda activate vdnamic

# Install Python dependencies
pip install -r requirements.txt
````

`requirements.txt` includes:

```
numpy>=1.21.0
scipy>=1.7.0
pandas>=1.3.0
biopython>=1.79
scikit-learn>=0.24.0
faiss-cpu>=1.7.0
annoy>=1.17.0
pymetis>=2023.1
numba>=0.53.0
joblib>=1.0.0
matplotlib>=3.4.0
```

### Recommended hardware

* 4+ CPU cores
* 16+ GB RAM
* 100+ GB free disk (dataset-dependent)

---

## Verification (5-Minute Test)

Run a complete end-to-end check (simulate data → run `lib` → run `GSE` → make a validation plot):

```bash
# From the root of the repository
cd SAMPLE_DIRECTORY/test_directory/ && sh run.sh
```

A successful run prints **`TESTS PASSED`** and writes `scatter_plot_with_procrustes.png` in the same directory.

> If your local clone doesn’t include this harness, see the **Quickstart** below or the PDF docs for a manual mini-test.

---

## Quickstart

1. **Clone & enter repo**

```bash
git clone https://github.com/wlab-bio/vdnamic.git
cd vdnamic
```

2. **Install deps** (see [Installation](#installation)).

3. **Prepare inputs**
   Create a data directory with `R1.fastq`/`R2.fastq` and an analysis directory with `lib.settings`.
   **Important:** Working directory arguments to vDNAmic should end with a **double slash** (`//`).

4. **Run sequence analysis (`lib`)**

```bash
python main.py lib /path/to/analysis//
```

5. **Run spatial reconstruction (GSE)**
   `-calc_final` location to search for label_pt0.txt and label_pt1.txt for annotation of coordinates in final_labels.txt

```bash
python main.py GSE -path /path/to/analysis// -inference_dim 2 -calc_final /path/to/labels
```

6. **Visualize**
   Use `./out/final_coords.txt` and `./out/final_labels.txt` for plotting and downstream analysis.

---

## Input layout & config

### Directory layout

```
/path/to/data//
├── R1.fastq
└── R2.fastq

/path/to/analysis//
└── lib.settings
```

### `lib.settings` essentials (examples)

* **FASTQ sources**

  ```
  -source_for ../data/R1.fastq
  -source_rev ../data/R2.fastq
  ```
* **Sequence format grammar**

  ```
  -seqform_for U_NWNN..._0:31|U_AGGN..._39:76
  -seqform_rev U_GCTN..._2:39
  ```
* **UMI/UEI definitions**

  ```
  -u0 ...
  -u1 ...
  -u2 0,0,0+0,0,2:revcomp
  ```
* **Quality & filtering**

  ```
  -min_mean_qual 30
  -min_reads_per_assoc 2
  ```

---

## Sequence analysis (`lib`)

Parses FASTQs, extracts UMIs/UEIs and amplicons, clusters to correct errors, assembles consensuses, and prepares inputs for inference and optional alignment.

**Run**

```bash
python main.py lib /path/to/analysis//
```

**Notable outputs**

* `uei_grp*//` — per-group association graphs for downstream GSE
* `uxi*.txt` — UMI/UEI sequence tables
* `readcounts.txt` — processing summary
* `rejected.txt.gz` — rejected reads (compressed)
* (Optional) `STARalignment*/` — alignments if STAR is configured

---

## Spatial reconstruction (GSE)

Geodesic Spectral Embedding (GSE) computes spatial coordinates from the UMI association graph produced by `lib`.

**Typical run**

```bash
python main.py GSE \
  -path /path/to/analysis// \
  -inference_dim 2 \
  -inference_eignum 30 \
  -final_eignum 225 \
  -sub_num 30 \
  -sub_size 15000 \
  -ncpus 10 \
  -filter_criterion 5 \
  -calc_final ../path/to/labels
```

**Common flags**

* `-inference_dim` — embedding dimension (2 or 3)
* `-sub_num`, `-sub_size` — number/size of subsamples
* `-ncpus` — CPU cores to use
* `-filter_criterion` — percentile for dropping low-connectivity nodes
* `-calc_final PATH` — location to search for label_pt0.txt and label_pt1.txt for annotation of coordinates in final_labels.txt

> **Tip:** For very large graphs, increase `-sub_size`, reduce `-sub_num`, and consider a modest `-filter_criterion`.

---

## FASTQ simulator

Generate paired-end reads from a ground-truth coordinate file to validate and benchmark the pipeline.

**Recommended usage**

```bash
python vdnamic_fastq_sim_patched.py \
  --build-from-posfile ./pos.csv \
  --rescale 2.0 --mperPt 10 --neg-bin-p 0.8 \
  --encode-digits ACGT --encode-width 10 \
  --cdna-settings cdna.lib.settings \
  --uei-settings uei.lib.settings \
  --avg-reads-uei 4.5 --min-reads-uei 1 \
  -o sim_fastq
```

**Notes**

* Use distinct output FASTQ names per library to avoid collisions.
* The simulator respects quality thresholds (e.g., `-min_mean_qual`) from your settings.

---

## Troubleshooting

* **Too many reads failing quality** — Lower `-min_mean_qual`.
* **Low-complexity UMI over-pruning** — Adjust the filter if valid UMIs are dropped.
* **Memory issues on large graphs** — Reduce `-final_eignum`, tune `-sub_num`/`-sub_size`, or use more RAM.

---

## Citation

If you use vDNAmic in your research, please cite:

> Qian, N., Weinstein, J.A. Spatial transcriptomic imaging of an intact organism using volumetric DNA microscopy. Nat Biotechnol (2025). [https://doi.org/10.1038/s41587-025-02613-z](https://doi.org/10.1038/s41587-025-02613-z)

## License

This project is licensed under the MIT License.

## Contact

For questions and support, please contact [jaweinst@uchicago.edu](mailto:jaweinst@uchicago.edu).
