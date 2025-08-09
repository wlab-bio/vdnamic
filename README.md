# vDNAmic Pipeline

## Overview
vDNAmic is a comprehensive data analysis pipeline for DNA microscopy experiments. It provides tools for processing raw sequencing data, performing spatial reconstruction, and analyzing gene expression profiles.
## Key Features
- Quality control and filtering of raw sequencing reads
- UMI/UEI extraction and clustering
- Consensus sequence assembly
- Spatial reconstruction using Geodesic Spectral Embedding (GSE)
- Integration with genome alignment tools
## Quick Start

### 1 · Clone the repository
```bash
git clone https://github.com/wlab-bio/vdnamic.git
cd vdnamic
```

## 2 · Standard pip install
```bash
python -m pip install -r requirements.txt
```

</details>

### 3 · Verify installation + end-to-end pipeline (5-7 min test on a laptop)
```bash
cd SAMPLE_DIRECTORY/test_directory/ && sh run.sh
```

### 4 · Analyze your data
```bash
# library construction / UMI clustering
python main.py lib /path/to/data//

# spatial reconstruction (non-hierarchical GSE)
python main.py GSE \
       -path /path/to/data/uei_grp0// \
       -inference_dim 3 \
       -inference_eignum 30 \
       -final_eignum 250
```
## Sample Data Sets
For simulated data for GSE (spatial reconstruction) alone, we recommend using one of the simulated data set in [the Sample Directory](https://github.com/wlab-bio/vdnamic/tree/main/SAMPLE_DIRECTORY).
For sample end-to-end (sequencer-to-image) analysis, we recommend using a small data set from 
Weinstein, J. A., Regev, A., & Zhang, F. (2019). DNA microscopy: optics-free spatio-genetic imaging by a stand-alone chemical reaction. Cell, 178(1), 229-241. 
Fastq's can most easily be downloaded [here](https://uchicago.box.com/s/go4vhae97to7ozu4nwdcq77eah1tlf6x).
## Documentation
See the [PDF manual](./vdnamic_documentation.pdf) .

Key runtime tools:

* STAR ≥ 2.7.9a &nbsp;•&nbsp; samtools ≥ 1.18  
* gawk / bioawk (FASTQ preprocessing)  

## Citation
If you use vDNAmic in your research, please cite:
Qian, N., Weinstein, J.A. Spatial transcriptomic imaging of an intact organism using volumetric DNA microscopy. Nat Biotechnol (2025). https://doi.org/10.1038/s41587-025-02613-z
## License
This project is licensed under the MIT License.
## Contact
For questions and support, please contact jaweinst@uchicago.edu.
