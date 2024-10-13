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
1. Clone/[download](https://github.com/wlab-bio/vdnamic#:~:text=with%20GitHub%20Desktop-,Download%20ZIP) the repository:
2. Install dependencies:
pip install -r requirements.txt
3. Run the sequence analysis:
python main.py lib /path/to/data//
4. Perform spatial reconstruction:
python main.py GSE -path /path/to/data//uei_grp0// -inference_dim [INT] -inference_eignum [INT] -final_eignum [INT]

## Sample Data Sets
For simulated data for GSE (spatial reconstruction) alone, we recommend using one of the simulated data set in [the Sample Directory](https://github.com/wlab-bio/vdnamic/tree/main/SAMPLE_DIRECTORY).

For sample end-to-end (sequencer-to-image) analysis, we recommend using a small data set from 
Weinstein, J. A., Regev, A., & Zhang, F. (2019). DNA microscopy: optics-free spatio-genetic imaging by a stand-alone chemical reaction. Cell, 178(1), 229-241. 
Fastq's can most easily be downloaded [here](https://uchicago.box.com/s/go4vhae97to7ozu4nwdcq77eah1tlf6x).

## Documentation
For detailed usage instructions, parameter explanations, and troubleshooting tips, please refer to the [Documentation](https://github.com/wlab-bio/vdnamic/blob/main/vdnamic_documentation.pdf).

## Requirements
- Python 3.8+
- NumPy, SciPy, Pandas, Biopython
- awk
- bioawk
- STAR Aligner 2.7.9a+ (if using genome alignment wrapper)

## Citation
If you use vDNAmic in your research, please cite:
Qian, N., & Weinstein, J. A. (2023). Volumetric imaging of an intact organism by a distributed molecular network. bioRxiv. https://doi.org/10.1101/2023.08.11.553025

## License
This project is licensed under the MIT License.

## Contact
For questions and support, please contact jaweinst@uchicago.edu.
