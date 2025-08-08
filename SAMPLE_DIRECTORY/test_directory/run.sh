ECHO "Extracting"

gunzip *

python vdnamic_fastq_sim_patched.py \
  --build-from-posfile ./pos.csv \
  --rescale 2.0 \
  --mperPt 10 \
  --neg-bin-p 0.8 \
  --amp-dispersion 0.0 \
  --dropout0 0.0 \
  --dropout1 0.0 \
  --encode-digits ACGT \
  --encode-width 10 \
  --cdna-settings cdna.lib.settings \
  --uei-settings uei.lib.settings \
  --avg-reads-uei 4.5 \
  --min-reads-uei 1 \
  --avg-reads-cdna 3.0 \
  --min-reads-cdna 1 \
  --use-uei-weights \
  -o sim_fastq \

ECHO "Completed random sequence-generator"

mv R1_uei.fastq sim_fastq/.
mv R2_uei.fastq sim_fastq/.
mv R1_cdna.fastq sim_fastq/.
mv R2_cdna.fastq sim_fastq/.
mkdir sim_fastq/cdna
mkdir sim_fastq/uei
cp -p cdna.lib.settings sim_fastq/cdna/lib.settings
cp -p uei.lib.settings sim_fastq/uei/lib.settings

ECHO "Initiating UEI analysis"
python ../../main.py lib sim_fastq/uei/
ECHO "Initiating cDNA analysis"
python ../../main.py lib sim_fastq/cdna/

ECHO "Initiating image inference"
python ../../main.py GSE -path sim_fastq/uei/uei_grp0/. -inference_dim 2 -inference_eignum 15 -final_eignum 100 -calc_final ../.

ECHO "Plotting"
python plot.py

ECHO "Test complete"
