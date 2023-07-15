# A curated census of pathogenic and likely pathogenic UTR variants and evaluation of deep learning models for variant effect prediction

Authors: *Emma Bohn, Tammy Lau, Omar Wagih, Tehmina Masud, Daniele Merico*

Preprint available now at medrxiv: https://doi.org/10.1101/2023.07.10.23292474

This repository accompanies our publication by making public the code that runs the
deep learning-based artificial intelligence (DL-AI) models to generate variant effect predictions
and the code to generate the figures in the paper.

The models are:

* [Saluki](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-022-02811-x) - mRNA stability (3' UTR)
* [Enformer](https://www.nature.com/articles/s41592-021-01252-x) - transcription (5' UTR)
* [FramePoolCombined](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008982) - ORF recognition by the translation machinery  (5' UTR)


## To reproduce

### Download data

The `data` directory already contains the 5' and 3' UTR P/LP benchmarks.

Download the following files/install the following repos into `data`:

```
mkdir -p data/
cd data/

# hg38 reference
$ wget -O - http://hgdownload.cse.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz | gunzip -c > hg38.fa
# index the fasta file
$ samtools faidx hg38.fa

# most recent RefSeq annotations
# see https://linear.app/dg-eng/issue/TID-54#comment-a86a90a9 for more details
$ wget http://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/genes/hg38.ncbiRefSeq.gtf.gz

# UCSC phyloP 100way annotations
$ wget https://hgdownload.cse.ucsc.edu/goldenpath/hg38/phyloP100way/hg38.phyloP100way.bw

# Repo for Karollus - Needed to actually use the model
$ mkdir -p repos/
$ cd repos
$ git clone git@github.com:Karollus/5UTR.git

# Repo for basenji - Needed for conda env purposes
$ git clone git@github.com:calico/basenji.git

# Dataset from zenodo for Saluki - Needed to actually use the model
$ mkdir saluki
$ cd saluki
$ wget https://zenodo.org/api/files/98281317-fd0b-4c7f-891d-0d246a99737a/datasets.zip
$ unzip datasets.zip

```

The final structure of the `data` directory should look like this:

```
- data/
  - hg38.fa
  - hg38.fa.fai
  - hg38.ncbiRefSeq.gtf.gz
  - hg38.phyloP100way.bw
  - utr5_plp_benchmark.tsv
  - utr3_plp_benchmark.tsv
  - repos/
    - 5UTR/
    - basenji/
    - saluki/
      - datasets/
```

### Create conda environments

To avoid conflicts with the requirements for the different predictors, multiple
conda environments were created and used.

* `utr_curation_manuscript` - for running Enformer predictions, adding phyloP annotations, and creating figures
```
$ mamba env create --file env_general.yml
```

* `framepool` - for running FramePoolCombined
```
$ mamba env create --file env_karollus.yml
```

* `saluki` - for running Saluki
```
$ mamba env create --file env_saluki.yml
$ mamba activate saluki
$ mamba install tensorflow-gpu
$ cd data/repos/basenji
$ python setup.py develop --no-deps
```

### Commands to generate predictions

Note that for results presented in the manuscript, this code was run on a machine with the following specs:

* Kernel: GNU/Linux 5.4.0-146-generic x86_64
* OS: Ubuntu 20.04.4 LTS
* NVIDIA Driver version: 510.108.03
* CUDA version 11.6

To reproduce, run the following commands:

```
$ mkdir -p output/
$ mkdir -p plots/

$ mamba activate utr_curation_manuscript
$ python 01a_generate_enformer_predictions.py

$ mamba deactivate utr_curation_manuscript
$ mamba activate framepool
$ python 02a_generate_framepool_predictions.py

$ mamba deactivate framepool
$ mamba activate saluki
$ python 03a_generate_saluki_predictions.py --train_gru_dir data/repos/saluki/datasets/deeplearning/train_gru/

$ mamba deactivate saluki
$ mamba activate utr_curation_manuscript
$ python 04_phylop_annotations.py

# open 05_analyze_predictors.ipynb and run all cells
```

Description of each script is here:

* `01a_generate_enformer_predictions.py`
  * Generates raw predictions using Enformer loaded from tfhubdev
  * Predictions (all tracks) are stored in an output shelve file (that needs to be processed after)
* `02a_generate_karollus_predictions.py`
  * Pre-processes the GTF file to create a .bed file of 5' UTR regions
  * Runs FramePoolCombined via `kipoi`
  * Predictions are stored in an output .tsv
* `03a_generate_saluki_predictions.py`
  * Runs Saluki, constructing the 6D tracks for each variant and using the models provided
  in the downloaded datasets file from zenodo
  * Predictions are stored in an output .tsv (one prediction per model that needs to be averaged after)
* `04_phylop_annotations.py`
  * Annotates with PhyloP scores from the downloaded bigwig file
  * annotated variants are stored in an output .tsv
* `05a_analyze_predictors.ipynb`
  * Post-processes predictions and creates figures 3, S2-S6 for the paper
* `05b_UTRManuscriptFigures.Rmd`
  * Creates Figures 2 and S1 for the paper
