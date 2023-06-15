"""
To be run with the `utr_curation_manuscript` conda environment activated.
Run from the repo root directory.
"""

import pyBigWig

import numpy as np
import pandas as pd

PHYLOP_BW = 'data/hg38.phyloP100way.bw'
UTR3_FILE = 'data/utr3_plp_benchmark.tsv'
UTR5_FILE = 'data/utr5_plp_benchmark.tsv'

bw = pyBigWig.open(PHYLOP_BW)

utr3_df = pd.read_csv(UTR3_FILE, sep='\t')
utr5_df = pd.read_csv(UTR5_FILE, sep='\t')
utr3_df['region'] = 'utr3'
utr5_df['region'] = 'utr5'

utr_df = pd.concat([utr5_df, utr3_df])

def annotate_phylop(variant_str):
    chr, pos, ref, alt = variant_str.split(':')
    # bigwig is 0 based
    start = int(pos) - 1
    end = start + len(ref)
    score = bw.stats(chr, start, end, type="max")
    return score[0]

utr_df['phylop_100way'] = utr_df['variant'].apply(annotate_phylop)
utr_df.to_csv('output/phylop_annotations_20230222.tsv', sep='\t', index=False)