"""
Utilities shared amongst the predictor codes
"""
import kipoiseq
import pyfaidx
import re

import pandas as pd
import numpy as np

##### sequence specific utils
# Source: https://github.com/deepmind/deepmind-research/blob/cb555c241b20c661a3e46e5d1eb722a0a8b0e8f4/enformer/enformer-usage.ipynb

class FastaStringExtractor:

    def __init__(self, fasta_file):
        self.fasta = pyfaidx.Fasta(fasta_file)
        self._chromosome_sizes = {k: len(v) for k, v in self.fasta.items()}

    def extract(self, interval: kipoiseq.Interval, **kwargs) -> str:
        # Truncate interval if it extends beyond the chromosome lengths.
        chromosome_length = self._chromosome_sizes[interval.chrom]
        trimmed_interval = kipoiseq.Interval(interval.chrom,
                                    max(interval.start, 0),
                                    min(interval.end, chromosome_length),
                                    )
        # pyfaidx wants a 1-based interval
        sequence = str(self.fasta.get_seq(trimmed_interval.chrom,
                                          trimmed_interval.start + 1,
                                          trimmed_interval.stop).seq).upper()
        # Fill truncated values with N's.
        pad_upstream = 'N' * max(-interval.start, 0)
        pad_downstream = 'N' * max(interval.end - chromosome_length, 0)
        return pad_upstream + sequence + pad_downstream

    def close(self):
        return self.fasta.close()

def reverse_compliment(seq: str):
    comp = {
        'A': 'T',
        'T': 'A',
        'G': 'C',
        'C': 'G',
        'N': 'N'
    }
    seq = seq[::-1]
    seq = "".join([comp[x] for x in seq])
    return seq


def variant_generator(variant_id):
  """Yields a kipoiseq.dataclasses.Variant from a variant string."""
  chrom, pos, ref, alt = variant_id.split(':')
  return kipoiseq.dataclasses.Variant(
    chrom=chrom, pos=pos, ref=ref, alt=alt, id=variant_id)


def one_hot_encode(sequence):
  return kipoiseq.transforms.functional.one_hot_dna(sequence).astype(np.float32)


##### refseq gtf processing utils

REFSEQ_GTF_COLUMNS = [
    'seqname',
    'source',
    'feature',
    'start',
    'end',
    'score',
    'strand',
    'frame',
    'attribute'
]

def _extract_feature_from_attributes(split_attribute: str, feature_name: str):
    assert split_attribute.startswith(feature_name)
    v = re.findall(r'"([^"]*)"', split_attribute)
    assert len(v) == 1
    return v[0]

def _extract_transcript_id(attributes: str):
    transcript_id = attributes.split('; ')[1]
    transcript_id = _extract_feature_from_attributes(transcript_id, 'transcript_id')
    # the SMARCB1 transcript on the primary chr 22
    # is suffixed with _2, likely because there is also a mapping to the alt contig of chr22
    # which got precedence
    # so strip off any suffixes
    transcript_id = '_'.join(transcript_id.split('_')[0:2])
    return transcript_id

def _extract_exon_number(feature: str, attributes: str):
    if feature == 'transcript':
        return 0
    else:
        exon_number = attributes.split('; ')[2]
        return int(_extract_feature_from_attributes(exon_number, 'exon_number'))

def read_gtf(file_path: str):
    """
    Formats the NCBI Refseq GTF to add a transcript_id and exon_number column.
    """
    data = pd.read_csv(file_path, compression='gzip', names = REFSEQ_GTF_COLUMNS, sep='\t')
    # only keep main chromosomes
    chromosomes = [f'chr{x}' for x in range(1, 23)] + ['chrX', 'chrY']
    data = data.query('seqname in @chromosomes')
    data['transcript_id'] = data['attribute'].apply(_extract_transcript_id)
    data['exon_number'] = data.apply(lambda x: _extract_exon_number(x.feature, x.attribute), axis=1)
    # convert to 0-based indexing
    data['start'] = data['start'] - 1
    return data
