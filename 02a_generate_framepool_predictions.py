"""
To be run with the `framepool` conda environment activated.
Run from the repo root directory.
"""
import csv
import os

import kipoi
import pybedtools

import pandas as pd

from utils import read_gtf

GTF_PATH = 'data/hg38.ncbiRefSeq.gtf.gz'
BED_PATH = 'data/hg38_ncbiRefSeq.bed'
FASTA_PATH = 'data/hg38.fa'

# create a bed file for Karollus from the gtf
gtf_df = read_gtf(GTF_PATH)
gtf_df = gtf_df.query('feature == "5UTR"')
# format and write output
output = gtf_df[['seqname', 'start', 'end', 'transcript_id', 'score', 'strand']]
output['score'] = 0
bed_file = pybedtools.BedTool.from_dataframe(output)
bed_file = bed_file.sort()
bed_file.saveas(BED_PATH)


# format for VCF
utr5_df = pd.read_csv('data/utr5_plp_benchmark.tsv', sep='\t')
utr5_df[['CHROM', 'POS', 'REF', 'ALT']] = utr5_df['variant'].str.split(':', expand=True)
utr5_df.rename(columns={'variant': 'ID', 'proposed_mechanism': 'INFO'}, inplace=True)
utr5_df['QUAL'] = 30
utr5_df['FILTER'] = 'PASS'
utr5_df = utr5_df[['CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER', 'INFO']]

# set up FramepoolCombined
karollus_model = kipoi.get_model('data/repos/5UTR/kipoi/5UTR_Model', source='dir')
id_column = 4

# because the FramepoolCombined 5' UTR kipoi implementation will insert all variants
# that overlap a 5' UTR as the alternate sequence
# to get individual predictions, we must create a separate VCF for each variant
os.system('mkdir -p data/tmp')
all_results = list()
for _,row in utr5_df.iterrows():
    # write VCF
    variant_id = row['ID']
    tmp_vcf_file = f'data/tmp/tmp_{variant_id}.vcf'
    with open(tmp_vcf_file, 'w') as outfile:
        writer = csv.writer(outfile, delimiter='\t', lineterminator='\n')
        writer.writerow(['##fileformat=VCFv4.1'])
        writer.writerow(['#CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER', 'INFO'])
        writer.writerow(row)
    # bgzip and tabix through cmd line
    os.system(f'bgzip -c {tmp_vcf_file} > {tmp_vcf_file}.gz')
    os.system(f'tabix -p vcf {tmp_vcf_file}.gz')
    # run FramepoolCombined
    tmp_output = f'data/tmp/tmp_{variant_id}_results.tsv'
    karollus_model.pipeline.predict_to_file(
        tmp_output,
        {
            'intervals_file': BED_PATH,
            'fasta_file': FASTA_PATH,
            'vcf_file': f'{tmp_vcf_file}.gz',
            'id_column': id_column,
            'num_chr': False
        },
        batch_size=64,
        #   keep_inputs=True,
        keep_metadata=True
    )
    # read back in results
    results = pd.read_csv(tmp_output, sep='\t')
    all_results.append(results)

results_df = pd.concat(all_results)
results_df.to_csv('output/karollus_predictions_20230221.tsv', sep='\t')
# clean up
os.system('rm -rf data/tmp/')
