"""
To be run with the `saluki` conda environment activated.
Run from the repo root directory.
"""

import argparse
import json
import shelve
import os
import kipoiseq
import pandas as pd
import numpy as np

from basenji import rnann

from utils import FastaStringExtractor, reverse_compliment, read_gtf

UTR3_VARIANTS_FILE = 'data/utr3_plp_benchmark.tsv'
GTF_FILE = 'data/hg38.ncbiRefSeq.gtf.gz'
FASTA_FILE = 'data/hg38.fa'
MAXLEN = 12288

def _construct_sequences(transcript_df, variant=None):
    # build sequence
    seq_extractor = kipoiseq.extractors.VariantSeqExtractor(
        reference_sequence=FastaStringExtractor(FASTA_FILE)
    )
    cds_exons = sorted(transcript_df.query('feature == "CDS"')['exon_number'].to_list())
    chr = transcript_df.seqname.values[0]
    strand = transcript_df.strand.values[0]
    assert strand in ['+', '-']
    # 5' UTR
    utr5_df = transcript_df.query('feature == "5UTR"')
    utr5_itvs = [kipoiseq.Interval(
        row['seqname'], row['start'], row['end']) for _, row in utr5_df.iterrows()]
    # CDS
    cds_itvs = list()
    for exon_number in range(min(cds_exons), max(cds_exons)+1):
        temp = transcript_df.query('feature == "CDS" & exon_number == @exon_number')
        if exon_number < max(cds_exons):
            cds_itv = kipoiseq.Interval(chr, temp.start.values[0], temp.end.values[0])
        else:
            # add 3 bp to include stop codon
            if strand == '+': 
                cds_itv = kipoiseq.Interval(
                    chr,
                    temp.start.values[0],
                    temp.end.values[0] + 3
                )
            elif strand == '-':
                cds_itv = kipoiseq.Interval(
                    chr,
                    temp.start.values[0] - 3,
                    temp.end.values[0]
                )
        cds_itvs.append(cds_itv)
    if strand == '-':
        cds_itvs.reverse()
    # 3' UTR
    utr3_df = transcript_df.query('feature == "3UTR"')
    utr3_itvs = [kipoiseq.Interval(
        row['seqname'], row['start'], row['end']) for _, row in utr3_df.iterrows()]
    
    # construct sequence
    utr5_seq = ''.join(
        [seq_extractor.extract(itv, [], anchor=itv.center()) for itv in utr5_itvs])
    cds_seq = ''.join(
        [seq_extractor.extract(itv, [], anchor=itv.center()) for itv in cds_itvs])
    # construct utr3 seq depending on if there is a variant to be inserted
    if variant:
        utr3_seq = ''.join(
            [seq_extractor.extract(itv, [variant], anchor=itv.center()) for itv in utr3_itvs]
        )
    else:
        utr3_seq = ''.join(
            [seq_extractor.extract(itv, [], anchor=itv.center()) for itv in utr3_itvs]
        )
    if strand == '+':
        return {
            'utr5': utr5_seq,
            'cds': cds_seq,
            'utr3': utr3_seq,
            # 'strand': strand
        }
    elif strand == '-':
        return {
            'utr5': reverse_compliment(utr5_seq),
            'cds': reverse_compliment(cds_seq),
            'utr3': reverse_compliment(utr3_seq),
            # 'strand': reverse_compliment(strand)
        }

def _construct_6d_track(utr5_seq, cds_seq, utr3_seq, ss_idx):
    # encode the actual sequence
    seq = utr5_seq + cds_seq + utr3_seq
    if len(seq) > MAXLEN:
        raise NotImplementedError('Transcript is too long')
    seq_onehot = kipoiseq.transforms.functional.one_hot_dna(seq).astype(np.float32)
    # cds encoding
    assert int(len(cds_seq)) % 3 == 0
    coding = np.append(
        np.zeros(len(utr5_seq)),
        np.tile([1, 0, 0], int(len(cds_seq) / 3))
    )
    # SS encoding
    ss_encoding = np.zeros(len(seq))
    ss_encoding[ss_idx] = 1
    
    # create full track
    batch = np.zeros((1, MAXLEN, 6))
    batch[0, 0:len(seq), 0:4] = seq_onehot
    batch[0, 0:len(coding), 4] = coding
    batch[0, 0:len(ss_encoding), 5] = ss_encoding

    return batch


def _get_ss_positions(transcript_df):
    # a known limitation is that this f'n doesn't take into account indels altering
    # the SS locations (since all variants are in the 3' UTR,
    # it would likely only impact the location of the final SS -- i.e., transcript end)
    # but the predictions seem like they work well enough
    ss_last_idx = 0
    ss_idx = list()
    max_exon = transcript_df['exon_number'].max()
    for exon_number in range(1, max_exon+1):
        start = transcript_df.query('feature == "exon" & exon_number == @exon_number').start.values[0]
        end = transcript_df.query('feature == "exon" & exon_number == @exon_number').end.values[0]
        pos = ss_last_idx + (end - start - 1)
        ss_last_idx = pos
        ss_idx.append(pos)
    return ss_idx

def construct_batch(transcript_df, variant=None):
    # create the track for saluki
    seq_dict = _construct_sequences(transcript_df, variant)
    splice_sites = _get_ss_positions(transcript_df)
    batch = _construct_6d_track(seq_dict['utr5'], seq_dict['cds'], seq_dict['utr3'], splice_sites)
    return batch

def main():
    args = _parse_args()
    data_dir = args.train_gru_dir

    # read UTR3 dataframe
    utr3_df = pd.read_csv(UTR3_VARIANTS_FILE, sep='\t')
    # read GTF annotations
    gtf_data = read_gtf(GTF_FILE)
    
    # predictions
    variants = list()
    refs = list()
    alts = list()
    scores = list()
    models = list()
    pred_cache = shelve.open('output/saluki_predictions_20230221.shelve')
    for idx, row in utr3_df.iterrows():
        # create variant object
        chrom, pos, ref, alt = row['variant'].split(':')
        variant = kipoiseq.dataclasses.Variant(
            chrom=chrom, pos=pos, ref=ref, alt=alt
        )
        # subset annotations to just that transcript
        transcript_id = row['clinvar_transcript_id']
        transcript_data = gtf_data.query('transcript_id == @transcript_id')

        # construct the 6D track
        ref_batch = construct_batch(transcript_data)
        alt_batch = construct_batch(transcript_data, variant)

        # generate a prediction for each model
        params_file = os.path.join(data_dir, 'params.json')
        with open(params_file) as params_open:
            params = json.load(params_open)
        params_model = params['model']
        params_train = params['train']
        for x in range(0, 10):
            for y in range(0, 5):
                # init model
                model_n = f'f{x}_c{y}'
                model_file = os.path.join(data_dir, f'{model_n}/train/model0_best.h5')
                seqnn_model = rnann.RnaNN(params_model)
                seqnn_model.restore(model_file)

                # predict 
                ref_pred = seqnn_model.predict(ref_batch)[0][0]
                alt_pred = seqnn_model.predict(alt_batch)[0][0]
                pred_diff = alt_pred - ref_pred

                # populate lists for df
                variants.append(row['variant'])
                refs.append(ref_pred)
                alts.append(alt_pred)
                scores.append(pred_diff)
                models.append(model_n)
                # also write to shelve in case script exits
                pred_cache[row['variant']] = {
                    'ref': ref_pred,
                    'alt': alt_pred,
                    'score': pred_diff,
                    'model': model_n
                }
    # create final df
    df = pd.DataFrame(data={
        'variant': variants, 'ref_pred': refs, 'alt_pred': alts,
        'score': scores, 'model': models
    })
    df.to_csv('output/saluki_predictions_20230221.tsv', sep='\t', index=False)
    pred_cache.close()

def _parse_args():
    parser = argparse.ArgumentParser(
        description='Generates predictions from each version of Saluki.')
    parser.add_argument(
        '--train_gru_dir',
        help='Path to the train_gru/ directory that has each cross-fold trained model.'
    )
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main()
