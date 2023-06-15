"""
To be run with the `utr_curation_manuscript` conda environment activated.
Run from the repo root directory.
"""
import argparse
import os
import shelve

import kipoiseq

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd

from .utils import FastaStringExtractor, reverse_compliment, variant_generator, one_hot_encode


FASTA_FILE = 'data/hg38.fa'
SEQUENCE_LENGTH = 393_216


# useful utilities from the Enformer example
# https://colab.research.google.com/github/deepmind/deepmind_research/blob/master/enformer/enformer-usage.ipynb
class Enformer:
    
  def __init__(self, tfhub_url):
    self._model = hub.load(tfhub_url).model

  def predict_on_batch(self, inputs):
    predictions = self._model.predict_on_batch(inputs)
    return {k: v.numpy() for k, v in predictions.items()}

  @tf.function
  def contribution_input_grad(self, input_sequence,
                              target_mask, output_head='human'):
    input_sequence = input_sequence[tf.newaxis]

    target_mask_mass = tf.reduce_sum(target_mask)
    with tf.GradientTape() as tape:
      tape.watch(input_sequence)
      prediction = tf.reduce_sum(
          target_mask[tf.newaxis] *
          self._model.predict_on_batch(input_sequence)[output_head]) / target_mask_mass

    input_grad = tape.gradient(prediction, input_sequence) * input_sequence
    input_grad = tf.squeeze(input_grad, axis=0)
    return tf.reduce_sum(input_grad, axis=-1)


def variant_centered_sequences(fasta_file, variant_list, sequence_length, chr_prefix=''):
  seq_extractor = kipoiseq.extractors.VariantSeqExtractor(
    reference_sequence=FastaStringExtractor(fasta_file))

  variants = [variant_generator(variant) for variant in variant_list]

  for variant in variants:
    interval = kipoiseq.Interval(chr_prefix + variant.chrom,
                        variant.pos, variant.pos)
    interval = interval.resize(sequence_length)
    center = interval.center() - interval.start

    reference = seq_extractor.extract(interval, [], anchor=center)
    alternate = seq_extractor.extract(interval, [variant], anchor=center)

    yield {'inputs': {'ref': reference,
                      'alt': alternate},
           'metadata': {'chrom': chr_prefix + variant.chrom,
                        'pos': variant.pos,
                        'id': variant.id,
                        'ref': variant.ref,
                        'alt': variant.alt}}


def run_enformer_and_avg_strands(model, seq):
    pred_fwd = model.predict_on_batch(one_hot_encode(seq)[np.newaxis])['human'][0]
    pred_rev = model.predict_on_batch(one_hot_encode(reverse_compliment(seq))[np.newaxis])['human'][0]
    # average them 
    pred = (pred_fwd + pred_rev[::-1]) / 2
    return pred


def main():
    args = _parse_args()
    overwrite = args.overwrite
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_visible_device)

    # read in variants
    utr3_df = pd.read_csv('data/utr3_plp_benchmark.tsv', sep='\t')
    utr5_df = pd.read_csv('data/utr5_plp_benchmark.tsv', sep='\t')
    variants = list()
    variants.extend(utr3_df.variant.drop_duplicates().to_list())
    variants.extend(utr5_df.variant.drop_duplicates().to_list())

    # set up Enformer
    enformer = Enformer('https://tfhub.dev/deepmind/enformer/1')

    pred_cache = shelve.open('output/enformer_predictions_20230216.shelve')

    i = 0 
    for variant in variant_centered_sequences(FASTA_FILE, variants, SEQUENCE_LENGTH):
        variant_id = variant['metadata']['id']
        if (pred_cache.get(variant_id) is None) or (pred_cache.get(variant_id) is not None and overwrite):
            pred_ref = run_enformer_and_avg_strands(enformer, variant['inputs']['ref'])
            pred_alt = run_enformer_and_avg_strands(enformer, variant['inputs']['alt'])

            pred_cache[variant_id] = {
              'ref': pred_ref,
              'alt': pred_alt
            }
        else:
            pass
        
        i += 1
        if i % 25 == 0:
            print(f'Done processing {i} variants.')
    
    pred_cache.close()


def _parse_args():
    parser = argparse.ArgumentParser(
        description='Runs Enformer to generate predictions. Writes out .pickle files.')
    parser.add_argument(
        '--overwrite',
        help='Overwrites predictions in the output file',
        action='store_true', default=False)
    parser.add_argument(
        '--cuda_visible_device', '-cuda', help='Sets the CUDA_VISIBLE_DEVICE variable',
        default='0'
    )


    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()