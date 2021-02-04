# coding=utf-8
# Adapted from Tensor2Tensor under Apache-2.0 license
# Source: https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/bleu_hook.py
# Copyright 2020 The Tensor2Tensor Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import collections
import math
import numpy as np


class BLEU:
    def __init__(self,
                 max_order: int = 4,
                 use_bp: bool = True):
        """
        max_order: Maximum n-gram order to use when computing BLEU score.
        use_bp: boolean, whether to apply brevity penalty.
        """
        self.max_order = max_order
        self.use_bp = use_bp
        self.reset()

    def reset(self):
        self.reference_length = 0
        self.translation_length = 0

        self.matches_by_order = [0] * self.max_order
        self.possible_matches_by_order = [0] * self.max_order

    def update_state(self, references, translations):
        """Computes BLEU score of translated segments against one or more references.

        Args:
          references: References for translation, should be tokenized into a list of tokens.
          translations: list of translations to score. Each translation
              should be tokenized into a list of tokens.

        Returns:
          BLEU score.
        """
        self.reference_length += len(references)
        self.translation_length += len(translations)
        ref_ngram_counts = _get_ngrams(references, self.max_order)
        translation_ngram_counts = _get_ngrams(translations, self.max_order)

        overlap = dict((ngram,
                        min(count, translation_ngram_counts[ngram]))
                       for ngram, count in ref_ngram_counts.items())

        for ngram in overlap:
            self.matches_by_order[len(ngram) - 1] += overlap[ngram]
        for ngram in translation_ngram_counts:
            self.possible_matches_by_order[len(ngram) - 1] += translation_ngram_counts[ngram]

    def result(self):
        """
        Return the final BLEU value.
        :return: np.float32
        """
        bp = 1.0
        geo_mean = 0

        precisions = [0] * self.max_order
        smooth = 1.0
        for i in range(0, self.max_order):
            if self.possible_matches_by_order[i] > 0:
                precisions[i] = self.matches_by_order[i] / self.possible_matches_by_order[i]
                if self.matches_by_order[i] > 0:
                    precisions[i] = self.matches_by_order[i] / self.possible_matches_by_order[i]
                else:
                    smooth *= 2
                    precisions[i] = 1.0 / (smooth * self.possible_matches_by_order[i])
            else:
                precisions[i] = 0.0

        if max(precisions) > 0:
            p_log_sum = sum(math.log(p) for p in precisions if p)
            geo_mean = math.exp(p_log_sum / self.max_order)

        if self.use_bp:
            if not self.reference_length:
                bp = 1.0
            else:
                ratio = self.translation_length / self.reference_length
                if ratio <= 0.0:
                    bp = 0.0
                elif ratio >= 1.0:
                    bp = 1.0
                else:
                    bp = math.exp(1 - 1. / ratio)
        bleu = geo_mean * bp
        return np.float32(bleu)


def _get_ngrams(segment, max_order):
    """Extracts all n-grams up to a given maximum order from an input segment.

    Args:
      segment: text segment from which n-grams will be extracted.
      max_order: maximum length in tokens of the n-grams returned by this
          methods.

    Returns:
      The Counter containing all n-grams up to max_order in segment
      with a count of how many times each n-gram occurred.
    """
    ngram_counts = collections.Counter()
    for order in range(1, max_order + 1):
        for i in range(0, len(segment) - order + 1):
            ngram = tuple(segment[i:i + order])
            ngram_counts[ngram] += 1
    return ngram_counts


def compute_bleu(reference_corpus,
                 translation_corpus,
                 max_order=4,
                 use_bp=True):
    """Computes BLEU score of translated segments against one or more references.

    Args:
      reference_corpus: list of references for each translation. Each
          reference should be tokenized into a list of tokens.
      translation_corpus: list of translations to score. Each translation
          should be tokenized into a list of tokens.
      max_order: Maximum n-gram order to use when computing BLEU score.
      use_bp: boolean, whether to apply brevity penalty.

    Returns:
      BLEU score.
    """
    reference_length = 0
    translation_length = 0
    bp = 1.0
    geo_mean = 0

    matches_by_order = [0] * max_order
    possible_matches_by_order = [0] * max_order
    precisions = []

    for (references, translations) in zip(reference_corpus, translation_corpus):
        reference_length += len(references)
        translation_length += len(translations)
        ref_ngram_counts = _get_ngrams(references, max_order)
        translation_ngram_counts = _get_ngrams(translations, max_order)

        overlap = dict((ngram,
                        min(count, translation_ngram_counts[ngram]))
                       for ngram, count in ref_ngram_counts.items())

        for ngram in overlap:
            matches_by_order[len(ngram) - 1] += overlap[ngram]
        for ngram in translation_ngram_counts:
            possible_matches_by_order[len(ngram) - 1] += translation_ngram_counts[ngram]
    precisions = [0] * max_order
    smooth = 1.0
    for i in range(0, max_order):
        if possible_matches_by_order[i] > 0:
            precisions[i] = matches_by_order[i] / possible_matches_by_order[i]
            if matches_by_order[i] > 0:
                precisions[i] = matches_by_order[i] / possible_matches_by_order[i]
            else:
                smooth *= 2
                precisions[i] = 1.0 / (smooth * possible_matches_by_order[i])
        else:
            precisions[i] = 0.0

    if max(precisions) > 0:
        p_log_sum = sum(math.log(p) for p in precisions if p)
        geo_mean = math.exp(p_log_sum / max_order)

    if use_bp:
        if not reference_length:
            bp = 1.0
        else:
            ratio = translation_length / reference_length
            if ratio <= 0.0:
                bp = 0.0
            elif ratio >= 1.0:
                bp = 1.0
            else:
                bp = math.exp(1 - 1. / ratio)
    bleu = geo_mean * bp
    return np.float32(bleu)
