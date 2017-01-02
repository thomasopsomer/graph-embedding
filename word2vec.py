# -*- coding: utf-8 -*-
# @Author: ThomasO
from gensim.models.word2vec import Word2Vec, Vocab
from collections import defaultdict
from six import iteritems, itervalues
import logging
from math import sqrt


logger = logging.getLogger("gensim")


class Word2Vec(Word2Vec):
    """
    """
    def build_vocab(self, sentences, keep_raw_vocab=False, trim_rule=None,
                    progress_per=10000, update=False):
        """
        Build vocabulary from a sequence of sentences
            (can be a once-only generator stream).
        Each sentence must be a list of unicode strings.

        """
        self.scan_vocab(sentences, progress_per=progress_per,
                        trim_rule=trim_rule, update=update)
        self.scale_vocab(keep_raw_vocab=keep_raw_vocab,
                         trim_rule=trim_rule, update=update)
        self.finalize_vocab(update=update)

    def scan_vocab(self, sentences, progress_per=10000, trim_rule=None,
                   update=False):
        """Do an initial scan of all words appearing in sentences."""
        logger.info("collecting all nodes and their counts")
        sentence_no = -1
        total_words = 0
        vocab = defaultdict(int)

        for sentence_no, sentence in enumerate(sentences):
            for word in sentence:
                vocab[word] += 1
        total_words += sum(itervalues(vocab))
        logger.info("collected %i word types from a corpus of %i raw words and %i sentences",
                    len(vocab), total_words, sentence_no + 1)
        self.corpus_count = sentence_no + 1
        self.raw_vocab = vocab
        self.total_words = total_words

    def scale_vocab(self, sample=None, dry_run=False,
                    keep_raw_vocab=False, trim_rule=None, update=False):
        """
        Apply vocabulary settings for `min_count` (discarding less-frequent words)
        and `sample` (controlling the downsampling of more-frequent words).

        Calling with `dry_run=True` will only simulate the provided settings and
        report the size of the retained vocabulary, effective corpus length, and
        estimated memory requirements. Results are both printed via logging and
        returned as a dict.

        Delete the raw vocabulary after the scaling is done to free up RAM,
        unless `keep_raw_vocab` is set.

        """
        sample = sample or self.sample

        logger.info("Loading a fresh vocabulary")

        # Discard words less-frequent than min_count
        if not dry_run:
            self.index2word = []
            # make stored settings match these applied settings
            self.sample = sample
            self.vocab = {}

        for word, v in iteritems(self.raw_vocab):
            if not dry_run:
                self.vocab[word] = Vocab(count=v, index=len(self.index2word))
                self.index2word.append(word)

        retain_total = self.total_words

        # Precalculate each vocabulary item's threshold for sampling
        if not sample:
            # no words downsampled
            threshold_count = retain_total
        elif sample < 1.0:
            # traditional meaning: set parameter as proportion of total
            threshold_count = sample * retain_total
        else:
            # new shorthand: sample >= 1 means downsample all words with
            # higher count than sample
            threshold_count = int(sample * (3 + sqrt(5)) / 2)

        downsample_total, downsample_unique = 0, 0
        for w in self.raw_vocab.iterkeys():
            v = self.raw_vocab[w]
            word_probability = (sqrt(v / threshold_count) + 1) * (threshold_count / v)
            if word_probability < 1.0:
                downsample_unique += 1
                downsample_total += word_probability * v
            else:
                word_probability = 1.0
                downsample_total += v
            if not dry_run:
                self.vocab[w].sample_int = int(round(word_probability * 2**32))

        if not dry_run and not keep_raw_vocab:
            logger.info("deleting the raw counts dictionary of %i items",
                        len(self.raw_vocab))
            self.raw_vocab = defaultdict(int)

        logger.info("sample=%g downsamples %i most-common words",
                    sample, downsample_unique)
        logger.info("downsampling leaves estimated %i word corpus (%.1f%% of prior %i)",
                    downsample_total, downsample_total * 100.0 / max(retain_total, 1), retain_total)

        # print extra memory estimates
        memory = self.estimate_memory(vocab_size=len(self.vocab))

        return memory

    def finalize_vocab(self, update=False):
        """
        Build tables and model weights based on final vocabulary settings.
        """
        if not self.index2word:
            self.scale_vocab()
        if self.sorted_vocab and not update:
            self.sort_vocab()
        if self.hs:
            # add info about each word's Huffman encoding
            self.create_binary_tree()
        if self.negative:
            # build the table for drawing random words (for negative sampling)
            self.make_cum_table()
        # set initial input/projection and hidden weights
        if not update:
            self.reset_weights()
        else:
            self.update_weights()

