import os
import sys
import numpy as np
import gensim
import spacy
import nltk
import torch
from nltk.corpus      import wordnet as wn
from torchtext.vocab  import load_word_vectors
from itertools        import repeat
from operator         import itemgetter

class NLP:

    def __init__(self):
        self.wordnet = wn
        self.w2v     = gensim.models.KeyedVectors.load_word2vec_format(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'text8-vector.bin'), binary=True)
        self.spacy   = spacy.load('en')
        self.resnik  = nltk.corpus.wordnet_ic.ic('ic-brown.dat')
        self.glove_dict, self.glove_arr, self.glove_size = load_word_vectors('.', 'glove.6B', 100)

    def expand(self, word):
        expansions       = []
        merged           = []
        final            = []
        wn_expansions    = np.unique(self.expand_wordnet(word))
        w2v_expansions   = np.unique(self.expand_w2v(word))
        spacy_expansions = np.unique(self.expand_spacy(word))
        glove_expansions = np.unique(self.expand_glove(word))

        for e in wn_expansions:
            expansions.append(e)

        for e in w2v_expansions:
            for f in expansions:
                if f['word'] is e['word']:
                    f['score'] += e['score']
                    merged.append(f)
                else:
                    merged.append(e)

        for e in spacy_expansions:
            for f in expansions:
                if f['word'] is e['word']:
                    f['score'] += e['score']
                    merged.append(f)
                else:
                    merged.append(e)

        for e in glove_expansions:
            for f in expansions:
                if f['word'] is e['word']:
                    f['score'] += e['score']
                    merged.append(f)
                else:
                    merged.append(e)

        score_total = 0.0

        for w in merged:
            score_total += w['score']

        average = score_total / len(merged)

        for w in merged:
            if w['score'] >= average:
                final.append(w)

        return np.unique(final)

    def compute_score(self, word, match, score=0.0):
        try:
            word   = self.wordnet.synsets(word)[0]
            match  = self.wordnet.synsets(match)[0]
            score += word.path_similarity(match)
            score += word.lch_similarity(match)
            score += word.wup_similarity(match)
            score += word.jcn_similarity(match, self.resnik)
        except:
            pass

        return score

    def expand_wordnet(self, word):
        expansions = []

        for synset in self.wordnet.synsets(word):
            for lemma in synset.lemma_names():
                score = self.compute_score(word, lemma.lower())
                expansions.append({
                    'word':  lemma.lower(),
                    'score': self.sigmoid(score)
                })

        return expansions

    def expand_w2v(self, word):
        expansions = []

        try:
            for w in self.w2v.most_similar(word):
                score = self.compute_score(word, w[0].lower(), w[1])
                expansions.append({
                    'word':  w[0].lower(),
                    'score': self.sigmoid(score)
                })
        except:
            pass

        return expansions

    def expand_spacy(self, word):
        expansions = []

        for w in self.most_similar(self.spacy.vocab[u''.join([word])]):
            score = self.compute_score(word, w.lower_)
            expansions.append({
                'word':  w.lower_,
                'score': self.sigmoid(score)
            })

        return expansions

    def expand_glove(self, word):
        expansions = []

        for w in self.closest(self.glove_dict, self.glove_arr, self.get_word(word, self.glove_dict, self.glove_arr)):
            score = self.compute_score(word, w[0].lower(), self.sigmoid(w[1]))
            expansions.append({
                'word':  w[0].lower(),
                'score': self.sigmoid(score)
            })

        return expansions

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def most_similar(self, word):
        queries = [w for w in word.vocab if w.is_lower == word.is_lower and w.prob >= -15]
        by_similarity = sorted(queries, key=lambda w: word.similarity(w), reverse=True)
        return by_similarity[:20]

    def get_word(self, word, glove_dict, glove_arr):
        return glove_arr[glove_dict[word]]

    def closest(self, glove_dict, glove_arr, d, n=10):
        all_dists = [(w, torch.dist(d, self.get_word(w, glove_dict, glove_arr))) for w in glove_dict]
        return sorted(all_dists, key=lambda t: t[1])[:n]
