# -*- coding: utf-8 -*-

import json
import collections
import logging
import numpy as np
import gensim.models.word2vec as w2v
import os
import argparse

from pathlib import Path


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class CorpusIterator:
    
    def __init__(self, fname):
        self.fname = fname
    
    def __iter__(self):
        with open(self.fname, encoding='utf-8', errors='ignore') as rf:
            for line in rf:
                line = line.strip()
                if not line:
                    continue
                jsonl = json.loads(line)
                sentence_tokens = jsonl['text'].split()
                yield sentence_tokens


def train_and_dump_word2vec(
        medline_entities_linked_fname,
        output_dir,
        n_workers=4,
        n_iter=3
    ):
    # fix embed dim = 50 and max vocab size to 50k
    model = w2v.Word2Vec(size=50, workers=n_workers, iter=n_iter, max_final_vocab=50000)
    sentences = CorpusIterator(medline_entities_linked_fname)
    
    logger.info(f'Building word2vec vocab on {medline_entities_linked_fname}...')
    model.build_vocab(sentences)
    
    logger.info('Training ...')
    model.train(sentences, total_examples=model.corpus_count, epochs=model.iter)
    
    os.makedirs(output_dir, exist_ok=True)
    logger.info('Saving word2vec model ...')
    model.save(os.path.join(output_dir, 'word2vec.pubmed2019.50d.gz'))
    
    wv = model.wv
    del model # free up memory
    
    word2id = {"<PAD>": 0, "<UNK>": 1}
    mat = np.zeros((len(wv.vocab.keys()) + 2, 50))
    # initialize UNK embedding with random normal
    mat[1] = np.random.randn(50)
    
    for word in sorted(wv.vocab.keys()):
        vocab_item = wv.vocab[word]
        vector = wv.vectors[vocab_item.index]
        mat[len(word2id)] = vector
        word2id[word] = len(word2id)
    
    mat_fname = Path(output_dir) / f'word2vec.pubmed2019.50d_mat.npy'
    map_fname = Path(output_dir) / f'word2vec.pubmed2019.50d_word2id.json'
    
    logger.info(f'Saving word2id at {map_fname} and numpy matrix at {mat_fname} ...')
    
    np.save(mat_fname, mat)
    with open(map_fname, 'w', encoding='utf-8', errors='ignore') as wf:
        json.dump(word2id, wf)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--medline_entities_linked_fname", action="store", required=True, type=str,
        help="Path to *.jsonl concepts linked file."
    )
    parser.add_argument(
        "--output_dir", action="store", required=True, type=str,
        help="Path to output directory where the word2id and numpy matrix will be saved."
    )
    
    args = parser.parse_args()
    
    import pprint
    pprint.pprint(vars(args))
    
    train_and_dump_word2vec(
        args.medline_entities_linked_fname,
        args.output_dir
    )
