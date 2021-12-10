# -*- coding: utf-8 -*-

import json
import itertools
import logging
import pickle as pkl
import pandas as pd
import collections
import pickle
import os
import random

from pathlib import Path
from tqdm import tqdm


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# WARN: please don't change this to another seed, required
# for reproducing the corpus when we have multiple pairs
# matching for a given sentence
random.seed(0)

class Triples:
    
    def __init__(self, fname):
        self.__triples(fname)
        self.__rel_to_pairs()
        self.__pair_to_rels()
        self.__counts()
    
    def __triples(self, fname):
        self.triples = list()
        self.entities = set()
        self.relations = set()
        with open(fname) as rf:
            for line in tqdm(rf, desc='Reading raw triples ...'):
                line = line.strip()
                if not line:
                    continue
                h, r, t = line.split('\t')
                self.triples.append((h, r, t))
                self.entities.update([h, t])
                self.relations.add(r)
        self.df = pd.DataFrame(self.triples, columns=['h', 'r', 't'])
    
    def __rel_to_pairs(self):
        self.rel2pairs = collections.defaultdict(list)
        for h, r, t in tqdm(self.triples, desc='Counting relations to pairs ...'):
            self.rel2pairs[r].append((h, t))
    
    def __pair_to_rels(self):
        self.pair2rels = collections.defaultdict(list)
        for h, r, t in tqdm(self.triples, desc='Counting pair to relations ...'):
            self.pair2rels[(h, t)].append(r)
    
    def __counts(self):
        self.ent_counts = collections.Counter()
        self.rel_counts = collections.Counter()
        self.pair_counts = collections.Counter()
        for h, r, t in tqdm(self.triples, desc='Counting facts stats ...'):
            self.ent_counts.update([h, t])
            self.pair_counts.update([(h, t)])
            self.rel_counts.update([r])
    
    def __query_one_hop_neighbors_by_entity(self, ent, h_or_t):
        sub_df = self.df[self.df[h_or_t] == ent]
        col = 't' if h_or_t == 'h' else 'h'
        neighbors = list(zip(sub_df[col].tolist(), sub_df['r'].tolist()))
        return neighbors
    
    def __query_relational_context(self, h, r, t):
        relational_context = list()
        _, rels1 = zip(*self.__query_one_hop_neighbors_by_entity(h, 'h'))
        _, rels2 = zip(*self.__query_one_hop_neighbors_by_entity(t, 't'))
        relational_context.extend(rels1)
        relational_context.extend(rels2)
        relational_context = collections.Counter(relational_context)
        del relational_context[r] # remove self
        return relational_context
    
    def get(self, idx, with_context=False):
        h, r, t = self.triples[idx]
        return {
            'h': h,
            'r': r,
            't': t,
            'h_neighbors': self.__query_one_hop_neighbors_by_entity(h, 'h') if with_context else None,
            't_neighbors': self.__query_one_hop_neighbors_by_entity(t, 't') if with_context else None,
            'r_context': self.__query_relational_context(h, r, t) if with_context else None
        }
    
    def __len__(self):
        return len(self.triples)


def save_items(items, wf):
    n = len(items)
    for _ in range(n):
        idx, pair = items.pop()
        wf.write('\t'.join([str(idx), ','.join(pair)]) + '\n')


class BioDSRECorpus:
    
    def __init__(
            self,
            medline_entities_linked_fname,
            snomed_triples_dir,
            split="ind"
        ):
        self.medline_entities_linked_fname = medline_entities_linked_fname
        self.snomed_triples_dir = snomed_triples_dir
        self.triples = dict()
        self.entities = set()
        self.split = split
        for _split in ['train', 'dev', 'test']:
            prefix = split + '-' if split == 'ind' else ''
            fname = Path(self.snomed_triples_dir) / f'{prefix}{_split}.tsv'
            self.triples[_split] = Triples(fname)
            self.entities.update(self.triples[_split].entities)
    
    def __iter__(self):
        with open(self.medline_entities_linked_fname) as rf:
            idx = 0
            for jsonl in tqdm(rf, 'Reading entities linked file ...'):
                jsonl = jsonl.strip()
                if not jsonl:
                    continue
                jsonl = json.loads(jsonl)
                yield idx, jsonl
                idx += 1
    
    def search_pos_and_neg_instances(self):
        """
        144796033it [51:13, 59797.06it/s]12/06/2021 18:34:38 - INFO - [Progress @ 144800000] -- 
                    train: 3850865 / dev 121132 / test 132037 || neg 107331011
        
        144794585it [50:34, 62430.07it/s]12/07/2021 12:16:20 - INFO - [Progress @ 144800000] -- 
                     train: 3850865 / dev 104916 / test 120087 || neg 107331011

        144798425it [1:28:02, 33846.23it/s]12/07/2021 23:07:50 - INFO - [Progress @ 144800000] -- 
                     train: 3281943 / dev 391647 / test 430435 || neg 107331020

        """
        base_dir = os.path.split(self.medline_entities_linked_fname)[0]
        pos_fname = lambda split: os.path.join(base_dir, f'{self.split}-{split}_pos_idxs_linked.tsv')
        neg_fname = os.path.join(base_dir, f'{self.split}-neg_idxs_linked.tsv')
        pos_idxs = {'train': list(), 'dev': list(), 'test': list()}
        neg_idxs = list()
        train_pairs = set(self.triples['train'].pair2rels.keys())
        dev_pairs = set(self.triples['dev'].pair2rels.keys())
        test_pairs = set(self.triples['test'].pair2rels.keys())
        ntr, nval, nte, nneg = 0, 0, 0, 0
        
        for idx, jsonl in iter(self): 
            cuis = {item['id'] for item in jsonl['mentions']}
            snomed_cuis = cuis.intersection(self.entities)
            
            # permutations of size for concepts in a sentence that has SNOMED CUIs
            matching_snomed_permutations = set(itertools.permutations(snomed_cuis, 2))
            
            # consider only the cases with at least two SNOMED concepts
            # so a possible relation can be searched for
            if len(matching_snomed_permutations) < 2:
                continue
            
            # check if we have the matching pairs in any of the splits
            pairs_in_train = matching_snomed_permutations.intersection(train_pairs)
            pairs_in_dev = matching_snomed_permutations.intersection(dev_pairs)
            pairs_in_test = matching_snomed_permutations.intersection(test_pairs)
            
            # unlikely, but we make sure that no pairs are seen across the split; even though
            # the triples creation process has taken care of it, this is to be on the safe-side
            try:
                assert not (pairs_in_train & pairs_in_dev)
                assert not (pairs_in_train & pairs_in_test)
                assert not (pairs_in_dev & pairs_in_test)
            except AssertionError:
                continue
            
            # gather the pairs that have been matched
            pruned_snomed_permutations = pairs_in_train | pairs_in_dev | pairs_in_test
            
            # we find no positive pairs across all the splits, this sentence
            # can be considered for not applicable (NA) relations
            if not pruned_snomed_permutations:
                neg_pair = random.choice(list(matching_snomed_permutations))
                neg_idxs.append((idx, neg_pair))
            else:
                # give test the highest priority
                if pairs_in_test:
                    pair = random.choice(list(pairs_in_test))
                    pos_idxs['test'].append((idx, pair))
                elif pairs_in_dev:
                    pair = random.choice(list(pairs_in_dev))
                    pos_idxs['dev'].append((idx, pair))
                elif pairs_in_train:
                    pair = random.choice(list(pairs_in_train))
                    pos_idxs['train'].append((idx, pair))
            
            if idx % 1000000 == 0 and idx > 0:
                ntr += len(pos_idxs['train'])
                nval += len(pos_idxs['dev'])
                nte += len(pos_idxs['test'])
                nneg += len(neg_idxs)
                
                logger.info(f'[Progress @ {idx}] -- train: {ntr} / dev {nval} / test {nte} || neg {nneg}')
                
                for split in ['train', 'dev', 'test']:
                    with open(pos_fname(split), 'a') as wf:
                        save_items(pos_idxs[split], wf)
                
                with open(neg_fname, 'a') as wf:
                    save_items(neg_idxs, wf)


medline_entities_linked_fname = '/netscratch/samin/projects/med_distant_2019/MEDLINE/medline_pubmed_2019_entity_linked.jsonl'
triples_dir = '/netscratch/samin/projects/med_distant_2019/UMLS'
subset = 'ind'

corpus = BioDSRECorpus(medline_entities_linked_fname, triples_dir, 'ind')
pos_idxs, neg_idxs = corpus.search_pos_and_neg_instances()
