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
import argparse

from pathlib import Path
from tqdm import tqdm
from typing import List, Set, Dict, Union, Tuple, Optional


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# --------------------------------------------------
# WARNING: please don't change this to another seed, 
# required for reproducing the corpus when we have 
# multiple pairs matching for a given sentence
# --------------------------------------------------
random.seed(0)


def add_logging_handlers(logger, dir_name, fname):
    log_file = os.path.join(dir_name, fname)
    fh = logging.FileHandler(log_file)
    fh.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', '%m/%d/%Y %H:%M:%S'
    ))
    logger.addHandler(fh)


class Triples:
    
    def __init__(
            self, 
            fname: Union[str, Path], 
            cui2sty_fname: Union[str, Path] = None
        ):
        self.read_triples(fname)
        self.set_rel_to_pairs()
        self.set_pair_to_rels()
        
        if cui2sty_fname:
            self.set_pair_to_type(cui2sty_fname)
        
        self.counts()
    
    def read_triples(self, fname):
        self.triples: List[str, str, str] = list()
        self.entities: Set[str] = set()
        self.relations: Set[str] = set()
        
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
    
    def set_rel_to_pairs(self):
        self.rel2pairs = collections.defaultdict(list)
        for h, r, t in tqdm(self.triples, desc=' --- Intializing relation -> pairs map ...'):
            self.rel2pairs[r].append((h, t))
    
    def set_pair_to_rels(self):
        self.pair2rels = collections.defaultdict(list)
        for h, r, t in tqdm(self.triples, desc=' --- Intializing pair -> relations map ...'):
            self.pair2rels[(h, t)].append(r)
    
    def set_pair_to_type(self, cui2sty_fname):
        with open(cui2sty_fname) as rf:
            cui2sty = json.load(rf)
        self.pair2type = dict()
        for h, r, t in tqdm(self.triples, desc=' --- Intializing pair -> type map ...'):
            self.pair2type[(h, t)] = (cui2sty[h], cui2sty[t])
    
    def counts(self):
        self.ent_counts = collections.Counter()
        self.rel_counts = collections.Counter()
        self.pair_counts = collections.Counter()
        for h, r, t in tqdm(self.triples, desc=' --- Counting stats ...'):
            self.ent_counts.update([h, t])
            self.pair_counts.update([(h, t)])
            self.rel_counts.update([r])
    
    def query_one_hop_neighbors_by_entity(self, ent, h_or_t):
        sub_df = self.df[self.df[h_or_t] == ent]
        col = 't' if h_or_t == 'h' else 'h'
        neighbors = list(zip(sub_df[col].tolist(), sub_df['r'].tolist()))
        return neighbors
    
    def query_relational_context(self, h, r, t):
        relational_context = list()
        _, rels1 = zip(*self.query_one_hop_neighbors_by_entity(h, 'h'))
        _, rels2 = zip(*self.query_one_hop_neighbors_by_entity(t, 't'))
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
            'h_neighbors': self.query_one_hop_neighbors_by_entity(h, 'h') if with_context else None,
            't_neighbors': self.query_one_hop_neighbors_by_entity(t, 't') if with_context else None,
            'r_context': self.query_relational_context(h, r, t) if with_context else None
        }
    
    def __len__(self):
        return len(self.triples)


class BioDSRECorpus:
    
    def __init__(
            self,
            medline_entities_linked_fname: Union[str, Path],
            snomed_triples_dir: Union[str, Path],
            split: Optional[str] = None
        ):
        self.medline_entities_linked_fname = medline_entities_linked_fname
        self.snomed_triples_dir = snomed_triples_dir
        
        # map from split to its triples
        self.triples: Dict[str, List[str, str, str]] = dict()
        # entities across all splits
        self.entities = set()
        
        # identify inductive or transductive split
        self.split = (split + '-') if split == 'ind' else ''
        
        # type information is required to prune the negative pairs
        cui2sty_fname = Path(self.snomed_triples_dir) / 'cui2sty.json'
        with open(cui2sty_fname) as rf:
            self.cui2sty = json.load(rf)
        
        for data_split in ['train', 'dev', 'test']:
            fname = Path(self.snomed_triples_dir) / f'{self.split}{data_split}.tsv'
            self.triples[data_split] = Triples(fname, cui2sty_fname)
            self.entities.update(self.triples[data_split].entities)
        
        base_dir = os.path.split(medline_entities_linked_fname)[0]
        self.pos_fname = lambda split: os.path.join(base_dir, f'{self.split}{split}_pos_idxs_linked.tsv')
        self.neg_fname = os.path.join(base_dir, f'{self.split}neg_idxs_linked.tsv')
        self.base_dir = base_dir
    
    def iter_entities_linked_file(self):
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
        """Read positive pairs from triples and search for them in size-2
        permutations of entities linked file, with mentions field. These
        pairs are created using the CUIs and matched against SNOMED-CT
        pairs' CUIs.
        
        """
        # list of positive instances containing the fact pair with CUIs
        # e.g. pos_idxs['train'] = [(128, ('C0205253', 'C0014653')), (.., ..), ...]
        pos_idxs: Dict[str, List[Tuple[int, Tuple[str, str]]]] = {
            'train': list(), 'dev': list(), 'test': list()
        }
        
        # list of all negatives regardless of the split; we call a pair
        # negative if is 
        neg_idxs: List[Tuple[int, Tuple[str, str]]] = list()
        
        train_pairs = set(self.triples['train'].pair2rels.keys())
        train_pairs_inv = {(t, h) for h, t in train_pairs}
        train_pairs_types = set(self.triples['train'].pair2type.values())
        
        dev_pairs = set(self.triples['dev'].pair2rels.keys())
        dev_pairs_inv = {(t, h) for h, t in dev_pairs}
        dev_pairs_types = set(self.triples['dev'].pair2type.values())
        
        test_pairs = set(self.triples['test'].pair2rels.keys())
        test_pairs_inv = {(t, h) for h, t in test_pairs}
        test_pairs_types = set(self.triples['test'].pair2type.values())
        
        pairs = train_pairs | dev_pairs | test_pairs
        heads, tails = zip(*pairs)
        heads, tails = set(heads), set(tails)
        
        # combine all inverse pairs and types pairs
        inv = train_pairs_inv | dev_pairs_inv | test_pairs_inv
        types = train_pairs_types | dev_pairs_types | test_pairs_types
        
        heads_list, tails_list = list(heads), list(tails)
        
        # similar to usual corruption mechanism in KGC, we remove head
        # or tail entity with probability 0.5 and sample from all entities;
        # depending on the number of entities we sample for generating
        # negatives, the positive to negative ratio will change.
        # Too less negatives will result in lesser (more constrained)
        # negative examples, i.e., for the NA relation type. These are
        # raw candidate negative pairs but not the final set, as we
        # further prune based on the positive semantic type pairs and
        # check if the resulting heads appeared in any of the heads
        # across all the splits and same for the tails.
        neg_pairs = set()
        n_samples = 50
        
        for h, t in tqdm(pairs, desc='Creating candidate negative triples from positives ...'):
            # choose left or right side to corrupt
            h_or_t = random.choice([0, 1])
            if h_or_t:
                neg_tails = random.choices(tails_list, k=n_samples)
                neg_pairs.update({(h, t_neg) for t_neg in neg_tails})
            else:
                neg_heads = random.choices(heads_list, k=n_samples)
                neg_pairs.update({(h_neg, t) for h_neg in neg_heads})
        
        neg_pairs = neg_pairs - pairs - inv
        
        ntr, nval, nte, nneg = 0, 0, 0, 0
        
        for idx, jsonl in self.iter_entities_linked_file():
            
            if idx % 1000000 == 0 and idx > 0:
                ntr += len(pos_idxs['train'])
                nval += len(pos_idxs['dev'])
                nte += len(pos_idxs['test'])
                nneg += len(neg_idxs)
                    
                logger.info(f'[Progress @ {idx}] -- # train {ntr} / # dev {nval} / # test {nte} / # NA {nneg}')
                
                for split in ['train', 'dev', 'test']:
                    with open(self.pos_fname(split), 'a') as wf:
                        save_items(pos_idxs[split], wf)
                
                with open(self.neg_fname, 'a') as wf:
                    save_items(neg_idxs, wf)
            
            # Check if any entity is present more than once, drop this sentence
            # akin to: https://github.com/suamin/umls-medline-distant-re/blob/master/data_utils/link_entities.py#L45
            cui2count = collections.Counter([item['id'] for item in jsonl['mentions']])
            cuis = {cui for cui, count in cui2count.items() if count == 1}
            if not cuis:
                continue
            
            # NB. this intersection prunes out entities which are not covered in any of the splits
            snomed_cuis = cuis.intersection(self.entities)
            if not snomed_cuis:
                continue
            
            # permutations of size for concepts in a sentence that has SNOMED CUIs
            matching_snomed_permutations = set(itertools.permutations(snomed_cuis, 2))
            
            # first remove inverses, we are not modeling them
            matching_snomed_permutations = matching_snomed_permutations - inv
            
            # consider only the cases with at least two SNOMED concepts appeared so a possible relation can be searched for
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
                
                # we prune out pairs which do not respect the type constraint, i.e.,
                # the negative pair's type (TYPE_HEAD, TYPE_TAIL) must appear in
                # pairs' types from fact triples. Next, we check if such entities
                # appeared as head / tail **somewhere** across all the splits of facts.
                # this is regarded as argument-role constraint. This filters out
                # many of easy NA samples, which model can learn by simple heuristics.
                matching_types_based = {
                    (h, t) for h, t in matching_snomed_permutations 
                    if (self.cui2sty[h], self.cui2sty[t]) in types 
                    and (h in heads) and (t in tails)
                }
                
                if not matching_types_based:
                    continue
                
                # here, we consider the resultant pairs and see if they overlap
                # with our candidates generated by simple head / tail entity replacements
                candid_neg_pairs = matching_types_based.intersection(neg_pairs)
                
                if not candid_neg_pairs:
                    continue
                
                # we might have multiple matches, so we only consider one pair to
                # associate with this sentence as negative to avoid duplicate sents in NA
                pair = random.choice(list(candid_neg_pairs))
                neg_idxs.append((idx, pair))
            
            else:
                if pairs_in_test:
                    pair = random.choice(list(pairs_in_test))
                    pos_idxs['test'].append((idx, pair))
                elif pairs_in_dev:
                    pair = random.choice(list(pairs_in_dev))
                    pos_idxs['dev'].append((idx, pair))
                elif pairs_in_train:
                    pair = random.choice(list(pairs_in_train))
                    pos_idxs['train'].append((idx, pair))
        
        return ntr, nval, nte, nneg
    
    def process_jsonl_with_pair(self, jsonl: str, pair: Tuple[str, str]):
        h, t = pair
        # look for multiple appearence of head / tail entities, discard such ambigious examples
        mention_ids = [item['id'] for item in jsonl['mentions']]
        counts = collections.Counter(mention_ids)
        # make sure head and tail are present
        assert h in counts
        assert t in counts
        head_mention, tail_mention = None, None
        # since multiple mentions can cause ambiguity, we only pick instances of single h/t mention
        if counts[h] == 1 and counts[t] == 1:
            for item in jsonl['mentions']:
                if item['id'] == h:
                    head_mention = item
                if item['id'] == t:
                    tail_mention = item
            return head_mention, tail_mention
        else:
            return None
    
    def create_corpus(self, sample=1.0, neg_prop=0.7, dataset='med_distant19', size='L'):
        """Once positive and negative pairs have been read, we now create the corpora in
        OpenNRE format with different sizes and proportion of negative samples.
        
        """
        assert 0 < sample <= 1.0
        assert 0 < neg_prop <= 1.0
        neg_idx2pair = read_idx_file(self.neg_fname)
        counts = dict()
        
        for split in ['train', 'dev', 'test']:
            
            logger.info(f'Creating corpus for split {split} ...')
            
            pos_idx2pair = read_idx_file(self.pos_fname(split))
            pos_idxs = list(pos_idx2pair.keys())
            random.shuffle(pos_idxs)
            
            neg_idxs = list(neg_idx2pair.keys())
            random.shuffle(neg_idxs)
            
            # subsample the positive proportion to len(pos) * sample
            if sample < 1.0:
                pos_idxs_split = set(random.sample(pos_idxs, int(len(pos_idxs) * sample)))
            else:
                pos_idxs_split = set(pos_idxs)
            
            n_pos = len(pos_idxs_split)
            n_total = int(n_pos / (1 - neg_prop))
            n_neg = n_total - n_pos
            neg_idxs_split = set(neg_idxs[:n_neg])
            
            triples = self.triples[split]
            
            # go through positive examples
            pair2rel = dict()
            for pos_idx in pos_idxs_split:
                pair = pos_idx2pair[pos_idx]
                rels = triples.pair2rels[pair]
                # when labeled with multiple relations, consider a random one
                if len(rels) > 1:
                    rel = random.choice(rels)
                else:
                    rel = rels[0]
                pair2rel[pair] = rel
            
            corpus = list()
            n_pos, n_neg = 0, 0 # fact instances count and NA instances count
            
            for idx, jsonl in self.iter_entities_linked_file():
                if idx in pos_idxs_split:
                    pair = pos_idx2pair[idx]
                    ret = self.process_jsonl_with_pair(jsonl, pair)
                    if ret is None:
                        continue
                    head_mention, tail_mention = ret
                    rel = pair2rel[pair]
                    corpus.append({
                        'text': jsonl['text'],
                        'h': head_mention,
                        't': tail_mention,
                        'relation': rel
                    })
                    n_pos += 1
                if idx in neg_idxs_split:
                    pair = neg_idx2pair[idx]
                    ret = self.process_jsonl_with_pair(jsonl, pair)
                    if ret is None:
                        continue
                    head_mention, tail_mention = ret
                    corpus.append({
                        'text': jsonl['text'],
                        'h': head_mention,
                        't': tail_mention,
                        'relation': 'NA'
                    })
                    n_neg += 1
            
            logger.info(f'Collected {len(corpus)} lines with pos: {n_pos} and neg: {n_neg}')
            
            output_fname = os.path.join(self.base_dir, f'{self.split}{dataset}-{size}_{split}.txt')
            logger.info(f'Saving the collected corpus to output file {output_fname} ...')
            
            random.shuffle(corpus)
            
            with open(output_fname, 'w', encoding='utf-8', errors='ignore') as wf:
                for line in corpus:
                    wf.write(json.dumps(line) + '\n')
            
            # remove these negative examples of split from the rest
            for idx in neg_idxs_split:
                del neg_idx2pair[idx]
            
            counts[split] = {'npos': n_pos, 'nneg': n_neg}
        
        return counts


def save_items(items, wf):
    n = len(items)
    for _ in range(n):
        idx, pair = items.pop()
        wf.write('\t'.join([str(idx), ','.join(pair)]) + '\n')


def read_idx_file(fname):
    idx2pair = dict()
    with open(fname) as rf:
        for line in tqdm(rf, desc=f'Reading idxs file {fname}'):
            line = line.strip()
            if not line:
                continue
            idx, pair = line.split('\t')
            pair = tuple(pair.split(','))
            idx2pair[int(idx)] = pair
    return idx2pair


def main(args):
    corpus = BioDSRECorpus(
        args.medline_entities_linked_fname,
        args.triples_dir,
        'ind' if args.split == 'ind' else None
    )
    # see if pos and neg linked files have been created before, simplz check for train
    check = all(list(
        map(os.path.exists, [corpus.pos_fname(split) for split in ['train', 'dev', 'test']] + [corpus.neg_fname,])
    ))
    add_logging_handlers(logger, corpus.base_dir, f'corpus_{args.split}_{args.size}.log')
    if not check:
        # this will take time, go grab 2 cups of coffee :)
        ntr, nval, nte, nneg = corpus.search_pos_and_neg_instances()
        logger.info(f'Positive and negative instances statistics ...')
        logger.info(f'--- train instances (+ve) = {ntr}')
        logger.info(f'--- dev instances (+ve) = {nval}')
        logger.info(f'--- test instances (+ve) = {nte}')
        logger.info(f'--- negative instances (NA) = {nneg}')
    logger.info(f'Creating `{args.size}` corpus ...')
    if args.size != 'O':
        if args.size == 'L':
            counts = corpus.create_corpus(sample=1.0, neg_prop=0.7, size='L')
        elif args.size == 'M':
            counts = corpus.create_corpus(sample=0.1, neg_prop=0.7, size='M')
        elif args.size == 'S':
            counts = corpus.create_corpus(sample=0.05, neg_prop=0.7, size='S')
    else:
        counts = corpus.create_corpus(
            sample=args.sample, neg_prop=args.neg_prop, dataset='med_distant19_x',size=args.size
        )
    logger.info(f'Final corpus statistics ...')
    for split in counts:
        logger.info(f'{split} +ve and NA instances = {counts[split]}')


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--medline_entities_linked_fname", action="store", required=True, type=str,
        help="Path to *.jsonl concepts linked file."
    )
    parser.add_argument(
        "--triples_dir", action="store", required=True, type=str,
        help="Path to triples files *.tsv."
    )
    parser.add_argument(
        "--split", action="store", type=str, default="trans", choices=["ind", "trans"],
        help="Which triples split to consider, inductive (`ind`) or transductive (`trans`)."
    )
    parser.add_argument(
        "--size", action="store", type=str, default="S", choices=["S", "M", "L", "O"],
        help="What size to create Small (`S`), Medium (`M`), Large (`L`) or Other (`O`)."
        "For `O` use the --sample and --neg_prop flags."
    )
    parser.add_argument(
        "--sample", action="store", type=float, default=1.0,
        help="Sub-sample the triples for a given split to obtain different sized corpora."
    )
    parser.add_argument(
        "--neg_prop", action="store", type=float, default=0.7,
        help="The NA rate for negative instances."
    )
    
    args = parser.parse_args()
    
    import pprint
    pprint.pprint(vars(args))
    
    main(args)
