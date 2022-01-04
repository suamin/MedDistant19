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
import spacy

from pathlib import Path
from tqdm import tqdm
from scispacy.umls_linking import UmlsEntityLinker
from scispacy.linking_utils import KnowledgeBase
from typing import List, Set, Dict, Union, Tuple, Optional


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


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
        logger.info(f'Top-10 frequent entities: {self.ent_counts.most_common(10)}')
        logger.info(f'Top-10 frequent arg pairs: {self.pair_counts.most_common(10)}')
        logger.info(f'Top-10 frequent relations: {self.rel_counts.most_common(10)}')
    
    def __getitem__(self):
        return self.triples[idx]
    
    def __len__(self):
        return len(self.triples)


class BioDSRECorpus:
    
    def __init__(
            self,
            medline_entities_linked_fname: Union[str, Path],
            snomed_triples_dir: Union[str, Path],
            split: Optional[str] = None,
            has_def: Optional[bool] = False
        ):
        self.medline_entities_linked_fname = medline_entities_linked_fname
        self.snomed_triples_dir = snomed_triples_dir
        
        # map from split to its triples
        self.triples: Dict[str, List[str, str, str]] = dict()
        # entities across all splits
        self.entities = set()
        
        # identify inductive or transductive split
        self.split = (split + ('_def' if has_def else '') + '-') if split == 'ind' else ''
        
        # type information is required to prune the negative pairs
        cui2sty_fname = Path(self.snomed_triples_dir) / 'cui2sty.json'
        with open(cui2sty_fname) as rf:
            self.cui2sty = json.load(rf)
        
        for data_split in ['train', 'dev', 'test']:
            fname = Path(self.snomed_triples_dir) / f'{self.split}{data_split}.tsv'
            self.triples[data_split] = Triples(fname, cui2sty_fname)
            self.entities.update(self.triples[data_split].entities)
        
        base_dir = os.path.split(medline_entities_linked_fname)[0]
        self.pos_fname = lambda split: os.path.join(
            base_dir, f'{self.split}{split}_pos_idxs_linked.tsv'
        )
        self.neg_fname = lambda split: os.path.join(
            base_dir, f'{self.split}{split}_neg_idxs_linked.tsv'
        )
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
    
    def search_pos_and_neg_instances(
            self, 
            raw_neg_sample_size: int = 500,
            corrupt_arg: bool = True,
            remove_multimentions_sents: bool = True,
            use_type_constraint: bool = True, 
            use_arg_constraint: bool = True
        ):
        """Read positive pairs from triples and search for them in size-2
        permutations of entities linked file, with mentions field. These
        pairs are created using the CUIs and matched against SNOMED-CT
        pairs' CUIs.
        
        """
        # list of positive instances containing the fact pair with CUIs
        # e.g. pos_idxs['train'] = [(128, ('C0205253', 'C0014653')), (.., ..), ...]
        pos_idxs: Dict[str, List[Tuple[int, Tuple[str, str]]]] = dict(
            train=list(), dev=list(), test=list()
        )
        # similarly, list of all negatives
        neg_idxs: Dict[str, List[Tuple[int, Tuple[str, str]]]] = dict(
            train=list(), dev=list(), test=list()
        )
        
        # ----------------------------------------
        # **TRAIN** pairs, inverse pairs and types 
        # ----------------------------------------
        train_pairs = set(self.triples['train'].pair2rels.keys())
        train_pairs_inv = {(t, h) for h, t in train_pairs}
        train_pairs_types = set(self.triples['train'].pair2type.values())
        
        # ----------------------------------------
        # **DEV** pairs, inverse pairs and types 
        # ----------------------------------------
        dev_pairs = set(self.triples['dev'].pair2rels.keys())
        dev_pairs_inv = {(t, h) for h, t in dev_pairs}
        dev_pairs_types = set(self.triples['dev'].pair2type.values())
        
        # ----------------------------------------
        # **TEST** pairs, inverse pairs and types 
        # ----------------------------------------
        test_pairs = set(self.triples['test'].pair2rels.keys())
        test_pairs_inv = {(t, h) for h, t in test_pairs}
        test_pairs_types = set(self.triples['test'].pair2type.values())
        
        #-----------------------------------------------------------------
        
        pairs = train_pairs | dev_pairs | test_pairs
        heads, tails = zip(*pairs)
        heads, tails = set(heads), set(tails)
        heads_list, tails_list = sorted(list(heads)), sorted(list(tails))
        
        # combine all inverse pairs and types pairs
        inv = train_pairs_inv | dev_pairs_inv | test_pairs_inv
        types = train_pairs_types | dev_pairs_types | test_pairs_types
        
        entities = sorted(list(self.entities))
        
        ################################################################
        #
        # 1. COLLECT RAW NEGATIVE SAMPLES FROM +ve GROUPS
        #
        ################################################################
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

        neg_pairs = dict(train=set(), dev=set(), test=set())
        n_samples = raw_neg_sample_size
        
        for split, split_pairs in (
                ('train', train_pairs), 
                ('dev', dev_pairs), 
                ('test', test_pairs)
            ):
            for h, t in tqdm(
                sorted(list(split_pairs)), 
                desc=f'Creating candidate negative triples from positives for {split} ...'
            ):
                # choose left or right side to corrupt
                h_or_t = random.choice([0, 1])
                
                # corrupt the **TAIL**
                if h_or_t:
                    if corrupt_arg:
                        neg_tails = random.choices(tails_list, k=n_samples)
                    else:
                        neg_tails = random.choices(entities, k=n_samples)
                    neg_pairs[split].update({(h, t_neg) for t_neg in neg_tails})
                # corrupt the **HEAD**
                else:
                    if corrupt_arg:
                        neg_heads = random.choices(heads_list, k=n_samples)
                    else:
                        neg_heads = random.choices(entities, k=n_samples)
                    neg_pairs[split].update({(h_neg, t) for h_neg in neg_heads})
            
            # remove positive groups if created during negative sampling and inverses as well
            neg_pairs[split] = (neg_pairs[split] - pairs) - inv
            logger.info(f'collected {len(neg_pairs[split])} number of negative samples for {split}')
        
        ntr, nval, nte = 0, 0, 0
        nntr, nnval, nnte = 0, 0, 0
        
        for idx, jsonl in self.iter_entities_linked_file():
            
            if idx % 1000000 == 0 and idx > 0:
                
                ntr += len(pos_idxs['train'])
                nval += len(pos_idxs['dev'])
                nte += len(pos_idxs['test'])
                
                nntr += len(neg_idxs['train'])
                nnval += len(neg_idxs['dev'])
                nnte += len(neg_idxs['test'])
                
                ptr = (nntr / (nntr + ntr)) * 100
                pval = (nnval / (nnval + nval)) * 100
                pte = (nnte / (nnte + nte)) * 100
                logger.info(
                    f'[Progress @ {idx}] -- POSITIVE # train {ntr} / # dev {nval} / # test {nte}'
                )
                logger.info(
                    f'[Progress @ {idx}] -- NA # train {nntr} / # dev {nnval} / # test {nnte}'
                )
                logger.info(
                    f'[Progress @ {idx}] -- NA (%) # train {ptr:.2f} / # dev {pval:.2f} / # test {pte:.2f}'
                )
                
                # periodically dump the collected positive and negative samples
                # for each split to clear the memory; it grows fast :)
                for split in ['train', 'dev', 'test']:
                    
                    with open(self.pos_fname(split), 'a') as wf:
                        save_items(pos_idxs[split], wf)
                    
                    with open(self.neg_fname(split), 'a') as wf:
                        save_items(neg_idxs[split], wf)
                    
                    # reset the split indices
                    pos_idxs[split] = list()
                    neg_idxs[split] = list()
            
            ################################################################
            #
            # 2. REMOVE ENTITIES APPEARING MORE THAN ONCE (IF SET)
            #
            ################################################################
            cui2count = collections.Counter([item['id'] for item in jsonl['mentions']])
            # check if any entity is present more than once, drop this sentence
            # akin to: https://github.com/suamin/umls-medline-distant-re/blob/master/data_utils/link_entities.py#L45
            if remove_multimentions_sents:
                if sum(cui2count.values()) > len(cui2count):
                    continue
            cuis = set(cui2count.keys())
            
            ################################################################
            #
            # 3. FIND OVERLAPPING ENTITES WITH THOSE FROM SNOMED TRIPLES
            #
            ################################################################
            # NB. this intersection prunes out entities which are not covered in any of the splits
            snomed_cuis = cuis.intersection(self.entities)
            if not snomed_cuis:
                continue
            
            ################################################################
            #
            # 4. CREATE SIZE 2 (PAIRS) PERMUTATIONS OF INTERSECTING ENTITIES
            #
            ################################################################
            # permutations of size for concepts in a sentence that has SNOMED CUIs
            matching_snomed_permutations = set(itertools.permutations(snomed_cuis, 2))
            
            ################################################################
            #
            # 5. LEAKAGE CHECK: REMOVE INVERSE GROUPS FROM THESE PAIRS
            #
            ################################################################
            # first remove inverses, we are not modeling them
            matching_snomed_permutations = matching_snomed_permutations - inv
            
            if not matching_snomed_permutations:
                continue
            
            ################################################################
            #
            # 6. CHECK INTERSECTING PAIRS in TRAIN, DEV, OR TEST
            #
            ################################################################
            # check if we have the matching pairs in any of the splits
            pairs_in_train = matching_snomed_permutations.intersection(train_pairs)
            pairs_in_dev = matching_snomed_permutations.intersection(dev_pairs)
            pairs_in_test = matching_snomed_permutations.intersection(test_pairs)
            
            ################################################################
            #
            # 7. LEAKAGE CHECK - NO MATCHING PAIRS ACROSS THE SPLITS
            #
            ################################################################
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
            
            ################################################################
            #
            # 8. NO MATCHING PAIRS; CANDIDATE FOR NEGATIVE (NA) CLASS
            #
            ################################################################
            # we find no positive pairs across all the splits, this sentence
            # can be considered for not applicable (NA) relations
            if not pruned_snomed_permutations:
                
                ################################################################
                #
                # 8a. APPLY PAIRS TYPE CONSTRAINT ON NEGATIVE GROUPS
                #
                ################################################################
                if use_type_constraint:
                    # we prune out pairs which do not respect the type constraint, i.e.,
                    # the negative pair's type (TYPE_HEAD, TYPE_TAIL) must appear in
                    # pairs' types from fact triples. Next, we check if such entities
                    # appeared as head / tail **somewhere** across all the splits of facts.
                    # this is regarded as argument-role constraint. This filters out
                    # many of easy NA samples, which model can learn by simple heuristics.
                    temp = set()
                    for ent1, ent2 in matching_snomed_permutations:
                        ent1t = self.cui2sty[ent1]
                        ent2t = self.cui2sty[ent2]
                        if (ent1t, ent2t) in types:
                            pair = (ent1, ent2)
                        elif (ent2t, ent1t) in types:
                            pair = (ent2, ent1)
                        else:
                            continue
                        temp.add(pair)
                    
                    matching_snomed_permutations = temp
                    
                    if not matching_snomed_permutations:
                        continue
                
                ################################################################
                #
                # 8b. APPLY PAIRS ARG ROLE CONSTRAINT ON NEGATIVE GROUPS
                #
                ################################################################
                if use_arg_constraint:
                    temp = set()
                    for ent1, ent2 in matching_snomed_permutations:
                        if ent1 in heads and ent2 in tails:
                            pair = (ent1, ent2)
                        elif ent1 in tails and ent2 in heads:
                            pair = (ent2, ent1)
                        else:
                            continue
                        temp.add(pair)
                    
                    matching_snomed_permutations = temp
                    
                    if not matching_snomed_permutations:
                        continue
                
                ################################################################
                #
                # 8c. FROM COLLECTED GROUPS MATCH WITH THE RAW ONES
                #
                ################################################################
                # here, we consider the resultant pairs and see if they overlap
                # with our candidates generated by simple head / tail entity replacements
                candid_neg_pairs_in_test = matching_snomed_permutations.intersection(neg_pairs['test'])
                candid_neg_pairs_in_dev = matching_snomed_permutations.intersection(neg_pairs['dev'])
                candid_neg_pairs_in_train = matching_snomed_permutations.intersection(neg_pairs['train'])
                
                candid_neg_pairs = candid_neg_pairs_in_train | candid_neg_pairs_in_dev | candid_neg_pairs_in_test
                
                # after all the filtration steps we didn't find anything then we skip this sentence
                if not candid_neg_pairs:
                    continue
                
                ################################################################
                #
                # 8d. LEAKAGE CHECK - NO MATCHING PAIRS IN NEGATIVE SPLITS
                #
                ################################################################
                try:
                    assert not (candid_neg_pairs_in_train & candid_neg_pairs_in_dev)
                    assert not (candid_neg_pairs_in_train & candid_neg_pairs_in_test)
                    assert not (candid_neg_pairs_in_dev & candid_neg_pairs_in_test)
                except AssertionError:
                    continue
                
                # we might have multiple matches, so we only consider one pair to
                # associate with this sentence as negative to avoid duplicate sents in NA
                
                ################################################################
                #
                # FINAL STEP WHICH COLLECTS THE **NEGATIVE** GROUPS
                #
                ################################################################
                if candid_neg_pairs_in_test:
                    pair = random.choice(sorted(list(candid_neg_pairs_in_test)))
                    neg_idxs['test'].append((idx, pair))
                
                elif candid_neg_pairs_in_dev:
                    pair = random.choice(sorted(list(candid_neg_pairs_in_dev)))
                    neg_idxs['dev'].append((idx, pair))
                
                elif candid_neg_pairs_in_train:
                    pair = random.choice(sorted(list(candid_neg_pairs_in_train)))
                    neg_idxs['train'].append((idx, pair))
            
            else:
                
                ################################################################
                #
                # FINAL STEP WHICH COLLECTS THE **POSITIVE** GROUPS
                #
                ################################################################
                if pairs_in_test:
                    pair = random.choice(sorted(list(pairs_in_test)))
                    pos_idxs['test'].append((idx, pair))
                
                elif pairs_in_dev:
                    pair = random.choice(sorted(list(pairs_in_dev)))
                    pos_idxs['dev'].append((idx, pair))
                
                elif pairs_in_train:
                    pair = random.choice(sorted(list(pairs_in_train)))
                    pos_idxs['train'].append((idx, pair))
            
            h, t = pair
            # safety check
            assert h in cui2count
            assert t in cui2count
        
        return ntr, nval, nte, nntr, nnval, nnte
    
    def process_jsonl_with_pair(
            self, 
            jsonl: str, 
            pair: Tuple[str, str], 
            canonical_only: bool = False, 
            kb: KnowledgeBase = None
        ):
        h, t = pair
        
        # look for multiple appearence of head / tail entities, discard such ambigious examples
        mention_ids = [item['id'] for item in jsonl['mentions']]
        counts = collections.Counter(mention_ids)
        
        # make sure head and tail are present
        assert h in counts
        assert t in counts
        
        head_mention, tail_mention = None, None
        
        ################################################################
        #
        # KEEP ONLY SENTENCES WITH SINGLE MENTION OF HEAD / TAIL
        #
        ################################################################
        # since multiple mentions can cause ambiguity, we only pick instances of single h/t mention
        if counts[h] == 1 and counts[t] == 1:
            other_mentions = list()
            
            for item in jsonl['mentions']:
                if canonical_only:
                    ent = kb.cui_to_entity[item['id']]
                    names = [ent.canonical_name.lower(),] + list(map(str.lower, ent.aliases))
                    names = set(names)
                    if item['name'].lower() not in names:
                        continue
                if item['id'] == h:
                    head_mention = item
                elif item['id'] == t:
                    tail_mention = item
                else:
                    other_mentions.append(item)
            
            if head_mention == None or tail_mention == None:
                return None
            
            return head_mention, tail_mention, other_mentions
        
        else:
            return None
    
    def create_corpus(
            self, 
            train_size: float = 0.7,
            dev_size: float = 0.1,
            sample: float = 1.0, 
            use_sent_level_noise: bool = False,
            neg_prop: float = None, 
            remove_mention_overlaps: bool = True,
            canonical_or_aliases_only: bool = True,
            prune_frequent_bags: bool = True,
            max_bag_size: int = 500,
            prune_frequent_mentions: bool = True,
            max_mention_freq: int = 1000,
            min_rel_freq: int = 1,
            include_other_mentions: bool = True,
            dataset: str = 'med_distant'
        ):
        """Once positive and negative pairs have been read, we now create the corpora in
        OpenNRE format with different sizes and proportion of negative samples.
        
        """
        assert 0 < sample <= 1.0
        if neg_prop is not None:
            assert 0 < neg_prop <= 1.0
        
        counts = dict()
        
        if canonical_or_aliases_only:
            kb = UmlsEntityLinker(name='scispacy_linker').kb
        else:
            kb = None
        
        for split in ['test', 'dev', 'train']:
            
            logger.info(f'Creating corpus for split {split} ...')
            
            pos_idx2pair = read_idx_file(self.pos_fname(split))
            
            pair2pos_idxs = collections.defaultdict(list)
            for pos_idx, pair in pos_idx2pair.items():
                pair2pos_idxs[pair].append(pos_idx)
            
            logger.info(f'Found {len(pair2pos_idxs)} positive pairs in `{split}`')
            logger.info('Pruning noisy (high-frequency) positive pairs ...')
            
            ################################################################
            #
            # PRUNE HIGH-FREQUENCY POSITIVE (+ve) BAGS
            #
            ################################################################
            if prune_frequent_bags:
                # remove highly-frequent (non-informative) pairs
                for pair in list(pair2pos_idxs.keys()):
                    if len(pair2pos_idxs[pair]) > max_bag_size:
                        del pair2pos_idxs[pair]
            
            logger.info(f'Number of positive pairs after pruning = {len(pair2pos_idxs)}')
            pos_idxs = list(pos_idx2pair.keys())
            
            neg_idx2pair = read_idx_file(self.neg_fname(split))
            
            pair2neg_idxs = collections.defaultdict(list)
            for neg_idx, pair in neg_idx2pair.items():
                pair2neg_idxs[pair].append(neg_idx)
            
            logger.info(f'Found {len(pair2neg_idxs)} negative pairs in `{split}`')
            logger.info('Pruning noisy (high-frequency) negative pairs ...')
            
            ################################################################
            #
            # PRUNE HIGH-FREQUENCY NEGATIVE (-ve) BAGS
            #
            ################################################################
            if prune_frequent_bags:
                # remove highly-frequent (non-informative) pairs
                for pair in list(pair2neg_idxs.keys()):
                    if len(pair2neg_idxs[pair]) > max_bag_size:
                        del pair2neg_idxs[pair]
            
            logger.info(f'Number of negative pairs after pruning = {len(pair2neg_idxs)}')
            neg_idxs = list(neg_idx2pair.keys())
            
            ################################################################
            #
            # SUBSAMPLING POSITIVE SAMPLES
            #
            ################################################################
            # subsample the positive proportion to len(pos) * sample
            if sample < 1.0:
                logger.info(f'Subsampling positive idxs ...')
                pos_idxs = set(random.sample(pos_idxs, int(len(pos_idxs) * sample)))
            else:
                pos_idxs = set(pos_idxs)
            
            ################################################################
            #
            # ADJUST +ve TO -ve's RATIO IF NEG PROP SPECIFIED
            #
            ################################################################
            if neg_prop is not None:
                n_pos, n_neg = len(pos_idxs), len(neg_idxs)
                # if positives sample size bigger than negatives
                if n_pos > n_neg:
                    m = int(n_neg / neg_prop)
                    k_pos = m - n_neg
                    pos_idxs = set(sorted(list(pos_idxs))[:k_pos])
                # else we have more negatives than positives
                else:
                    m = int(n_pos / (1 - neg_prop))
                    k_neg = m - n_pos
                    neg_idxs = set(sorted(list(neg_idxs))[:k_neg])
            
            n_pos, n_neg = len(pos_idxs), len(neg_idxs)
            # for fast lookup
            pos_idxs = set(pos_idxs)
            neg_idxs = set(neg_idxs)
            
            ################################################################
            #
            # CHECK THERE FOR OVERLAP BETWEEN POSITIVE AND NEGATIVE SAMPLES
            #
            ################################################################
            assert not pos_idxs.intersection(neg_idxs)
            
            logger.info(f'number of positive {n_pos} and negative {n_neg} idxs')
            triples = self.triples[split]
            
            # go through positive examples
            pair2rel = dict()
            for pos_idx in pos_idxs:
                pair = pos_idx2pair[pos_idx]
                rels = triples.pair2rels[pair]
                # when labeled with multiple relations, consider a random one
                if len(rels) > 1:
                    rel = random.choice(rels)
                else:
                    rel = rels[0]
                pair2rel[pair] = rel
            
            logger.info(f'number of facts = {len(pair2rel)}')
            
            corpus = list()
            n_pos, n_neg = 0, 0 # fact instances count and NA instances count
            
            for idx, jsonl in self.iter_entities_linked_file():
                
                if idx % 1000000 == 0 and idx > 0:
                    logger.info(
                        f'[Progress @ {idx}] Collected {len(corpus)} lines with pos: {n_pos} and neg: {n_neg}'
                    )
                
                if idx in pos_idxs:
                    pair = pos_idx2pair[idx]
                    ret = self.process_jsonl_with_pair(jsonl, pair, canonical_or_aliases_only, kb)
                    if ret is None:
                        continue
                    head_mention, tail_mention, other_mentions = ret
                    
                    ################################################################
                    #
                    # APPLY SENTENCE LEVEL NOISE (cf. Amin et al., 2020)
                    #
                    ################################################################
                    if use_sent_level_noise:
                        # choose left or right side to corrupt
                        h_or_t = random.choice([0, 1])
                        has_neg = False
                        
                        for idx, m in enumerate(other_mentions):
                            # corrupt the tail
                            if h_or_t:
                                candid_neg_pair = head_mention['id'], m['id']
                                tail_mention_noise = m
                                head_mention_noise = None
                            # corrupt the head
                            else:
                                candid_neg_pair = m['id'], tail_mention['id']
                                head_mention_noise = m
                                tail_mention_noise = None
                            
                            if candid_neg_pair not in pair2neg_idxs:
                                continue
                            else:
                                has_neg = True
                            
                            example = {
                                'text': jsonl['text'],
                                'h': head_mention if h_or_t else head_mention_noise,
                                't': tail_mention if not h_or_t else tail_mention_noise,
                                'relation': 'NA'
                            }
                            
                            if include_other_mentions:
                                example['o'] = other_mentions
                            
                            corpus.append(example)
                            n_neg += 1
                        
                        if not has_neg:
                            continue
                    
                    rel = pair2rel[pair]
                    example = {
                        'text': jsonl['text'],
                        'h': head_mention,
                        't': tail_mention,
                        'relation': rel
                    }
                    
                    if include_other_mentions:
                        example['o'] = other_mentions
                    
                    corpus.append(example)
                    n_pos += 1
                
                elif idx in neg_idxs and not use_sent_level_noise:
                    pair = neg_idx2pair[idx]
                    ret = self.process_jsonl_with_pair(jsonl, pair, canonical_or_aliases_only, kb)
                    if ret is None:
                        continue
                    head_mention, tail_mention, other_mentions = ret
                    example = {
                        'text': jsonl['text'],
                        'h': head_mention,
                        't': tail_mention,
                        'relation': 'NA'
                    }
                    
                    if include_other_mentions:
                        example['o'] = other_mentions
                    
                    corpus.append(example)
                    n_neg += 1
            
            logger.info(f'Final = Collected {len(corpus)} lines with pos: {n_pos} and neg: {n_neg}')
            output_fname = os.path.join(self.base_dir, f'{self.split}{dataset}_{split}.txt')
            logger.info(f'Saving the collected corpus to output file {output_fname} ...')
            
            random.shuffle(corpus)
            
            with open(output_fname, 'w', encoding='utf-8', errors='ignore') as wf:
                for line in corpus:
                    wf.write(json.dumps(line) + '\n')
            
            counts[split] = {'npos': n_pos, 'nneg': n_neg}
        
        if remove_mention_overlaps:
            for src, tgt in [('train', 'dev'), ('train', 'test'), ('dev', 'test')]:
                src_jsonl_fname = os.path.join(self.base_dir, f'{self.split}{dataset}_{src}.txt')
                tgt_jsonl_fname = os.path.join(self.base_dir, f'{self.split}{dataset}_{tgt}.txt')
                ################################################################
                #
                # REMOVE MENTION LEVEL OVERLAPS (IF ANY)
                #
                ################################################################
                remove_overlapping_pairs(src_jsonl_fname, tgt_jsonl_fname, tgt)
        
        if prune_frequent_mentions:
            for split in ['train', 'dev', 'test']:
                jsonl_fname = os.path.join(self.base_dir, f'{self.split}{dataset}_{split}.txt')
                prune_by_type_mentions(
                    jsonl_fname, self.cui2sty, max_freq=max_mention_freq
                )
        
        # remove relations which have no sample in dev / test from train as well, use min rel freq as well
        split2rels = dict()
        for split in ['train', 'dev', 'test']:
            rels = collections.Counter()
            jsonl_fname = os.path.join(self.base_dir, f'{self.split}{dataset}_{split}.txt')
            with open(jsonl_fname, encoding='utf-8', errors='ignore') as rf:
                for line in rf:
                    line = line.strip()
                    if not line:
                        continue
                    line = json.loads(line)
                    rels.update([line['relation'],])
            split2rels[split] = rels
        
        train_relations = {rel for rel, rel_count in split2rels['train'].items() if rel_count > min_rel_freq}
        dev_relations = {rel for rel, rel_count in split2rels['dev'].items() if rel_count > min_rel_freq}
        test_relations = {rel for rel, rel_count in split2rels['test'].items() if rel_count > min_rel_freq}
        rels_to_discard = (train_relations - dev_relations) | (train_relations - test_relations)
        
        for split in ['train', 'dev', 'test']:
            rels = collections.Counter()
            jsonl_fname = os.path.join(self.base_dir, f'{self.split}{dataset}_{split}.txt')
            n_pos, n_neg = prune_by_rels(jsonl_fname, rels_to_discard)
            
            logger.info(f'Final corpus statistics ...')
            logger.info(f'{split}: {n_pos} +ve and {n_neg} NA instances')


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


def read_triples(fname):
    triples = set()
    with open(fname) as rf:
        for line in rf:
            line = line.strip()
            if not line:
                continue
            h, r, t = line.split('\t')
            triples.add((h, r, t))
    return triples


def read_triples_from_jsonl(fname):
    triples = set()
    with open(fname) as rf:
        for line in rf:
            line = line.strip()
            if not line:
                continue
            line = json.loads(line)
            h = line['h']['name']
            r = line['relation']
            t = line['t']['name']
            triples.add((h, r, t))
    return triples


def remove_overlapping_pairs(src_jsonl_fname, tgt_jsonl_fname, name):
    triples_to_remove, pairs_to_remove = check_overlap(
        read_triples_from_jsonl(src_jsonl_fname),
        read_triples_from_jsonl(tgt_jsonl_fname),
        name
    )
    
    orig = 0
    updated = 0
    updated_jsonls = list()
    with open(src_jsonl_fname) as rf:
        for jsonl in rf:
            jsonl = jsonl.strip()
            if not jsonl:
                continue
            line = json.loads(jsonl)
            orig += 1
            h = line['h']['name']
            r = line['relation']
            t = line['t']['name']
            if (h, r, t) in triples_to_remove:
                continue
            elif (t, r, h) in triples_to_remove:
                continue
            elif (h, t) in pairs_to_remove:
                continue
            elif (t, h) in pairs_to_remove:
                continue
            updated_jsonls.append(jsonl)
            updated += 1
    
    logger.info(f'Removed {orig - updated} instances from {orig}')
    
    with open(src_jsonl_fname, 'w', encoding='utf-8', errors='ignore') as wf:
        for jsonl in updated_jsonls:
            wf.write(jsonl + '\n')


def check_overlap(train_triples, test_triples, name):
    train_triples_inv = {(t, r, h) for h, r, t in train_triples}
    train_pairs = {(h, t) for h, _, t in train_triples}
    train_pairs_inv = {(t, h) for h, t in train_pairs}
    
    test_triples_inv = {(t, r, h) for h, r, t in test_triples}
    test_pairs = {(h, t) for h, _, t in test_triples}
    test_pairs_inv = {(t, h) for h, t in test_pairs}
    
    inter_triples = train_triples & test_triples
    union_triples = train_triples | test_triples
    
    inter_triples_inv = train_triples_inv & test_triples
    union_triples_inv = train_triples_inv | test_triples
    
    inter_pairs = train_pairs & test_pairs
    inter_pairs_inv = train_pairs_inv & test_pairs
    
    triples_to_remove = inter_triples | inter_triples_inv
    pairs_to_remove = inter_pairs | inter_pairs_inv
    
    logger.info(f'Training/{name} intersection size: {len(inter_triples)}')
    
    logger.info(f'Number of {name} triples in Training: '
                f'{(len(inter_triples) / len(test_triples)) * 100:.2f}%')
    
    logger.info(f'Inverse Training/{name} intersection size: {len(inter_triples_inv)}')
    
    logger.info(f'Number of {name} triples in Inverse Training: '
                f'{(len(inter_triples_inv) / len(test_triples)) * 100:.2f}%')
    
    logger.info(f'Number of {name} triples in Training or Inverse Training: '
                f'{(len(inter_triples | inter_triples_inv) / len(test_triples)) * 100:.2f}%')
    
    logger.info(f'Number of {name} pairs in Training: '
                f'{(len(inter_pairs) / len(test_pairs)) * 100:.2f}%')
    
    logger.info(f'Number of {name} pairs in Inverse Training: '
                f'{(len(inter_pairs_inv) / len(test_pairs)) * 100:.2f}%')
    
    return triples_to_remove, pairs_to_remove


def prune_by_type_mentions(benchmark_split_file, cui2sty, max_freq=1000):
    sty2mention = collections.defaultdict(set)

    logging.info(f'Reading file {benchmark_split_file} for type mentions-based pruning ...')
    
    with open(benchmark_split_file, encoding="utf-8", errors="ignore") as rf:
        for line in rf:
            line = line.strip()
            if not line:
                continue
            line = json.loads(line)
            
            h, t = line["h"]["id"], line["t"]["id"]
            h_n, t_n = line["h"]["name"], line["t"]["name"]
            r = line["relation"]
            ht = cui2sty[h]
            tt = cui2sty[t]
            if ht not in sty2mention:
                sty2mention[ht] = collections.Counter()
            if tt not in sty2mention:
                sty2mention[tt] = collections.Counter()
            sty2mention[ht].update([h_n.lower()])
            sty2mention[tt].update([t_n.lower()])
    
    mentions_to_discard = set()
    for sty, mentions in sty2mention.items():
        for mention, freq in mentions.most_common():
            if freq > max_freq: # 1000
                mentions_to_discard.add(mention)
    
    new_jsonls = list()
    count = 0
    old = 0
    with open(benchmark_split_file, encoding="utf-8", errors="ignore") as rf:
        for line in rf:
            line = line.strip()
            if not line:
                continue
            line = json.loads(line)
            old += 1 
            h_n, t_n = line["h"]["name"], line["t"]["name"]
            if h_n.lower() in mentions_to_discard or t_n.lower() in mentions_to_discard:
                count += 1
                continue
            new_jsonls.append(line)
    
    logger.info(f'removed {count} lines from total of {old} !')
    
    with open(benchmark_split_file, 'w', encoding="utf-8", errors="ignore") as wf:
        for line in new_jsonls:
            wf.write(json.dumps(line) + '\n')


def prune_by_rels(benchmark_split_file, rels_to_discard):
    logging.info(f'Reading file {benchmark_split_file} for relations pruning ...')
    
    new_jsonls = list()
    n_pos, n_neg = 0, 0
    with open(benchmark_split_file, encoding="utf-8", errors="ignore") as rf:
        for line in rf:
            line = line.strip()
            if not line:
                continue
            line = json.loads(line)
            if line["relation"] in rels_to_discard:
                continue
            if line['relation'] == 'NA':
                n_neg += 1
            else:
                n_pos += 1
            new_jsonls.append(line)
    
    with open(benchmark_split_file, 'w', encoding="utf-8", errors="ignore") as wf:
        for line in new_jsonls:
            wf.write(json.dumps(line) + '\n')
    
    return n_pos, n_neg


def add_logging_handlers(logger, dir_name, fname):
    log_file = os.path.join(dir_name, fname)
    fh = logging.FileHandler(log_file)
    fh.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', '%m/%d/%Y %H:%M:%S'
    ))
    logger.addHandler(fh)


def main(args):
    corpus = BioDSRECorpus(
        args.medline_entities_linked_fname,
        args.triples_dir,
        'ind' if args.split == 'ind' else None,
        args.has_def
    )
    
    # see if pos and neg linked files have been created before, simply check for train
    check = all(list(
        map(
            os.path.exists, 
            [corpus.pos_fname(split) for split in ['train', 'dev', 'test']] + 
            [corpus.neg_fname(split) for split in ['train', 'dev', 'test']]
        )
    ))
    
    log_file = f'corpus_{args.split}' + ('_def' if args.has_def else '') + '.log'
    add_logging_handlers(logger, corpus.base_dir, log_file)
    logger.info(vars(args))
    
    # --------------------------------------------------
    # WARNING: please don't change this to another seed, 
    # required for reproducing the corpus when we have 
    # multiple pairs matching for a given sentence
    # --------------------------------------------------
    random.seed(0)

    if not check:
        # this will take time, go grab 2 cups of coffee :)
        ntr, nval, nte, nntr, nnval, nnte = corpus.search_pos_and_neg_instances(
            raw_neg_sample_size=args.raw_neg_sample_size,
            corrupt_arg=args.corrupt_arg,
            remove_multimentions_sents=args.remove_multimentions_sents,
            use_type_constraint=args.use_type_constraint,
            use_arg_constraint=args.use_arg_constraint
        )
        
        logger.info(f'Positive and negative instances statistics ...')
        logger.info(f'--- train instances (+ve) = {ntr}')
        logger.info(f'--- dev instances (+ve) = {nval}')
        logger.info(f'--- test instances (+ve) = {nte}')
        logger.info(f'--- train NA instances = {nntr}')
        logger.info(f'--- dev NA instances = {nnval}')
        logger.info(f'--- test NA instances = {nnte}')
    
    logger.info(f'Creating corpus ...')
    
    corpus.create_corpus(
        train_size=args.train_size,
        dev_size=args.dev_size,
        sample=args.sample, 
        use_sent_level_noise=args.use_sent_level_noise,
        neg_prop=args.neg_prop, 
        remove_mention_overlaps=args.remove_mention_overlaps,
        canonical_or_aliases_only=args.canonical_or_aliases_only,
        prune_frequent_bags=args.prune_frequent_bags,
        max_bag_size=args.max_bag_size,
        prune_frequent_mentions=args.prune_frequent_mentions,
        max_mention_freq=args.max_mention_freq,
        min_rel_freq=args.min_rel_freq,
        include_other_mentions=args.include_other_mentions,
        dataset='med_distant'
    )


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
        "--has_def", action="store_true",
        help="Whether this split has definitions for the entities."
    )
    parser.add_argument(
        "--sample", action="store", type=float, default=1.0,
        help="Sub-sample the triples for a given split to obtain different sized corpora."
    )
    parser.add_argument(
        "--train_size", action="store", type=float, default=0.7,
        help="The proportion of the data to consider for training."
    )
    parser.add_argument(
        "--dev_size", action="store", type=float, default=0.1,
        help="The proportion of the data to consider for development."
    )
    parser.add_argument(
        "--neg_prop", action="store", type=float, default=None,
        help="The NA rate for negative instances."
    )
    parser.add_argument(
        "--raw_neg_sample_size", action="store", type=int, default=500,
        help="The corrupted samples fom positive pairs that are needed to subset NA candidates."
    )
    parser.add_argument(
        "--corrupt_arg", action="store_true",
        help="Corrupt entity should come from head / tail and not from full entity set."
    )
    parser.add_argument(
        "--remove_multimentions_sents", action="store_true",
        help="Remove sentences where entities occur more than once."
    )
    parser.add_argument(
        "--use_type_constraint", action="store_true",
        help="Whether to apply type constraint on argument pair of NA type sentences."
    )
    parser.add_argument(
        "--use_arg_constraint", action="store_true",
        help="Whether to argument role constraint on NA type sentences."
    )
    parser.add_argument(
        "--use_sent_level_noise", action="store_true", 
        help="Whether to generate sentence level noise."
    )
    parser.add_argument(
        "--prune_frequent_bags", action="store_true",
        help="Whether to prune most frequent bags."
    )
    parser.add_argument(
        "--max_bag_size", action="store", type=int, default=500,
        help="Remove pairs of bag sizes larger than this."
    )
    parser.add_argument(
        "--remove_mention_overlaps", action="store_true",
        help="Whether to check and remove mention level overlaps."
    )
    parser.add_argument(
        "--canonical_or_aliases_only", action="store_true",
        help="Whether to only keep the mentions which are in canonical or aliases form from UMLS."
    )
    parser.add_argument(
        "--prune_frequent_mentions", action="store_true",
        help="Whether to prune most frequent mentions."
    )
    parser.add_argument(
        "--max_mention_freq", action="store", type=int, default=1000,
        help="Maximum allowed mentions count if pruning frequent mentions."
    )
    parser.add_argument(
        "--min_rel_freq", action="store", type=int, default=1,
        help="Keep relations with sentences grater than this."
    )
    parser.add_argument(
        "--include_other_mentions", action="store_true",
        help="When creating corpora also include entity mentions other than head and tail."
    )
    
    args = parser.parse_args()
    
    import pprint
    pprint.pprint(vars(args))
    
    main(args)
