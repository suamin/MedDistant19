# -*- coding: utf-8 -*-

import os
import json
import collections
import pickle
import logging
import argparse
import networkx as nx
import numpy as np
import logging

from pathlib import Path


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def read_entity_rel_fact_na_maps(benchmark_split_file):
    entity_dict = dict() # map ent -> {'h' -> counts as head, 't' -> counts as tails, 'mention' -> mentions set}
    facts_dict = dict()  # map (h, r, t) -> count
    na_dict = dict()     # map (h, NA, t) -> count 
    rel_dict = dict()    # map r -> count
    bags = collections.defaultdict(set)
    
    instances = 0
    na_instances = 0
    
    logging.info(f'Reading file {benchmark_split_file} ...')
    
    with open(benchmark_split_file, encoding="utf-8", errors="ignore") as rf:
        for line in rf:
            line = line.strip()
            if not line:
                continue
            line = json.loads(line)
            
            h, t = line["h"]["id"], line["t"]["id"]
            h_n, t_n = line["h"]["name"], line["t"]["name"]
            r = line["relation"]
            triple = (h, r, t)
            
            if h not in entity_dict:
                entity_dict[h] = {"h": 0, "t": 0, "mention": set()}
            
            if t not in entity_dict:
                entity_dict[t] = {"h": 0, "t": 0, "mention": set()}
            
            entity_dict[h]["h"] += 1
            entity_dict[h]["mention"].add(h_n)
            entity_dict[t]["t"] += 1
            entity_dict[t]["mention"].add(t_n)
            
            if r == "NA":
                if triple not in na_dict:
                    na_dict[triple] = 1
                else:
                    na_dict[triple] += 1
                na_instances += 1
            else:
                if triple not in facts_dict:
                    facts_dict[triple] = 1
                else:
                    facts_dict[triple] += 1
            
            if r not in rel_dict:
                rel_dict[r] = 1
            else:
                rel_dict[r] += 1
            
            if r == 'NA':
                bags['neg'].add((h, t))
            else:
                bags['pos'].add((h, t))
            
            instances += 1
    
    na_percent = (na_instances / instances) * 100
    logger.info(f'# of instances = {instances}')
    logger.info(f'# of facts = {len(facts_dict)}')
    logger.info(f'# NA (%) = {na_percent:4.1f}%')
    logger.info(f'# of +ve bags = {len(bags["pos"])}')
    logger.info(f'# of -ve bags = {len(bags["neg"])}')
    logger.info(f'# of bags = {len(bags["pos"]) + len(bags["neg"])}')
    
    return entity_dict, rel_dict, facts_dict, na_dict


Split = collections.namedtuple(
    "Split", [
        "entity_dict", 
        "rel_dict", 
        "facts_dict",
        "na_dict"
    ]
)


def save_ents(fname, ents):
    with open(fname, "w") as wf:
        for ent in ents:
            wf.write(ent + '\n')


def save_triples(fname, triples):
    with open(fname, "w") as wf:
        for triple in triples:
            wf.write("\t".join(triple) + '\n')


def iter_split_lines(split_file):
    with open(split_file, encoding="utf-8", errors="ignore") as rf:
        for line in rf:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def create_rel2id(base_dir, splits):
    rel2id = dict()
    rels = set()
    rel2id['NA'] = 0
    for split in splits:
        path = base_dir / f'{args.dataset}_{split}.txt'
        for line in iter_split_lines(path):
            if line["relation"] != 'NA':
                rels.add(line["relation"])
    rels = sorted(list(rels))
    for idx, rel in enumerate(rels):
        rel2id[rel] = idx + 1
    return rel2id


def read_json_map(fname):
    with open(fname) as rf:
        json_map = json.load(rf)
    return json_map


def main(args):
    data = dict()
    ents = set()
    facts = set()
    na_facts = set()
    base_dir = Path(args.benchmark_dir) / args.dataset
    
    rel2id = create_rel2id(base_dir, args.splits.split(','))
    with open(base_dir / f'{args.dataset}_rel2id.json', 'w') as wf:
        json.dump(rel2id, wf)
    
    # relevant paths
    ent2type = read_json_map(Path(args.umls_dir) / 'cui2sty.json')
    ent2group = read_json_map(Path(args.umls_dir) / 'cui2sg.json')
    if 'def' in args.dataset:
        ent2def = dict()
        with open(Path(args.umls_dir) / 'cui2def.txt') as rf:
            for line in rf:
                line = line.strip()
                if not line:
                    continue
                cui, defi = line.split('\t')
                ent2def[cui] = defi
    else:
        ent2def = None
    rel2type = read_json_map(Path(args.umls_dir) / 'relation2broad.json')
    rel2cat = read_json_map(Path(args.umls_dir) / 'relation2oneormany.json')
    rel2cat_sg = read_json_map(Path(args.umls_dir) / 'relation2sg_oneormany.json')
    
    for split in args.splits.split(','):
        path = base_dir / f'{args.dataset}_{split}.txt'
        split_data = Split(*read_entity_rel_fact_na_maps(path))
        
        ent_file = base_dir / f'{split}-ents.txt'
        pos_triples_file = base_dir / f'{split}-triples.tsv'
        na_triples_file = base_dir / f'{split}-na-triples.tsv'
        
        save_ents(ent_file, sorted(split_data.entity_dict.keys()))
        save_triples(pos_triples_file, sorted(split_data.facts_dict.keys()))
        save_triples(na_triples_file, sorted(split_data.na_dict.keys()))
        
        if split != "train" and "train" in data:
            train_ents = set(data["train"].entity_dict.keys())
            unseen_split_ents = set(split_data.entity_dict.keys()) - train_ents
            unseen_ent_file = base_dir / f'{split}-unseen-ents.txt'
            save_ents(unseen_ent_file, sorted(unseen_split_ents))
            
            train_facts = set(data["train"].facts_dict.keys())
            unseen_split_facts = set(split_data.facts_dict.keys()) - train_facts
            unseen_facts_file = base_dir / f"{split}-unseen-triples.tsv"
            save_triples(unseen_facts_file, sorted(unseen_split_facts))
            
            train_na_facts = set(data["train"].na_dict.keys())
            unseen_split_na_facts = set(split_data.na_dict.keys()) - train_na_facts
            unseen_na_fact_file = base_dir / f"{split}-unseen-na-triples.tsv"
            save_triples(unseen_na_fact_file, sorted(unseen_split_na_facts))
        
        facts.update(split_data.facts_dict.keys())
        na_facts.update(split_data.na_dict.keys())
        
        ents.update(split_data.entity_dict.keys())
        data[split] = split_data
    
    ent2id = {ent: idx for idx, ent in enumerate(sorted(ents))}
    
    with open(base_dir / f'{args.dataset}_ent2id.json', "w") as wf:
        wf.write(json.dumps(ent2id))
    
    save_triples(base_dir / 'triples.tsv', sorted(facts))
    
    save_triples(base_dir / 'na-triples.tsv', sorted(na_facts))
    
    facts.update(na_facts)
    save_triples(base_dir / 'all-triples.tsv', sorted(facts))
    
    # subset relevant maps based on final entities and relations
    with open(base_dir / f'{args.dataset}_ent2type.json', "w") as wf:
        ent2type = {ent:t for ent, t in ent2type.items() if ent in ent2id}
        wf.write(json.dumps(ent2type))
    
    with open(base_dir / f'{args.dataset}_ent2group.json', "w") as wf:
        ent2group = {ent:g for ent, g in ent2group.items() if ent in ent2id}
        wf.write(json.dumps(ent2group))
    
    if ent2def:
        with open(base_dir / f'{args.dataset}_ent2def.json', "w") as wf:
            ent2def = {ent:d for ent, d in ent2def.items() if ent in ent2id}
            wf.write(json.dumps(ent2def))
    
    with open(base_dir / f'{args.dataset}_rel2type.json', "w") as wf:
        rel2type = {rel:t for rel, t in rel2type.items() if rel in rel2id and rel != 'NA'}
        rel2type['NA'] = 'None'
        wf.write(json.dumps(rel2type))
    
    with open(base_dir / f'{args.dataset}_rel2cat.json', "w") as wf:
        rel2cat = {rel:c for rel, c in rel2cat.items() if rel in rel2id and rel != 'NA'}
        rel2cat['NA'] = 'None'
        wf.write(json.dumps(rel2cat))
    
    with open(base_dir / f'{args.dataset}_rel2cat_sg.json', "w") as wf:
        rel2cat_sg = {rel:g for rel, g in rel2cat_sg.items() if rel in rel2id and rel != 'NA'}
        rel2cat['NA'] = 'None'
        wf.write(json.dumps(rel2cat_sg))
    
    data["rel2id"] = rel2id
    data["ent2id"] = ent2id
    
    with open(base_dir / f'metadata.pkl', 'wb') as wf:
        pickle.dump(data, wf)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark_dir', type=str)
    parser.add_argument('--umls_dir', type=str)
    parser.add_argument('--dataset', type=str, default='ind-med_distant')
    parser.add_argument('--splits', type=str, default='train,dev,test')
    args = parser.parse_args()
    main(args)


'''

python extract_benchmark_metadata.py --benchmark_dir benchmark --umls_dir UMLS --dataset med_distant19-S

'''