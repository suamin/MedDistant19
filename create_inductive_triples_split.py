# -*- coding: utf-8 -*-

"""

The code is due to @dfdazac, adopted from:
    
    https://github.com/dfdazac/blp/blob/master/data/utils.py

"""

import sys
import networkx as nx
import random
import os.path as osp
import logging
import json

from tqdm import tqdm
from argparse import ArgumentParser
from collections import Counter, defaultdict


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def parse_triples(triples_file):
    """Read a file containing triples, with head, relation, and tail
    separated by space. Returns list of lists."""
    triples = []
    rel_counts = Counter()
    file = open(triples_file)
    for line in file:
        head, rel, tail = line.split()
        triples.append([head, tail, rel])
        rel_counts[rel] += 1
    
    return triples, rel_counts


def read_entity_types(entity2type_file, keep_ents=set()):
    type2entities = defaultdict(set)
    with open(entity2type_file) as f:
        entity2type = json.load(f)
        for entity, label in entity2type.items():
            if keep_ents:
                if entity not in keep_ents:
                    continue
            type2entities[label].add(entity)
    
    return dict(type2entities)


def get_safely_removed_edges(graph, node, rel_counts, min_edges_left=100):
    """Get counts of edge removed by type, after safely removing a given node.
    Safely removing a node entails checking that no nodes are left
    disconnected, and not removing edge types with count less than
    a given amount.
    """
    neighbors = set(nx.all_neighbors(graph, node))
    removed_rel_counts = Counter()
    removed_edges = []
    
    for m in neighbors:
        # Check if m has more than 2 neighbors (node, and potentially itself)
        # before continuing
        m_neighborhood = set(nx.all_neighbors(graph, m))
        if len(m_neighborhood) > 2:
            # Check edges in both directions between node and m
            pair = [node, m]
            for i in range(2):
                edge_dict = graph.get_edge_data(*pair)
                if edge_dict is not None:
                    # Check that removing the edges between node and m
                    # does not leave less than min_edges_left
                    edges = edge_dict.values()
                    for edge in edges:
                        rel = edge['weight']
                        edges_left = rel_counts[rel] - removed_rel_counts[rel]
                        if edges_left >= min_edges_left:
                            removed_rel_counts[rel] += 1
                            head, tail = pair
                            removed_edges.append((head, tail, rel))
                        else:
                            return None
                
                # Don't count self-loops twice
                if node == m:
                    break
                
                pair = list(reversed(pair))
        else:
            return None
    
    return removed_edges, removed_rel_counts


def drop_entities(triples_file, train_size=0.8, valid_size=0.1, test_size=0.1,
                  seed=0, types_file=None, has_def=False):
    """Drop entities from a graph, to create training, validation and test
    splits.
    Entities are dropped so that no disconnected nodes are left in the training
    graph. Dropped entities are distributed between disjoint validation
    and test sets.
    """
    splits_sum = train_size + valid_size + test_size
    if splits_sum < 0 or splits_sum > 1:
        raise ValueError('Sum of split sizes must be between greater than 0'
                         ' and less than or equal to 1.')
    
    random.seed(seed)
    
    graph = nx.MultiDiGraph()
    triples, rel_counts = parse_triples(triples_file)
    graph.add_weighted_edges_from(triples)
    original_num_edges = graph.number_of_edges()
    original_num_nodes = graph.number_of_nodes()
    
    use_types = types_file is not None
    if use_types:
        type2entities = read_entity_types(types_file, set(graph.nodes))
        types = list(type2entities.keys())
    
    logger.info(f'Loaded graph with {graph.number_of_nodes():,} entities '
          f'and {graph.number_of_edges():,} edges')
    
    dropped_entities = []
    dropped_edges = dict()
    num_to_drop = int(original_num_nodes * (1 - train_size))
    num_val = int(original_num_nodes * valid_size)
    num_test = int(original_num_nodes * test_size)
    
    logger.info(f'Removing {num_to_drop:,} entities...')
    progress = tqdm(total=num_to_drop, file=sys.stdout)
    while len(dropped_entities) < num_to_drop:
        if use_types:
            # Sample an entity with probability proportional to its type count
            # (minus 1 to keep at least one entity of any type)
            weights = [len(type2entities[t]) - 1 for t in types]
            rand_type = random.choices(types, weights, k=1)[0]
            rand_ent = random.choice(list(type2entities[rand_type]))
        else:
            # Sample an entity uniformly at random
            rand_ent = random.choice(list(graph.nodes))
        
        removed_tuple = get_safely_removed_edges(graph, rand_ent, rel_counts)
        
        if removed_tuple is not None:
            removed_edges, removed_counts = removed_tuple
            dropped_edges[rand_ent] = removed_edges
            graph.remove_node(rand_ent)
            dropped_entities.append(rand_ent)
            rel_counts.subtract(removed_counts)
            
            if use_types:
                type2entities[rand_type].remove(rand_ent)
            
            progress.update(1)
    
    progress.close()
    
    # Are there indeed no disconnected nodes?
    assert len(list(nx.isolates(graph))) == 0
    
    # Did we keep track of removed edges correctly?
    num_removed_edges = sum(map(len, dropped_edges.values()))
    assert num_removed_edges + graph.number_of_edges() == original_num_edges
     
    # Test entities MUST come from first slice! This guarantees that
    # validation entities don't have edges with them (because nodes were
    # removed in sequence)
    test_ents = set(dropped_entities[:num_test])
    val_ents = set(dropped_entities[num_test:num_test + num_val])
    train_ents = set(graph.nodes())
    
    # Check that entity sets are disjoint
    assert len(train_ents.intersection(val_ents)) == 0
    assert len(train_ents.intersection(test_ents)) == 0
    assert len(val_ents.intersection(test_ents)) == 0
    
    # Check that validation graph does not contain test entities
    val_graph = nx.MultiDiGraph()
    val_edges = []
    for entity in val_ents:
        val_edges += dropped_edges[entity]
    val_graph.add_weighted_edges_from(val_edges)
    assert len(set(val_graph.nodes()).intersection(test_ents)) == 0
    
    names = ('train', 'dev', 'test')
    
    dirname = osp.dirname(triples_file)
    prefix_type = '_type' if use_types else ''
    prefix_def = '_def' if has_def else ''
    prefix = f'ind{prefix_type}{prefix_def}-'
    
    for entity_set, set_name in zip((train_ents, val_ents, test_ents), names):
        
        if set_name == 'train':
            # Triples for train split are saved later
            continue
        
        # Save file with triples for entities in set
        with open(osp.join(dirname, f'{prefix}{set_name}.tsv'), 'w') as file:
            for entity in entity_set:
                triples = dropped_edges[entity]
                for head, tail, rel in triples:
                    file.write(f'{head}\t{rel}\t{tail}\n')
    
    with open(osp.join(dirname, f'{prefix}train.tsv'), 'w') as train_file:
        for head, tail, rel in graph.edges(data=True):
            train_file.write(f'{head}\t{rel["weight"]}\t{tail}\n')
    
    logger.info(f'Dropped {len(val_ents):,} entities for validation'
          f' and {len(test_ents):,} for test.')
    logger.info(f'{graph.number_of_nodes():,} entities are left for training.')
    logger.info(f'Saved output files to {dirname}/')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--file', help='Input file')
    parser.add_argument('--types_file', help='JSON file of entities'
                                             ' and their type', default=None)
    parser.add_argument('--has_def', action='store_true', help='Whether the'
                        ' in file has defintions')
    parser.add_argument('--train_size', help='Fraction of entities used for'
                        ' training.', default=0.8, type=float)
    parser.add_argument('--seed', help='Random seed', default=0)
    args = parser.parse_args()
    
    drop_entities(args.file, train_size=args.train_size, seed=args.seed,
                  types_file=args.types_file, has_def=args.has_def)
