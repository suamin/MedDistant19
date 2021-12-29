# -*- coding: utf-8 -*-

import spacy
import json
import logging
import os
import argparse
import re

from scispacy.umls_linking import UmlsEntityLinker
from pathlib import Path


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def load_umls_linker(cache_dir=None):
    if cache_dir is not None:
        os.environ['SCISPACY_CACHE'] = cache_dir
    logger.info('Loading ``UmlsEntityLinker`` ...')
    linker = UmlsEntityLinker(name='scispacy_linker')
    return linker


def load_cuis(umls_dir):
    with open(Path(umls_dir) / 'cui2string.json') as rf:
        cui2string = json.load(rf)
    return sorted(cui2string.keys())


def search_definitions(linker, cuis):
    logger.info(f'Searching for definitions of {len(cuis)} CUIs ...')
    missing_cuis_in_db = 0
    cuis_in_db_missing_definitions = 0
    cui2def = dict()
    for cui in cuis:
        if cui in linker.kb.cui_to_entity:
            items = linker.kb.cui_to_entity[cui]
            definition = items[4]
            if not definition:
                cuis_in_db_missing_definitions += 1
                continue
            else:
                definition = re.sub(r'\s+', ' ', definition)
                cui2def[cui] = definition
        else:
            missing_cuis_in_db += 1
            continue
    logger.info(f'Found definitions for {len(cui2def)} / {len(cuis)} CUIs')
    logger.info(f'CUIs missing in UMLS2020AA db = {missing_cuis_in_db}')
    logger.info(f'CUIs found in UMLS2020AA db but missing definition = {cuis_in_db_missing_definitions}')
    return cui2def


def create_triples(umls_dir, cui2def):
    all_triples_with_def = list()
    total = 0
    with_def = 0
    logger.info(f'Subsetting the triples that have definitions ...')
    with open(Path(umls_dir) / 'all-triples.tsv') as rf, open(Path(umls_dir) / 'all-triples_with-def.tsv', 'w') as wf:
        for line in rf:
            line = line.strip()
            if not line:
                continue
            h, r, t = line.split('\t')
            total += 1
            if h in cui2def and t in cui2def:
                wf.write(line + '\n')
                with_def += 1
    logger.info(f'Found triples with definitions {with_def} out of {total}')


def main(args):
    cui2def = search_definitions(load_umls_linker(args.cache_dir), load_cuis(args.umls_dir))
    with open(Path(args.umls_dir) / 'cui2def.txt', 'w') as wf:
        for cui, defi in sorted(cui2def.items()):
            wf.write(f'{cui}\t{defi}\n')
    create_triples(args.umls_dir, cui2def)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--umls_dir", action="store", required=True, type=str,
        help="Path to UMLS directory."
    )
    parser.add_argument(
        "--cache_dir", action="store", type=str, default=None,
        help="Path to SciSpacy cache directory. Optionally, set the environment "
        "variable ``SCISPACY_CACHE``."
    )
    
    args = parser.parse_args()
    
    import pprint
    pprint.pprint(vars(args))
    
    main(args)
