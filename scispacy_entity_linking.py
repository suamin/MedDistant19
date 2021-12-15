# -*- coding: utf-8 -*-

import spacy
import os
import pandas as pd
import numpy as np
import time
import logging
import warnings
import itertools
import json
import argparse
import fasttext

from tqdm import tqdm
from pathlib import Path
from scispacy.umls_linking import UmlsEntityLinker

from spacy.tokens import Doc
from spacy import Language
from typing import Iterable, Generator, List, Dict, Tuple, Union


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


def load_entity_linking_model(
        scispacy_model_name: str = "en_core_sci_lg",
        cache_dir: Union[str, Path] = None
    ) -> Language:
    """
    Parameters
    ----------
    
    scispacy_model_name: `str`, optional, (default = 'en_core_sci_lg')
        Name of the scispacy model to use for entity linking.
    cache_dir: `str` or `Path`, optional, (default = None)
        Path to set up for the scispacy cache directory. 
    
    Returns
    -------
    
        A spacy language pipeline``Language``.
    
    """
    if cache_dir is not None:
        os.environ['SCISPACY_CACHE'] = cache_dir
    
    nlp = spacy.load(scispacy_model_name)
    
    # We use the defaults set in scispacy see 
    # https://github.com/allenai/scispacy/blob/main/scispacy/linking.py#L67
    logger.info('Loading and adding ``UmlsEntityLinker`` to ``nlp.pipe`` ...')
    nlp.add_pipe('scispacy_linker')
    
    return nlp


def process_tagged_doc(doc: Doc) -> List[Dict[str, Union[str, Tuple[int, int]]]]:
    return [
        {
            'id': ent._.umls_ents[0][0],
            'pos': [ent.start_char, ent.end_char],
            'name': str(ent)
        } for ent in doc.ents if ent._.umls_ents
    ]


def iter_sentences_from_txt(args) -> Generator[str, None, None]:
    with open(args.medline_unique_sents_fname, encoding='utf-8', errors='ignore') as rf:
        for sent in tqdm(rf, desc='Reading sentences ...'):
            sent = sent.strip()
            if not sent:
                continue
            
            tokens = sent.split()
            # Remove too short or too long sentences
            if len(tokens) < args.min_sent_tokens or len(tokens) > args.max_sent_tokens:
                continue
            
            yield sent


def main(args):
    nlp = load_entity_linking_model(args.scispacy_model_name, args.cache_dir)
    
    # A bit slow and memory intensive but better option when doing language identification as well
    sents = iter_sentences_from_txt(args)
    output_file = args.output_file
    
    idx = 0
    jsonls = list()
    
    t = time.time()
    
    for sent in tqdm(nlp.pipe(sents, n_process=args.n_process, batch_size=args.batch_size)):
        
        if idx % 1000000 == 0 and idx > 0:
            speed = idx // ((time.time() - t) / 60)
            
            logger.info(f'Processed {idx} sents @ {speed} sents/min ...')
            logger.info(f'Dumping batch of sentences!')
            
            with open(output_file, 'a') as wf:
                count = len(jsonls)
                for _ in range(count):
                    wf.write(json.dumps(jsonls.pop()) + '\n') # clear out sents list

        mentions = process_tagged_doc(sent)
        
        # Consider only mentions with 2 or more (need at least two ents)
        if len(mentions) >= 2:
            jsonls.append({'text': sent.text, 'mentions': mentions})
        else:
            continue
        
        idx += 1
    
    t = (time.time() - t) // 60
    
    logger.info(f'Took {t} minutes !')
    


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--medline_unique_sents_fname", action="store", required=True, type=str,
        help="Path to unique sentences file extracted from PubMed MEDLINE."
    )
    parser.add_argument(
        "--output_file", action="store", required=True, type=str,
        help="Path to output file in jsonl format."
    )
    parser.add_argument(
        "--scispacy_model_name", action="store", type=str, default="en_core_sci_lg",
        help="SciSpacy model to use for UMLS concept linking."
    )
    parser.add_argument(
        "--cache_dir", action="store", type=str, default="/netscratch/samin/cache/scispacy",
        help="Path to SciSpacy cache directory. Optionally, set the environment "
        "variable ``SCISPACY_CACHE``."
    )
    parser.add_argument(
        "--n_process", action="store", type=int, default=8,
        help="Number of processes to run in parallel with spaCy multi-processing."
    )
    parser.add_argument(
        "--batch_size", action="store", type=int, default=256,
        help="Batch size to use in combination with spaCy multi-processing."
    )
    parser.add_argument(
        "--min_sent_tokens", action="store", type=int, default=5,
        help="Minimum sentence length in terms of tokens."
    )
    parser.add_argument(
        "--max_sent_tokens", action="store", type=int, default=128,
        help="Maximum sentence length in terms of tokens."
    )
    
    args = parser.parse_args()
    
    import pprint
    pprint.pprint(vars(args))
    
    main(args)
