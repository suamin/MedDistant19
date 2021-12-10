# -*- coding: utf-8 -*-

import os
import time
import spacy
import argparse
import logging

from tqdm import tqdm
from pathlib import Path

from spacy.tokens import Doc
from typing import List, Generator


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class AbstractsCorpus:
    
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
    
    def __iter__(self):
        for fname in tqdm(os.listdir(self.data_dir), desc='Reading abstract texts from *.xml.gz.txt files ...'):
            if fname.endswith('.xml.gz.txt'):
                fname = Path(self.data_dir) / fname
                yield from self.iter_abstracts_from_xml_gz_txt(fname)
    
    @staticmethod
    def iter_abstracts_from_xml_gz_txt(fname: str) -> Generator[str, None, None]:
        """Read an *.xml.gz.txt file and return abstract texts stored per line."""
        with open(fname, encoding='utf-8', errors='ignore') as rf:
            for abstract in rf:
                abstract = abstract.strip()
                if not abstract:
                    continue
                yield abstract


def process_doc(doc: Doc) -> List[str]:
    return [' '.join([tok.text for tok in sent]) for sent in doc.sents]


def main(args):
    nlp = spacy.load(args.scispacy_model_name)
    abstracts = iter(AbstractsCorpus(args.data_dir))
    output_file = Path(args.data_dir) / 'medline_pubmed_2019_sents.txt'
    idx = 0
    sents = list()
    num_sents = 0
    
    total = 0
    for fname in os.listdir(args.data_dir):
        if fname.endswith('.xml.gz.txt'): total += 1
    
    t = time.time()
    
    # ---------------------------------------------------------------------------------------
    # WARNING: depending on ``n_process`` and ``batch_size`` selection multi-processing
    #          can be worse then sequential processing. One has to play around with the
    #          system a bit before it finds the right combination. There is no one size fits all!
    #
    # more here: https://spacy.io/usage/processing-pipelines#multiprocessing
    # ---------------------------------------------------------------------------------------
    for doc in tqdm(nlp.pipe(abstracts, n_process=args.n_process, batch_size=args.batch_size)):
        
        if idx % 100000 == 0 and idx > 0:
            
            speed = idx // ((time.time() - t) / 60)
            
            logger.info(f'Processed {idx} abstracts from {total} pooled abstract files @ {speed} abstracts/min ...')
            logger.info(f'Dumping batch of sentences!')
            
            with open(output_file, 'a') as wf:
                count = len(sents)
                num_sents += count
                for _ in range(count):
                    wf.write(sents.pop() + '\n') # clear out sents list
        
        for sent in process_doc(doc):
            sents.append(sent)
        idx += 1
    
    t = (time.time() - t) // 60
    
    logger.info(f'Took {t} minutes and collected {num_sents} sentences !')


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--data_dir", action="store", required=True, type=str,
        help="Path to *.xml.gz files"
    )
    parser.add_argument(
        "--scispacy_model_name", action="store", type=str, default="en_core_sci_lg",
        help="ScispaCy model to use."
    )
    parser.add_argument(
        "--n_process", action="store", type=int, default=4,
        help="Number of processes to run in parallel with spaCy multi-processing."
    )
    parser.add_argument(
        "--batch_size", action="store", type=int, default=256,
        help="Batch size to use in combination with spaCy multi-processing."
    )
    
    args = parser.parse_args()
    
    import pprint
    pprint.pprint(vars(args))
    
    main(args)


"""

python scispacy_tokenization.py \
  --data_dir MEDLINE \
  --scispacy_model_name en_sci_core_lg \
  --n_process 32 \
  --batch_size 1024

>>> 2021-12-02 10:11:07,893 : INFO : Took 588.0 minutes and collected 151920518 sentences !

"""