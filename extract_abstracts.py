# -*- coding: utf-8 -*-

import sys
import gzip
import xml.etree.ElementTree as ET

from tqdm import tqdm
from typing import List


def main(fname):
    fnames: List[str] = [fname,]
    
    for fname in tqdm(fnames, desc='Reading ``AbstractText`` from PubMed MEDLINE 2019 abstracts ...'):
        with open(fname + '.txt', 'w', encoding='utf-8', errors='ignore') as wf:
            with gzip.open(fname, 'rb') as rf:
                
                tree = ET.fromstring(rf.read())
                abstracts = tree.findall('.//AbstractText')
                
                for abstract in abstracts:
                    try:
                        abstract = abstract.text.strip()
                    except:
                        continue
                    
                    if not abstract:
                        continue
                    
                    # Strip starting b' or b" and ending ' or "
                    if (abstract[:2] == "b'" and abstract[-1] == "'") or (abstract[:2] == 'b"' and abstract[-1] == '"'):
                        abstract = abstract[2:-1]
                    
                    if not abstract:
                        continue
                    
                    wf.write(abstract + '\n')


if __name__ == '__main__':
    main(sys.argv[1])
