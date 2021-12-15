#!/bin/sh

cd ..

# export SCISPACY_CACHE= set your scispacy cache path here

# first we extract from full set of triples and CUIs which have definition available from SciSpacy's UMLSEntityLinker KB
python extract_cui_definitions.py --umls_dir UMLS

# now, we can use this newly created triples subset file for inductive split with definition
python create_inductive_triples_split.py --file UMLS/all-triples_with-def.tsv --has_def

cd UMLS

