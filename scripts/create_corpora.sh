#!/bin/sh

$python create_kb_aligned_text_corpora.py \
--medline_entities_linked_fname MEDLINE/medline_pubmed_2019_entity_linked.jsonl \
--triples_dir UMLS \
--size L \
--split trans \
--train_size 0.7 \
--dev_size 0.1 \
--neg_prop 0.9 \
--raw_neg_sample_size 50 \
--use_type_constraint \
--use_arg_constraint \
--max_bag_size 500
