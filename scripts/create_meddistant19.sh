#!/bin/sh

python create_kb_aligned_text_corpora.py \
--medline_entities_linked_fname MEDLINE/medline_pubmed_2019_entity_linked.jsonl \
--triples_dir UMLS \
--split ind \
--sample 0.1 \
--train_size 0.7 \
--dev_size 0.1 \
--raw_neg_sample_size 500 \
--corrupt_arg \
--remove_multimentions_sents \
--use_type_constraint \
--use_arg_constraint \
--remove_mention_overlaps \
--canonical_or_aliases_only \
--prune_frequent_mentions \
--max_mention_freq 1000 \
--min_rel_freq 1 \
--prune_frequent_mentions \
--prune_frequent_bags \
--max_bag_size 500
