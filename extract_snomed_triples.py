# -*- coding: utf-8 -*-

"""

The code is due to @dchang56, adopted from:
    
    https://github.com/dchang56/snomed_kge/blob/main/notebooks/umls_utils.ipynb

"""

import pandas as pd
import os
import json
import numpy as np

from collections import defaultdict


snomed_dir = 'UMLS'
data_dir = snomed_dir

## 

relations_path = os.path.join(snomed_dir, 'active_relations.txt')
semantic_types_path = os.path.join(snomed_dir, 'semantic_types.txt')
concepts_path = os.path.join(snomed_dir, 'active_concepts.txt')
semgroups_path = os.path.join(snomed_dir, 'SemGroups_2018.txt')

##

relations = pd.read_csv(relations_path, sep='\t', header=None)
relations.columns = ['CUI1', 'REL', 'CUI2', 'RELA']
relations = relations[-relations.duplicated()]

semantic_types = pd.read_csv(semantic_types_path, sep='\t', header=None)
semantic_types.columns = ['CUI', 'TUI', 'STY']

semantic_groups = pd.read_csv(semgroups_path, sep='|', header=None)
semantic_groups.columns = ['SG', 'SG_string', 'TUI', 'STY']
semantic_groups = semantic_groups.set_index('TUI')

tui2sg = semantic_groups['SG'].to_dict()
semantic_types['SemGroup'] = [tui2sg[tui] for tui in semantic_types['TUI']]

##

# filter semantic types and groups
# We want to include these groups: ANAT, CHEM, CONC, DEVI, DISO, PHEN, PHYS, PROC
# And exclude semantic types: 
exclude_types = [
    'Cell', 'Cell Component', 'Embryonic Structure', 
    'Biomedical or Dental Material', 'Chemical Viewed Functionally', 
    'Chemical Viewed Structurally', 'Regulation or Law',
    'Experimental Model of Disease', 'Molecular Function', 
    'Cell Function', 'Genetic Function'
]
include_groups = [
    'CHEM', 'DISO', 'ANAT', 'PROC', 'CONC', 'DEVI', 'PHEN', 'PHYS'
]
filtered_semantic_types = semantic_types[semantic_types['SemGroup'].isin(include_groups)]
filtered_semantic_types = filtered_semantic_types[-filtered_semantic_types['STY'].isin(exclude_types)]

## 

semantic_types.to_csv(os.path.join(data_dir, 'semantic_info.csv'), sep='\t')

##

filtered_semantic_types.to_csv(os.path.join(data_dir, 'filtered_semantic_info.csv'), sep='\t')

## 

cui2sg = filtered_semantic_types.set_index('CUI')['SemGroup'].to_dict()
with open(os.path.join(data_dir, 'cui2sg.json'), 'w') as fp:
    json.dump(cui2sg, fp)

cui2sty = filtered_semantic_types.set_index('CUI')['STY'].to_dict()
with open(os.path.join(data_dir, 'cui2sty.json'), 'w') as fp:
    json.dump(cui2sty, fp)

##

active_concepts = pd.read_csv(concepts_path, sep='\t', header=None)
active_concepts.columns = ['CUI', 'STR']
cui2string = active_concepts.set_index('CUI')['STR'].to_dict()
with open(os.path.join(data_dir, 'cui2string.json'), 'w') as fp:
    json.dump(cui2string, fp)

##

def edge_split(graph_file, files, portions):
    """
    Divide a graph into several splits.
    Parameters:
        graph_file (str): graph file
        files (list of str): file names
        portions (list of float): split portions
    """
    assert len(files) == len(portions)
    np.random.seed(0)

    portions = np.cumsum(portions, dtype=np.float32) / np.sum(portions)
    files = [open(file, "w") for file in files]
    with open(graph_file, "r") as fin:
        for line in fin:
            i = np.searchsorted(portions, np.random.rand())
            files[i].write(line)
    for file in files:
        file.close()


def filter_triplets_by_cuis(triplets, cui_iterable):
    filtered = triplets[(triplets['CUI1'].isin(cui_iterable)) & (triplets['CUI2'].isin(cui_iterable))]
    return filtered


def create_datasets(triplets, data_dir, use_ro_only=True):
    """
    2. no reciprocal relations at all
    """
    # Case 2: no reprical relations at all, so no leakage
    case2 = triplets[triplets['RELA'].isin(reciprocal_relations_dict.keys())]
    if use_ro_only:
        rels = {k for k, v in broad_rel_types.items() if v == 'RO'}
        case2 = case2[case2['RELA'].isin(rels)]
    case2 = case2.sample(frac=1, random_state=0)
    case2.to_csv(os.path.join(data_dir, 'all-triples.tsv'), sep='\t', header=None, index=None)
    
    # ds = Dataset(name='case2')
    graph_file = os.path.join(data_dir, 'all-triples.tsv')
    files = ['train.tsv', 'dev.tsv', 'test.tsv']
    files = [os.path.join(data_dir, f) for f in files]
    portions = [70, 10, 20]
    edge_split(graph_file, files, portions)
    
    case2_train = pd.read_csv(os.path.join(data_dir, 'train.tsv'), sep='\t', header=None)
    case2_train.columns = ['CUI1', 'RELA', 'CUI2']
    case2_valid = pd.read_csv(os.path.join(data_dir, 'dev.tsv'), sep='\t', header=None)
    case2_valid.columns = ['CUI1', 'RELA', 'CUI2']
    case2_test = pd.read_csv(os.path.join(data_dir, 'test.tsv'), sep='\t', header=None)
    case2_test.columns = ['CUI1', 'RELA', 'CUI2']
    
    case2_train, case2_valid, case2_test = move_unseen_to_train(case2_train, case2_valid, case2_test)
    
    case2_train = remove_overlapping_pairs(case2_train, case2_test)
    case2_train = remove_overlapping_pairs(case2_train, case2_valid)
    case2_valid = remove_overlapping_pairs(case2_valid, case2_test)
    
    check_overlap(case2_train, case2_valid, 'valid')
    check_overlap(case2_train, case2_test, 'test')
    check_overlap(case2_valid, case2_test, 'test')
    
    case2_train.to_csv(os.path.join(data_dir, 'train.tsv'), sep='\t', header=None, index=None)
    case2_valid.to_csv(os.path.join(data_dir, 'dev.tsv'), sep='\t', header=None, index=None)
    case2_test.to_csv(os.path.join(data_dir, 'test.tsv'), sep='\t', header=None, index=None)
    
    case2_train_triples = case2_train.values
    case2_train_triples_sty = [(cui2sty[h], r, cui2sty[t]) for h, r, t in case2_train_triples]
    pd.DataFrame(case2_train_triples_sty, columns=None).to_csv(
        os.path.join(data_dir, 'train_types.tsv'), sep='\t', header=None, index=None
    )
    case2_train_triples_sg = [(cui2sg[h], r, cui2sg[t]) for h, r, t in case2_train_triples]
    pd.DataFrame(case2_train_triples_sg, columns=None).to_csv(
        os.path.join(data_dir, 'train_groups.tsv'), sep='\t', header=None, index=None
    )
    
    case2_valid_triples = case2_valid.values
    case2_valid_triples_sty = [(cui2sty[h], r, cui2sty[t]) for h, r, t in case2_valid_triples]
    pd.DataFrame(case2_valid_triples_sty, columns=None).to_csv(
        os.path.join(data_dir, 'valid_types.tsv'), sep='\t', header=None, index=None
    )
    case2_valid_triples_sg = [(cui2sg[h], r, cui2sg[t]) for h, r, t in case2_valid_triples]
    pd.DataFrame(case2_valid_triples_sg, columns=None).to_csv(
        os.path.join(data_dir, 'valid_groups.tsv'), sep='\t', header=None, index=None
    )
    
    case2_test_triples = case2_test.values
    case2_test_triples_sty = [(cui2sty[h], r, cui2sty[t]) for h, r, t in case2_test_triples]
    pd.DataFrame(case2_test_triples_sty, columns=None).to_csv(
        os.path.join(data_dir, 'test_types.tsv'), sep='\t', header=None, index=None
    )
    case2_test_triples_sg = [(cui2sg[h], r, cui2sg[t]) for h, r, t in case2_test_triples]
    pd.DataFrame(case2_test_triples_sg, columns=None).to_csv(
        os.path.join(data_dir, 'test_groups.tsv'), sep='\t', header=None, index=None
    )


def move_unseen_to_train(train, valid, test):
    train_cuis = set(train['CUI1']) | set(train['CUI2'])
    valid_unseen_idx = -((valid['CUI1'].isin(train_cuis)) & (valid['CUI2'].isin(train_cuis)))
    train = pd.concat([train, valid[valid_unseen_idx]], axis=0)
    test_unseen_idx = -((test['CUI1'].isin(train_cuis)) & (test['CUI2'].isin(train_cuis)))
    train = pd.concat([train, test[test_unseen_idx]], axis=0)
    valid = valid[-valid_unseen_idx]
    test = test[-test_unseen_idx]
    return train, valid, test


def remove_overlapping_pairs(train, test):
    train_pairs = {(h, t) for h, t in train[['CUI1', 'CUI2']].values.tolist()}
    train_pairs_inv = {(t, h) for h, t in train_pairs}
    
    test_pairs = {(h, t) for h, t in test[['CUI1', 'CUI2']].values.tolist()}
    test_pairs_inv = {(t, h) for h, t in test_pairs}
    
    inter_pairs = train_pairs & test_pairs
    inter_pairs_inv = train_pairs_inv & test_pairs
    
    pairs_to_remove = inter_pairs | inter_pairs_inv
    heads, tails = zip(*pairs_to_remove)
    heads, tails = set(heads), set(tails)
    locs = list()
    for idx, (cui1, cui2) in enumerate(train[['CUI1', 'CUI2']].values.tolist()):
        if (cui1, cui2) in pairs_to_remove:
            locs.append(idx)
    for idx, (cui2, cui1) in enumerate(train[['CUI2', 'CUI1']].values.tolist()):
        if (cui2, cui1) in pairs_to_remove:
            locs.append(idx)
    train.drop(train.index[locs], inplace=True)
    return train


def check_overlap(train, test, name):
    train_triples = {(h, r, t) for h, r, t in train[['CUI1', 'RELA', 'CUI2']].values.tolist()}
    train_triples_inv = {(t, r, h) for h, r, t in train_triples}
    train_pairs = {(h, t) for h, _, t in train_triples}
    train_pairs_inv = {(t, h) for h, t in train_pairs}
    
    test_triples = {(h, r, t) for h, r, t in test[['CUI1', 'RELA', 'CUI2']].values.tolist()}
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
    
    print(f'Training/{name} intersection size: {len(inter_triples)}')
    
    print(f'Number of {name} triples in Training: '
                f'{(len(inter_triples) / len(test_triples)) * 100:.2f}%')
    
    print(f'Inverse Training/{name} intersection size: {len(inter_triples_inv)}')
    
    print(f'Number of {name} triples in Inverse Training: '
                f'{(len(inter_triples_inv) / len(test_triples)) * 100:.2f}%')
    
    print(f'Number of {name} triples in Training or Inverse Training: '
                f'{(len(inter_triples | inter_triples_inv) / len(test_triples)) * 100:.2f}%')
    
    print(f'Number of {name} pairs in Training: '
                f'{(len(inter_pairs) / len(test_pairs)) * 100:.2f}%')
    
    print(f'Number of {name} pairs in Inverse Training: '
                f'{(len(inter_pairs_inv) / len(test_pairs)) * 100:.2f}%')
    
    return triples_to_remove, pairs_to_remove


##

# Filter relations on active concepts to get final triplets
# also flipping the directions because UMLS does (tail relation head)
filtered_relations = filter_triplets_by_cuis(relations, cui2string)
filtered_relations['string1'] = [cui2string[cui] for cui in filtered_relations['CUI1']]
filtered_relations['string2'] = [cui2string[cui] for cui in filtered_relations['CUI2']]

##

relation_counts = filtered_relations['RELA'].value_counts()

##

filtered_relations['REL'].value_counts()
# RO: has relationship Other than synonymous, narrower, or broader
# PAR: has parent relationship
# CHD: has child relationship
# SY: synonymy
# RB: has a broader relationship
# RN: has a narrower relationship

# unimportant relations we might take out
relatedness_relations = [
    "same_as", "possibly_equivalent_to", "associated_with", "temporally_related_to"
]
exclude_relations = [
    "mth_plain_text_form_of", "mth_has_xml_form", "mth_has_plain_text_form", 
    "mth_xml_form_of", "replaced_by", "replaces", "uses_energy", "energy_used_by", 
    "has_dependent", "dependent_of", "part_referred_to_by", "relative_to_part_of", 
    "inherent_location_of", "has_inherent_location", "has_process_output", 
    "process_output_of", "has_precondition", "precondition_of", 
    "definitional_manifestation_of", "has_definitional_manifestation", 
    "has_technique", "technique_of"
]

##

cleaned_relations = [r for r in relation_counts.index if r not in exclude_relations]
reciprocal_relations = [r for r in cleaned_relations if r not in relatedness_relations]


reciprocal_relations_dict = {
    "isa": "inverse_isa", 
    "finding_site_of": "has_finding_site", 
    "associated_morphology_of": "has_associated_morphology", 
    "method_of": "has_method", 
    "interprets": "is_interpreted_by", 
    "direct_procedure_site_of": "has_direct_procedure_site", 
    "causative_agent_of": "has_causative_agent", 
    "active_ingredient_of": "has_active_ingredient", 
    "pathological_process_of": "has_pathological_process", 
    "entire_anatomy_structure_of": "has_entire_anatomy_structure", 
    "count_of_base_of_active_ingredient_of": "has_count_of_base_of_active_ingredient", 
    "occurs_in": "has_occurrence", 
    "dose_form_of": "has_dose_form", 
    "interpretation_of": "has_interpretation", 
    "laterality_of": "has_laterality", 
    "disposition_of": "has_disposition", 
    "component_of": "has_component", 
    "indirect_procedure_site_of": "has_indirect_procedure_site", 
    "direct_morphology_of": "has_direct_morphology", 
    "basis_of_strength_substance_of": "has_basis_of_strength_substance", 
    "precise_active_ingredient_of": "has_precise_active_ingredient", 
    "cause_of": "due_to", 
    "was_a": "inverse_was_a", 
    "temporal_context_of": "has_temporal_context", 
    "intent_of": "has_intent", 
    "direct_substance_of": "has_direct_substance", 
    "subject_relationship_context_of": "has_subject_relationship_context", 
    "uses_device": "device_used_by", 
    "presentation_strength_numerator_value_of": "has_presentation_strength_numerator_value", 
    "clinical_course_of": "has_clinical_course", 
    "focus_of": "has_focus", 
    "presentation_strength_numerator_unit_of": "has_presentation_strength_numerator_unit", 
    "presentation_strength_denominator_value_of": "has_presentation_strength_denominator_value", 
    "unit_of_presentation_of": "has_unit_of_presentation", 
    # "presentation_strength_denominator_value_of": "has_unit_of_presentation", 
    # "unit_of_presentation_of": "has_presentation_strength_denominator_value", 
    "presentation_strength_denominator_unit_of": "has_presentation_strength_denominator_unit", 
    "direct_device_of": "has_direct_device", 
    "finding_method_of": "has_finding_method", 
    "procedure_site_of": "has_procedure_site", 
    "uses_substance": "substance_used_by", 
    "specimen_of": "has_specimen", 
    "associated_finding_of": "has_associated_finding", 
    "procedure_context_of": "has_procedure_context", 
    "finding_context_of": "has_finding_context", 
    "associated_procedure_of": "has_associated_procedure", 
    "occurs_after": "occurs_before", 
    "finding_informer_of": "has_finding_informer", 
    "is_modification_of": "has_modification", 
    "concentration_strength_numerator_unit_of": "has_concentration_strength_numerator_unit", 
    "concentration_strength_numerator_value_of": "has_concentration_strength_numerator_value", 
    "concentration_strength_denominator_unit_of": "has_concentration_strength_denominator_unit", 
    "concentration_strength_denominator_value_of": "has_concentration_strength_denominator_value", 
    "uses_access_device": "access_device_used_by", 
    "access_of": "has_access", 
    "realization_of": "has_realization", 
    "specimen_source_topography_of": "has_specimen_source_topography", 
    "moved_from": "moved_to", 
    "plays_role": "role_played_by", 
    "revision_status_of": "has_revision_status", 
    "specimen_substance_of": "has_specimen_procedure", 
    "specimen_procedure_of": "has_specimen_substance", 
    "refers_to": "referred_to_by", 
    "surgical_approach_of": "has_surgical_approach", 
    "indirect_morphology_of": "has_indirect_morphology", 
    "property_of": "has_property", 
    "scale_type_of": "has_scale_type", 
    "dose_form_intended_site_of": "has_dose_form_intended_site", 
    "part_anatomy_structure_of": "has_part_anatomy_structure", 
    "dose_form_administration_method_of": "has_dose_form_administration_method", 
    "dose_form_release_characteristic_of": "has_dose_form_release_characteristic", 
    "procedure_device_of": "has_procedure_device", 
    "has_basic_dose_form": "basic_dose_form_of", 
    "dose_form_transformation_of": "has_dose_form_transformation", 
    "priority_of": "has_priority", 
    "route_of_administration_of": "has_route_of_administration", 
    "procedure_morphology_of": "has_procedure_morphology", 
    "inheres_in": "has_inherent_attribute", 
    "specimen_source_morphology_of": "has_specimen_source_morphology", 
    "specimen_source_identity_of": "has_specimen_source_identity", 
    "characterizes": "characterized_by", 
    "recipient_category_of": "has_recipient_category", 
    "during": "inverse_during", 
    "indirect_device_of": "has_indirect_device", 
    "state_of_matter_of": "has_state_of_matter", 
    "severity_of": "has_severity", 
    "alternative_of": "has_alternative", 
    "direct_site_of": "has_direct_site", 
    "time_aspect_of": "has_time_aspect", 
    "measurement_method_of": "has_measurement_method", 
    # "same_as": "same_as", 
    # "possibly_equivalent_to": "possibly_equivalent_to", 
    # "associated_with": "associated_with", 
    # "temporally_related_to": "temporally_related_to"
}


##

len(reciprocal_relations_dict) #180 relations total; 176/2=88 reciprocals, 4 symmetric

##

with open(os.path.join(data_dir, 'reciprocal_relations.json'), 'w') as fp:
    json.dump(reciprocal_relations_dict, fp)

##

# snomed subset
snomed = filter_triplets_by_cuis(filtered_relations, filtered_semantic_types['CUI'])
snomed_triplets = snomed[['CUI1', 'RELA','CUI2']]
snomed_triplets = snomed_triplets[snomed_triplets['RELA'].isin(snomed_triplets['RELA'].value_counts()[snomed_triplets['RELA'].value_counts()>15].index)]

##

snomed_triplets[snomed_triplets['CUI1']=='C0037585']

##

# Original relations has 386692 concepts and 2386877 active relations.
# After filtering 346108 active concepts, we get 2288017 relations.
# After filtering by relevant semantic types/groups, we get 2074088 relations for 293892 concepts, among which 40240 only appear once and 158314 appear less than 5 times.
# After filtering out rare relations, we get 2073848 triplets, 293884 concepts, and 170 relations

##

# two ways of looking at broader relation type metrics is to break them down 
# to 1. broad types (RO, CHD, PAR, SY, RB, RN) and 2. one-or-many types
broad_rel_types = filtered_relations.set_index('RELA')['REL'].to_dict()

##

with open(os.path.join(data_dir, 'relation2broad.json'), 'w') as fp:
    json.dump(broad_rel_types, fp)


create_datasets(snomed_triplets, data_dir)

##

rela = pd.DataFrame(snomed_triplets['RELA'].unique())
rela.columns = ['relations']
rela.to_csv(os.path.join(data_dir, 'snomed_relations.csv'), index=None)

##

snomed_cui2string = snomed.set_index('CUI1')['string1'].to_dict()
with open(os.path.join(data_dir, 'snomed_cui2string.json'), 'w') as fp:
    json.dump(snomed_cui2string, fp)

##



##

relation2one_or_many = {}
for rela in set(snomed_triplets['RELA']):
    headlist = []
    taillist = []
    pairs = snomed_triplets[snomed_triplets['RELA']==rela][['CUI1','CUI2']]
    head_per_tail = len(pairs) / len(set(pairs['CUI2']))
    tail_per_head = len(pairs) / len(set(pairs['CUI1']))
    if head_per_tail < 1.5 and tail_per_head < 1.5:
        relation2one_or_many[rela] = 'one_to_one'
    elif head_per_tail >= 1.5 and tail_per_head < 1.5:
        relation2one_or_many[rela] = 'many_to_one'
    elif head_per_tail < 1.5 and tail_per_head >= 1.5:
        relation2one_or_many[rela] = 'one_to_many'
    else:
        relation2one_or_many[rela] = 'many_to_many'

    
with open(os.path.join(data_dir, 'relation2oneormany.json'), 'w') as fp:
    json.dump(relation2one_or_many, fp)


## TODO: do the same thing for semantic types/groups (type_one_to_many, group_one_to_many, etc)
snomed_triplets['STY1'] = [cui2sty[cui] for cui in snomed_triplets['CUI1']]
snomed_triplets['STY2'] = [cui2sty[cui] for cui in snomed_triplets['CUI2']]

snomed_triplets['SG1'] = [cui2sg[cui] for cui in snomed_triplets['CUI1']]
snomed_triplets['SG2'] = [cui2sg[cui] for cui in snomed_triplets['CUI2']]

##

relation2group_oneormany = {}
for rela in set(snomed_triplets['RELA']):
    if rela not in exclude_relations:
        headlist = []
        taillist = []
        pairs = snomed_triplets[snomed_triplets['RELA']==rela][['SG1', 'SG2']]
        num_source = len(set(pairs['SG1']))
        num_target = len(set(pairs['SG2']))
        target_cardinality = (num_target/num_source)
        homo = (sum(pairs['SG1'] == pairs['SG2']) / len(pairs))
        #how homogeneous is this relation? measures whether relation is within same types/groups or not
        if num_source < 1.1 and num_target < 1.1 and homo < 0.9:
            relation2group_oneormany[rela] = 'one_to_one'
        elif num_source < 1.1 and num_target < 1.1 and homo > 0.9:
            relation2group_oneormany[rela] = 'one_to_one_homogeneous'
        elif num_source >= 1.1 and num_target < 1.1:
            relation2group_oneormany[rela] = 'many_to_one'
        elif num_source < 1.1 and num_target >= 1.1:
            relation2group_oneormany[rela] = 'one_to_many'
        elif num_source >= 1.1 and num_target >= 1.1 and homo < 0.9:
            relation2group_oneormany[rela] = 'many_to_many'
        else:
            relation2group_oneormany[rela] = 'many_to_many_homogeneous'


with open(os.path.join(data_dir, 'relation2sg_oneormany.json'), 'w') as fp:
    json.dump(relation2group_oneormany, fp)

"""
target cardinality: bigger means the relation spans more groups (heterogeneous)
homogeneity: how often does it occur for concepts within same group

classes:
many_to_one: if it's n>1 to 1
one_to_many: 1 to n>1
many_to_many: n>1 to n>1
multi_homo: n>1 to n>1 and homo
one_to_one: 1 to 1 (not homo)
homogeneous: 1 to 1 and same group

"""
