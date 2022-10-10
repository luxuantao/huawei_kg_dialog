import os
import json
import argparse
from es import get_es_relevant_entity
from tqdm import tqdm
from collections import defaultdict


def load_kb(kbfile='data/kg.json'):
    kb = {}
    entity_mapping = defaultdict(list)
    with open(kbfile, 'r', encoding='utf-8') as fin:
        data = json.load(fin)
    for entity in data:
        if "（" in entity:
            new_entity = entity.split("（")[0]
            entity_mapping[new_entity].append(entity)  # 短映射成长  TODO check
        kb[entity] = {}
        for attr in data.get(entity):
            head, rel, tail = attr
            if rel == "Information":
                rel = "简介"
            if rel not in kb.get(entity):
                kb.get(entity)[rel] = []
            if tail not in kb.get(entity)[rel]:
                kb.get(entity)[rel].append(tail)
    print(f"length of kb: {len(kb)}")
    return kb, entity_mapping


def load_data(inputfile, outputfile, kb, entity_mapping):
    with open(inputfile, 'r', encoding='utf-8') as fin:
        data = json.load(fin)

    new_data = []
    error = 0
    for sample in tqdm(data):
        messages = sample.get("messages")
        previous_message = messages[0].get("message")
        candidates = []
        es_result = get_es_relevant_entity(previous_message, size=1)
        if es_result:
            entity = es_result[0]['entity']
            if entity in kb:
                rels = sorted(list(kb[entity].keys()))
                if [entity] + rels not in candidates:
                    candidates.append([entity] + rels)
                if entity in entity_mapping:
                    long_entities = entity_mapping[entity]
                    for long_entity in long_entities:
                        rels = sorted(list(kb[long_entity].keys()))
                        if [entity] + rels not in candidates:
                            candidates.append([long_entity] + rels)
            
        context = [previous_message]
        for i in range(1, len(messages)):
            if "attrs" in messages[i]:
                attrs = messages[i].get("attrs")
                gt_entity = attrs[0].get("name")
                labels = []
                for each in list(candidates):
                    e = each[0]
                    if e == gt_entity:
                        labels.append(1)
                    else:
                        labels.append(0)
                if sum(labels) == 0:
                    error += 1
                    
                qsample = {
                    'candidates': list(candidates),
                    'labels': labels,
                    'context': context.copy()
                }
                if previous_message.endswith("？"):
                    new_data.append(qsample)
            
            previous_message = messages[i].get("message")
            message = messages[i].get("message")
            context.append(message)        
            es_result = get_es_relevant_entity(message, size=1)
            if es_result:
                entity = es_result[0]['entity']
                if entity in kb:
                    rels = sorted(list(kb[entity].keys()))
                    if [entity] + rels not in candidates:
                        candidates.append([entity] + rels)
                    if entity in entity_mapping:
                        long_entities = entity_mapping[entity]
                        for long_entity in long_entities:
                            rels = sorted(list(kb[long_entity].keys()))
                            if [entity] + rels not in candidates:
                                candidates.append([long_entity] + rels)
                    
    with open(outputfile, 'w', encoding='utf-8') as fout:
        json.dump(new_data, fout, ensure_ascii=False, indent=4)

    print("error:", error)
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default="data")
    parser.add_argument('--output_path', type=str, default="KnowledgeSelection/data")
    args = parser.parse_args()

    kb, entity_mapping = load_kb()

    train_file = os.path.join(args.input_path, "train.json")
    output_train_file = os.path.join(args.output_path, "new_classification_train.json")
    load_data(train_file, output_train_file, kb, entity_mapping)

    valid_file = os.path.join(args.input_path, "valid.json")
    output_valid_file = os.path.join(args.output_path, "new_classification_valid.json")
    load_data(valid_file, output_valid_file, kb, entity_mapping)


if __name__ == "__main__":
    main()
