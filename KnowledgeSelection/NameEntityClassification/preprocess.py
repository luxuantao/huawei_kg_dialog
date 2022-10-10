import os
import json
import argparse
from es import get_es_relevant_entity
from tqdm import tqdm


def load_data(inputfile, outputfile):
    with open(inputfile, 'r', encoding='utf-8') as fin:
        data = json.load(fin)

    new_data = []
    error = 0
    for sample in tqdm(data):
        messages = sample.get("messages")
        previous_message = messages[0].get("message")
        candidates = set()
        es_result = get_es_relevant_entity(previous_message, size=1)
        if es_result:
            entity = es_result[0]['entity']
            candidates.add(entity)
        
        context = [previous_message]
        for i in range(1, len(messages)):
            if "attrs" in messages[i]:
                attrs = messages[i].get("attrs")
                gt_entity = attrs[0].get("name")
                if '（' in gt_entity:
                    gt_entity = gt_entity.split('（')[0]
                labels = []
                for each in list(candidates):
                    if each == gt_entity:
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
                candidates.add(entity)
                
    with open(outputfile, 'w', encoding='utf-8') as fout:
        json.dump(new_data, fout, ensure_ascii=False, indent=4)

    print("error:", error)
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default="data")
    parser.add_argument('--output_path', type=str, default="KnowledgeSelection/data")
    args = parser.parse_args()

    train_file = os.path.join(args.input_path, "train.json")
    output_train_file = os.path.join(args.output_path, "classification_train.json")
    load_data(train_file, output_train_file)

    valid_file = os.path.join(args.input_path, "valid.json")
    output_valid_file = os.path.join(args.output_path, "classification_valid.json")
    load_data(valid_file, output_valid_file)


if __name__ == "__main__":
    main()
