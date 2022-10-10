import os
import json
import argparse


def load_data(inputfile, outputfile):
    new_data = []
    with open(inputfile, 'r', encoding='utf-8') as fin:
        data = json.load(fin)
        for each in data:
            tmp = []
            historys = []
            messages = each['messages']
            i = 0
            while i + 1 < len(messages):
                user = messages[i]
                machine = messages[i+1]
                historys.append({
                    "speaker": "U",
                    "text": user['message']
                })
                tmp.append({
                    "history": historys.copy(), 
                    "response": machine["message"],
                    "knowledge": machine["attrs"] if 'attrs' in machine else []
                })
                
                historys.append({
                    "speaker": "S",
                    "text": machine['message']
                })
                i += 2
            new_data.append(tmp.copy())
        
    with open(outputfile, 'w', encoding='utf-8') as fout:
        json.dump(new_data, fout, ensure_ascii=False, indent=4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default="data")
    parser.add_argument('--output_path', type=str, default="Generation/data")
    args = parser.parse_args()

    train_file = os.path.join(args.input_path, "processed_train.json")
    output_train_file = os.path.join(args.output_path, "processed_train.json")
    load_data(train_file, output_train_file)

    valid_file = os.path.join(args.input_path, "processed_valid.json")
    output_valid_file = os.path.join(args.output_path, "processed_val.json")
    load_data(valid_file, output_valid_file)


if __name__ == "__main__":
    main()
