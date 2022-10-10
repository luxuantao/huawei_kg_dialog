import json
from transformers import AutoTokenizer
import pandas as pd
import matplotlib.pyplot as plt


# model_path = 'hfl/chinese-roberta-wwm-ext'
# tokenizer = AutoTokenizer.from_pretrained(model_path)


# def load_data(datafile, kb):
#     data = []
#     with open(datafile, 'r', encoding='utf-8') as fin:
#         for line in fin:
#             data.append(json.loads(line.strip()))
#     print(f"length of data: {len(data)}")

#     samples = []
#     for sample in data:
#         query = sample.get("question")
#         entity2attr = {}
#         for known in sample.get("knowledge"):
#             entity = known.get("name")
#             attrname = known.get("attrname")
#             if attrname == "Information":
#                 attrname = "简介"
#             if entity not in entity2attr:
#                 entity2attr[entity] = set()
#             entity2attr.get(entity).add(attrname)

#         for entity, attrs in entity2attr.items():
#             subgraph = kb.get(entity, {})
#             if len(subgraph) == 0:
#                 continue
#             text1 = query.replace(entity, "ne")
#             for attr in attrs:
#                 text2 = attr
#                 for key in subgraph:
#                     if key not in attrs:  # TODO check
#                         text3 = key
#                         samples.append([text1, text2, text3])
#     print(f"length of sample: {len(samples)}")
#     return samples


# def load_kb(kbfile):
#     kb = {}
#     with open(kbfile, 'r', encoding='utf-8') as fin:
#         data = json.load(fin)
#     for entity in data:
#         kb[entity] = {}
#         for attr in data.get(entity):
#             head, rel, tail = attr
#             if rel == "Information":
#                 rel = "简介"
#             if rel not in kb.get(entity):
#                 kb.get(entity)[rel] = []
#             kb.get(entity)[rel].append(str(tail))
#     print(f"length of kb: {len(kb)}")
#     return kb


# def load_dev_data(datafile, kb):
#     samples = []
#     with open(datafile, 'r', encoding='utf-8') as fin:
#         for line in fin:
#             samples.append(json.loads(line.strip()))
#     print(f"length of data: {len(samples)}")

#     data = []
#     for sample in samples:
#         query = sample.get("question")
#         entity2attr = {}
#         for known in sample.get("knowledge"):
#             entity = known.get("name")
#             attrname = known.get("attrname")
#             if attrname == "Information":
#                 attrname = "简介"
#             if entity not in entity2attr:
#                 entity2attr[entity] = set()
#             entity2attr.get(entity).add(attrname)

#         for entity, attrs in entity2attr.items():
#             subgraph = kb.get(entity, {})
#             text1 = query.replace(entity, "ne")

#             _data = {"text1": text1, "text2": [], "labels": []}
#             for attr in subgraph:
#                 text2 = attr
#                 _data.get("text2").append(text2)
#                 if attr in attrs:
#                     _data.get("labels").append(1)
#                 else:
#                     _data.get("labels").append(0)
#             data.append(_data)

#     print(f"length of sample: {len(data)}")
#     return data
    

# input_len = []
# max_in = 0
# kb = load_kb('data/kg.json')
# train_data = load_data('KnowledgeSelection/data/extractor_train.json', kb)
# for each in train_data:
#     max_in = max(max_in, len(tokenizer(each[0])['input_ids']))
#     input_len.append(len(tokenizer(each[0])['input_ids']))

# dev_data = load_dev_data('KnowledgeSelection/data/extractor_valid.json', kb)
# for each in dev_data:
#     max_in = max(max_in, len(tokenizer(each['text1'])['input_ids']))
#     input_len.append(len(tokenizer(each['text1'])['input_ids']))
# print(max_in)

# in_len = pd.Series(input_len)
# plt.plot(sorted(in_len)[:])
# plt.savefig("in_len.png")





with open('Generation/data/train/data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    
with open('Generation/data/val/data.json', 'r', encoding='utf-8') as f:
    data += json.load(f)

MODEL = "google/mt5-large"
tokenizer = AutoTokenizer.from_pretrained(MODEL)    

output_len = []
max_out_len = 0
for each in data:
    for d in each:
        out_len = len(tokenizer(d['response'])['input_ids'])
        max_out_len = max(max_out_len, out_len)
        output_len.append(out_len)
print(max_out_len)    

output_len = pd.Series(output_len)
plt.plot(sorted(output_len)[:])
plt.savefig("generation_out_len.png")
