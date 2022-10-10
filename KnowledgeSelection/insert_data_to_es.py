from elasticsearch import Elasticsearch
import json
from tqdm import tqdm
from collections import defaultdict


es = Elasticsearch()

# 清空原有数据
try:
    es.indices.delete(index='huawei')
except:
    pass

mapping = {  # 定义数据存储格式
    'properties': {
        'entity': {
            'type': 'text',
            'analyzer': 'ik_max_word',
            'search_analyzer': 'ik_max_word'
        }
    }
}
result = es.indices.create(index='huawei', ignore=400)  # 创建索引
print(result)
result = es.indices.put_mapping(index='huawei', body=mapping)  # 创建mapping
print(result)


def load_kb(kbfile='data/kg.json'):
    kb = {}
    with open(kbfile, 'r', encoding='utf-8') as fin:
        data = json.load(fin)
    for entity in data:
        kb[entity] = {}
        for attr in data.get(entity):
            head, rel, tail = attr
            if rel == "Information":
                rel = "简介"
            if rel not in kb.get(entity):
                kb.get(entity)[rel] = []
            kb.get(entity)[rel].append(str(tail))
    print(f"length of kb: {len(kb)}")
    return kb


kb = load_kb()
entities = kb.keys()

for entity in tqdm(entities):
    es.index(index='huawei', body={'entity': entity})




# 清空原有数据
try:
    es.indices.delete(index='zuofa')
except:
    pass

mapping = {  # 定义数据存储格式
    'properties': {
        'detail': {
            'type': 'text',
            'analyzer': 'ik_max_word',
            'search_analyzer': 'ik_max_word'
        }
    }
}
result = es.indices.create(index='zuofa', ignore=400)  # 创建索引
print(result)
result = es.indices.put_mapping(index='zuofa', body=mapping)  # 创建mapping
print(result)


def load_zuofa(kbfile='data/kg.json'):
    s = set()
    with open(kbfile, 'r', encoding='utf-8') as fin:
        data = json.load(fin)
    for entity in data:
        for attr in data.get(entity):
            head, rel, tail = attr
            if rel == "做法":
                s.add(tail)
    return s

s = load_zuofa()

for detail in tqdm(s):
    es.index(index='zuofa', body={'detail': detail})




try:
    es.indices.delete(index='chengyu')
except:
    pass

mapping = {  # 定义数据存储格式
    'properties': {
        'detail': {
            'type': 'text',
            'analyzer': 'ik_max_word',
            'search_analyzer': 'ik_max_word'
        }
    }
}
result = es.indices.create(index='chengyu', ignore=400)  # 创建索引
print(result)
result = es.indices.put_mapping(index='chengyu', body=mapping)  # 创建mapping
print(result)


def load_chengyu(kbfile='data/kg.json'):
    d = {}
    with open(kbfile, 'r', encoding='utf-8') as fin:
        data = json.load(fin)
    for entity in data:
        for attr in data.get(entity):
            head, rel, tail = attr
            if rel == "释义" and len(head) == 4:
                d[tail] = head
                break
    return d

d = load_chengyu()

for detail in tqdm(d):
    es.index(index='chengyu', body={'detail': detail})

with open('成语.json', 'w', encoding='utf-8') as f:
    json.dump(d, f, ensure_ascii=False, indent=4)





def load_location(kbfile='data/kg.json'):
    d = defaultdict(list)
    with open(kbfile, 'r', encoding='utf-8') as fin:
        data = json.load(fin)
    for entity in data:
        for attr in data.get(entity):
            head, rel, tail = attr
            if rel == "分布":
                d[tail].append(head)
    return d

d = load_location()

with open('分布.json', 'w', encoding='utf-8') as f:
    json.dump(d, f, ensure_ascii=False, indent=4)
