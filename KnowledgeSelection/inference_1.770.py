import jieba
import jieba.posseg as pseg
from tqdm import tqdm
import numpy as np
import json
import argparse
from collections import Counter
import torch
import os
from NameEntityRecognition.ner_infere import NERInfere
from IntentExtraction.mine import ExtractorModel
from NameEntityClassification.mine import ClassificationModel
from transformers import BertTokenizer
from es import get_es_relevant_entity
from collections import defaultdict


class KnowledgeSelection():
    def __init__(self, args):
        self.kb, self.entity_mapping = self.load_kb(args.kb_file)
        self.ner_infere = NERInfere(args.gpu, 
                                    args.tag_file,
                                    args.ner_pretrain_model_path,
                                    args.ner_save_model_path,
                                    args.ner_max_seq_len)

        self.intent_tokenizer = BertTokenizer.from_pretrained(args.extractor_pretrain_model_path)
        self.intent_model = ExtractorModel(torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'), 
                                           args.extractor_pretrain_model_path)
        self.intent_model.load_state_dict(torch.load(os.path.join(args.extractor_save_model_path, f"old_{args.extractor_pretrain_model_path.split('/')[-1]}.pt"), map_location='cpu'))
        self.intent_model.to(self.intent_model.device)
        self.intent_model.eval()
        
        self.classify_tokenizer = BertTokenizer.from_pretrained(args.classify_pretrain_model_path)
        self.classify_models = []
        for fold in range(5):
            self.classify_model = ClassificationModel(torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'), 
                                            args.classify_pretrain_model_path)
            self.classify_model.load_state_dict(torch.load(os.path.join(args.classify_save_model_path, f"{args.classify_pretrain_model_path.split('/')[-1]}_{fold}.pt"), map_location='cpu'))
            self.classify_model.to(self.classify_model.device)
            self.classify_model.eval()
            self.classify_models.append(self.classify_model)
            
        self.kb_enitites = list(self.kb.keys())
        self.idf = {}

        self.args = args

        for word in self.kb_enitites:
            jieba.add_word(word, 100, "entity")

        # 处理数据结构相关数据
        self.data_structure_kb, _ = self.load_data_structure_books(args.kb_file)
        self.data_structure_books = list(self.data_structure_kb.keys())
        self.data_structure_books.remove('数据结构与算法')
        self.kb.update(self.data_structure_kb)

    def load_data_structure_books(self, kbfile):
        kb = {}
        entity_mapping = {}
        with open(kbfile, 'r', encoding='utf-8') as fin:
            data = json.load(fin)
        for entity in data:
            if '数据结构' in entity:
                if "（" in entity:
                    new_entity = entity.split("（")[0]
                    entity_mapping[new_entity] = entity
                kb[entity] = {}
                for attr in data.get(entity):
                    head, rel, tail = attr
                    if rel == "Information":
                        rel = "简介"
                    if rel not in kb.get(entity):
                        kb.get(entity)[rel] = []
                    if tail not in kb.get(entity)[rel]:
                        kb.get(entity)[rel].append(tail)
        entity_mapping.pop('数据结构与算法分析')
        entity_mapping['数据结构与算法分析（C++版）'] = '数据结构与算法分析（C++版）（第二版）'
        for k, v in entity_mapping.items():
            kb[k] = kb[v]
        return kb, entity_mapping

    def load_kb(self, kbfile):
        kb = {}
        entity_mapping = defaultdict(list)
        with open(kbfile, 'r', encoding='utf-8') as fin:
            data = json.load(fin)
        for entity in data:
            if "（" in entity:
                new_entity = entity.split("（")[0]
                entity_mapping[new_entity].append(entity)  # 短映射成长
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

    def get_idf(self, sentences):
        idf = Counter()
        for sent in sentences:
            words = jieba.lcut(sent)
            words = list(set(words))
            idf.update(words)
        for key in idf:
            idf[key] = len(sentences) / idf[key]
        return idf

    def get_entity_by_jieba(self, context):
        candidates = []
        for seq in context:
            words = pseg.cut(seq)
            for (word, pos) in words:
                if pos == "entity":
                    candidates.append(word)

        pred_words = {}
        for word in candidates:
            if word not in self.kb_enitites:
                continue
            s = self.idf.get(word, 5)
            pred_words[word] = s

        pred_words = dict(sorted(pred_words.items(), key=lambda x: x[1], reverse=True))
        return list(pred_words.keys())[:1]

    def get_entity_by_es(self, context, size=1):
        es_result = get_es_relevant_entity(''.join(context).replace('你好', '').replace('。', ''), size=size)  # hacky
        re = []
        repeat = False
        if es_result:
            for each in es_result:
                if each['entity'] in ['别问我是谁', '我知道你是谁']:  # hacky
                    repeat = True
                    break
                re.append(each['entity'])
        
        if repeat:
            es_result = get_es_relevant_entity(''.join(context).replace('你好', '').replace('。', ''), size=size+2)
            re = []
            if es_result:
                for each in es_result:
                    if each['entity'] in ['别问我是谁', '我知道你是谁']:
                        continue
                    re.append(each['entity'])

        return re

    def get_entities(self, context):
        entities = Counter()
        for sent in context:
            pred_entities = self.ner_infere.ner(sent)
            for pred in pred_entities:
                if pred == "":
                    continue
                if pred in self.kb_enitites:
                    entities.update([pred])
        entities = list(entities.keys())
        if len(entities) == 0:
            entities = self.get_entity_by_jieba(context)
        return entities

    def get_intent(self, entities, query, context):
        # 纠错
        if '剧缇咗乏' in query:
            query = query.replace('剧缇咗乏', '具体做法')

        if '身嚜' in query:
            query = query.replace('身嚜', '什么')
        
        if '又挽' in query:
            query = query.replace('又挽', '游玩')
        
        # 处理数据结构相关问题
        skip_entity_search = False
        pred_entity = ''
        if '数据结构' in ''.join(context):
            tmp_context = ''.join(context).replace('。', '')
            for book in self.data_structure_books:
                if book in tmp_context:
                    pred_entity = book
                    break
            if pred_entity != '':
                skip_entity_search = True
                attrs = list(self.data_structure_kb.get(pred_entity, {}).keys())

        for i, c in enumerate(context):
            if '弍零零壹' in c:
                context[i] = c.replace('弍零零壹', '2001')

        debug = True
        if not skip_entity_search:
            es_entities = set(entities)
            for sen in context:  # 一句句找实体
                
                for each in self.get_entity_by_es([sen], size=self.args.es_size):
                    es_entities.add(each)
                    if each in self.entity_mapping:
                        for long_entity in self.entity_mapping.get(each):
                            es_entities.add(long_entity)
            es_entities = [each for each in es_entities if each in self.kb]
            
            context = ''.join(context)
                
            with torch.no_grad():
                entity_logits = []
                for fold in range(5):
                    tmp = []
                    self.classify_model = self.classify_models[fold]
                    for entity in es_entities:
                        rels = sorted(list(self.kb[entity].keys()))
                        entity = '，'.join([entity] + rels)
                        if len(context) + len(entity) + 3 > self.args.classify_max_seq_len:  # TODO
                            context = context[-(self.args.classify_max_seq_len - len(entity) - 3):]
                        sample = self.classify_tokenizer(context, entity,
                                        max_length=self.args.classify_max_seq_len,
                                        truncation=True, 
                                        padding='max_length', 
                                        return_tensors='pt')
                        outputs = self.classify_model(input_ids=sample['input_ids'].to(self.classify_model.device), 
                                    attention_mask=sample['attention_mask'].to(self.classify_model.device), 
                                    token_type_ids=sample['token_type_ids'].to(self.classify_model.device))
                        entity_logit = outputs[1][0][0][1].item()
                        tmp.append(entity_logit)
                    entity_logits.append(tmp.copy())
                entity_logits = np.array(entity_logits)
                entity_logits = np.mean(entity_logits, axis=0)
                index = np.argmax(entity_logits)
                pred_entity = es_entities[index]

                # 后处理，如果实体完全由数字组成，则舍弃
                while True:
                    check_num = 0
                    for digit in '0123456789':  
                        if digit not in pred_entity:
                            check_num += 1
                    if (10 - check_num) == len(pred_entity) and len(entity_logits) > 1:
                        entity_logits.pop(index)
                        es_entities.pop(index)
                        index = np.argmax(np.array(entity_logits))
                        pred_entity = es_entities[index]
                    else:
                        break
            
            attrs = list(self.kb.get(pred_entity, {}).keys())
        
        # 意图识别
        with torch.no_grad():
            logits = []
            for attr in attrs:
                text = query.replace(pred_entity, "ne")
                sample = self.intent_tokenizer(text, attr,
                                max_length=self.args.extractor_max_seq_len,
                                truncation=True, 
                                padding='max_length', 
                                return_tensors='pt')
                outputs = self.intent_model(input_ids=sample['input_ids'].to(self.intent_model.device), 
                            attention_mask=sample['attention_mask'].to(self.intent_model.device), 
                            token_type_ids=sample['token_type_ids'].to(self.intent_model.device))
                logit = outputs[1][0][0][1].item()
                logits.append(logit)

        intent_score_pairs = [[attr, logit] for attr, logit in zip(attrs, logits)]
        intent_score_pairs.sort(key=lambda x: x[1], reverse=True)
        i = 0
        pred_intents = []
        while i < len(intent_score_pairs) and intent_score_pairs[i][1] > self.args.multi_intent_threshold:
            pred_intents.append(intent_score_pairs[i][0])
            i += 1
        if len(pred_intents) == 0 and len(intent_score_pairs) > 0 and intent_score_pairs[0][1] > self.args.single_intent_threshold:
            # pred_intents.append(attrs[np.argmax(np.array(logits))])
            pred_intents.append(intent_score_pairs[0][0])
        return pred_intents, pred_entity, debug, intent_score_pairs

    def get_pred_knowledge(self, entity, intents):
        if entity is None:
            return []
        
        if entity not in self.kb:
            return []

        pred_knowledge = []
        for intent in intents:
            if intent not in self.kb.get(entity):
                print(f"{intent} not in {self.kb.get(entity)}")
                continue

            for value in self.kb.get(entity)[intent]:
                if intent == "简介":
                    intent = "Information"
                known = {"name": entity, "attrname": intent, "attrvalue": value}
                pred_knowledge.append(known)
        return pred_knowledge

    def load_test_data(self, test_file):
        with open(test_file, 'r', encoding='utf-8') as fin:
            data = json.load(fin)

        samples = {}
        all_messages = []
        for index in data:
            question = data[index][-1].get("message")
            context = [turn["message"] for turn in data[index]]
            all_messages.extend(context)
            sample = {"question": question, "context": context}
            samples[index] = sample
        self.idf = self.get_idf(all_messages)
        return samples

    def test(self, datafile, outputfile):
        data = self.load_test_data(datafile)

        debugs = []

        samples = {}
        for index in tqdm(data):
            question = data.get(index).get("question")
            context = data.get(index).get("context")

            entities = self.get_entities(context)
            long_entities = []
            for i, entity in enumerate(entities):
                if entity in self.entity_mapping:
                    long_entities += self.entity_mapping.get(entity)
            entities += long_entities

            intents, entity, debug, cands = self.get_intent(entities, question, context)
            if debug:
                debugs.append({
                    'question': question, 
                    'context': context,
                    'entity': entity, 
                    'intent': intents, 
                    'cands': cands
                })
                
            pred_knowledge = self.get_pred_knowledge(entity, intents)
            sample = {"question": question, "context": context, "attrs": pred_knowledge}
            samples[index] = sample

        with open(outputfile, 'w', encoding='utf-8') as fout:
            json.dump(samples, fout, ensure_ascii=False, indent=4)

        with open('debug.json', 'w', encoding='utf-8') as fout:
            json.dump(debugs, fout, ensure_ascii=False, indent=4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default="test")
    parser.add_argument('--tag_file', type=str, default="KnowledgeSelection/data/tag.txt")
    parser.add_argument('--ner_pretrain_model_path', type=str, default="luhua/chinese_pretrain_mrc_roberta_wwm_ext_large")
    parser.add_argument('--ner_save_model_path', type=str, default="KnowledgeSelection/model/ner")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--ner_max_seq_len', type=int, default=512)
    parser.add_argument('--extractor_pretrain_model_path', type=str, default="hfl/chinese-roberta-wwm-ext")
    parser.add_argument('--extractor_save_model_path', type=str, default="KnowledgeSelection/model/intent")
    parser.add_argument('--extractor_max_seq_len', type=int, default=64)
    parser.add_argument('--classify_pretrain_model_path', type=str, default="hfl/chinese-roberta-wwm-ext-large")
    parser.add_argument('--classify_save_model_path', type=str, default="KnowledgeSelection/model/classification")
    parser.add_argument('--classify_max_seq_len', type=int, default=512)
    parser.add_argument('--kb_file', type=str, default="data/kg.json")
    parser.add_argument('--test_file', type=str, default="data/test.json")
    parser.add_argument('--result_file', type=str, default="KnowledgeSelection/data/result.json")
    parser.add_argument('--multi_intent_threshold', type=float, default=2)  # TODO
    parser.add_argument('--single_intent_threshold', type=float, default=-3)  # TODO
    parser.add_argument('--es_size', type=int, default=2)  # TODO

    args = parser.parse_args()

    selector = KnowledgeSelection(args)
    if args.mode == "test":
        selector.test(args.test_file, args.result_file)


if __name__ == "__main__":
    main()
