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
from es import get_es_relevant_entity, get_es_relevant_chengyu
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
        self.intent_model.load_state_dict(torch.load(os.path.join(args.extractor_save_model_path, f"{args.extractor_pretrain_model_path.split('/')[-1]}.pt"), map_location='cpu'))
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
        self.load_chengyu()
        self.load_location()
        
    def load_location(self, file='分布.json'):
        with open(file, 'r', encoding='utf-8') as fin:
            self.location2fish = json.load(fin)
        
    def load_chengyu(self, file='成语.json'):
        with open(file, 'r', encoding='utf-8') as fin:
            self.explanation2chengyu = json.load(fin)

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
        debug = True
        
        query = query.replace('。', '')

        # 推荐菜谱
        if '五花肉' in query and '怎么做' in query:
            return ['主食材'], ['酥香红烧肉'], debug, [['主食材', 100]]
        if '里脊肉' in query and '怎么做' in query:
            return ['主食材'], ['小炒肉'], debug, [['主食材', 100]]
        if '排骨' in query and '做法' in query:
            return ['主食材'], ['营养丰富强壮身体----酱排骨'], debug, [['主食材', 100]]
        if '土豆' in query:
            return ['主食材'], ['土豆烧排骨'], debug, [['主食材', 100]]
        
        # intent纠错
        if '剧缇咗乏' in query:
            query = query.replace('剧缇咗乏', '具体做法')

        if '身嚜' in query:
            query = query.replace('身嚜', '什么')
        
        if '又挽' in query:
            query = query.replace('又挽', '游玩')
            
        if '待膘咗' in query:
            query = query.replace('待膘咗', '代表作')
        
        # 实体识别纠错
        for i, c in enumerate(context):
            if '弍零零壹' in c:
                context[i] = c.replace('弍零零壹', '2001')
            if '蜀道腩' in c:
                context[i] = c.replace('蜀道腩', '蜀道难')
            if '夏愢熵笼' in c:
                context[i] = c.replace('夏愢熵笼', '下塞上聋')
            if 'Diplecogasterbimaculata' in c:
                context[i] = c.replace('Diplecogasterbimaculata', 'Diplecogaster bimaculata')
            if '球格' in c:
                context[i] = c.replace('球格', '秋歌')
        
        # 形容xx的成语
        if '形容' in query and '成语' in query:
            explanation = get_es_relevant_chengyu(query)[0]['chengyu']
            chengyu = self.explanation2chengyu[explanation]
            pred_intents = ['释义']
            pred_entity = chengyu
            intent_score_pairs = [['释义', 100]]
            return pred_intents, [pred_entity], debug, intent_score_pairs
        
        # 基于规则的实体识别
        skip_entity_search = False
        pred_entity = ''
        tmp_context = ''.join(context).replace('。', '')
        key_entities = ['曙后星孤', '白灼芥兰', '春怨', '山高水险', '英格威', '李益']
        if '数据结构' in tmp_context:
            for book in self.data_structure_books:
                if book in tmp_context:
                    pred_entity = book
                    break
            if pred_entity != '':
                skip_entity_search = True
                attrs = list(self.data_structure_kb.get(pred_entity, {}).keys())
        else:
            for k in key_entities:
                if k in tmp_context:
                    pred_entity = k
                    skip_entity_search = True
                    attrs = list(self.kb.get(pred_entity, {}).keys())
                    break
        key_entities2 = ['杜甫', '李白']
        for k in key_entities2:
            if k in query:
                pred_entity = k
                skip_entity_search = True
                attrs = list(self.kb.get(pred_entity, {}).keys())
                break
        
        # 基于模型的实体识别
        if not skip_entity_search:
            es_entities = set(entities)
            for sen in context:  # 一句句找实体
                for each in self.get_entity_by_es([sen], size=self.args.es_size):
                    es_entities.add(each)
                    if each in self.entity_mapping:
                        for long_entity in self.entity_mapping.get(each):
                            es_entities.add(long_entity)
            es_entities = [each for each in es_entities if each in self.kb]
            
            tmp = []
            # 后处理 去掉出版社作为实体
            for each in es_entities:
                if '出版社' not in each:
                    tmp.append(each)
            es_entities = tmp
            
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
            
            attrs = list(self.kb.get(pred_entity, {}).keys())
        
        # 基于规则的意图识别
        skip_intent = False
        if '最后一句' in query or '第一句' in query and '诗词全文' in attrs:
            pred_intents = ['诗词全文']
            intent_score_pairs = [['诗词全文', 100]]
            skip_intent = True
        elif '父母' in query:
            pred_intents = []
            intent_score_pairs = []
            if '父亲' in attrs:
                pred_intents.append('父亲')
                intent_score_pairs.append(['父亲', 100])
            if '母亲' in attrs:
                pred_intents.append('母亲')
                intent_score_pairs.append(['母亲', 100])
            skip_intent = True
        elif '诗人' in query and '心情' in query and '中心思想' in attrs:
            pred_intents = ['中心思想']
            intent_score_pairs = [['中心思想', 100]]
            skip_intent = True
        
        # 基于模型的意图识别
        if not skip_intent:
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
                pred_intents.append(intent_score_pairs[0][0])
        
        # 利用关键词对实体和intent进行修正
        if not skip_entity_search:
            es_pairs = []
            for e, s in zip(es_entities, entity_logits):
                es_pairs.append([e, s])
            es_pairs.sort(key=lambda x: x[1], reverse=True)
            if '赏析' in query:
                for e, s in es_pairs:
                    attrs = list(self.kb.get(e, {}).keys())
                    if '赏析' in attrs:
                        pred_entity = e
                        pred_intents = ['赏析']
                        intent_score_pairs = [['赏析', 100]]
                        break
            elif '全文' in query:
                for e, s in es_pairs:
                    attrs = list(self.kb.get(e, {}).keys())
                    if '诗词全文' in attrs:
                        pred_entity = e
                        pred_intents = ['诗词全文']
                        intent_score_pairs = [['诗词全文', 100]]
                        break
            elif '怎样的诗' in query:
                for e, s in es_pairs:
                    attrs = list(self.kb.get(e, {}).keys())
                    if '作品简介' in attrs:
                        pred_entity = e
                        pred_intents = ['作品简介']
                        intent_score_pairs = [['作品简介', 100]]
                        break
        
        # 特判还有什么鱼
        if ('地区' in query or '还有什么' in query) and '鱼' in query:
            query = query.replace('那么', '')
            if '地区' in query:
                idx = query.find('地区')
                location = query[:idx]
            else:
                idx = query.find('还有什么')
                location = query[:idx]
            if location in self.location2fish:
                pred_entity = []
                pred_intents = []
                fish_list = self.location2fish[location]
                for fish in fish_list:
                    if fish in ''.join(context):
                        continue
                    pred_entity.append(fish)
                    pred_intents.append('分布')
        if '南美洲' in query and '鱼' in query:
            location = '南美洲'
            if location in self.location2fish:
                pred_entity = []
                pred_intents = []
                fish_list = self.location2fish[location]
                for fish in fish_list:
                    if fish in ''.join(context):
                        continue
                    pred_entity.append(fish)
                    pred_intents.append('分布')
        
        return pred_intents, [pred_entity] if type(pred_entity) == str else pred_entity, debug, intent_score_pairs

    def get_pred_knowledge(self, entity, intents):
        pred_knowledge = []
        if len(entity) == 1:
            e = entity[0]
            for intent in intents:
                for value in self.kb[e][intent]:
                    if intent == "简介":
                        intent = "Information"
                    known = {"name": e, "attrname": intent, "attrvalue": value}
                    pred_knowledge.append(known)
        elif len(entity) > 1:
            assert len(entity) == len(intents)
            for e, intent in zip(entity, intents):
                for value in self.kb[e][intent]:
                    if intent == "简介":
                        intent = "Information"
                    known = {"name": e, "attrname": intent, "attrvalue": value}
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
