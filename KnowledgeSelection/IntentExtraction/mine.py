import os
import argparse
import random
import json
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.adamw import AdamW
from torch.utils.data import Dataset
import transformers
from transformers import AutoTokenizer, BertTokenizer, AutoModel, AutoModelForSequenceClassification
# from nezha_modeling.modeling import NeZhaModel, NeZhaForSequenceClassification


class ExtractorModel(nn.Module):
    def __init__(self, device, model_path):
        super(ExtractorModel, self).__init__()
        self.device = device
        self.model_path = model_path
        if 'nezha' in model_path:
            self.model = NeZhaForSequenceClassification.from_pretrained(model_path, num_labels=2)
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        if labels is not None:
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)
            loss = outputs[0]
            logits = outputs[1]
        else:
            loss = None
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        return loss, logits


class DatasetExtractor(Dataset):
    def __init__(self, data, max_seq_len, tokenizer):
        self.data = data
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text1, text2, label = self.data[index]
        sample = self.tokenizer(text1, text2,
                                max_length=self.max_seq_len, 
                                truncation=True, 
                                padding='max_length', 
                                return_tensors='pt')
        return sample['input_ids'].squeeze(0), sample['attention_mask'].squeeze(0), sample['token_type_ids'].squeeze(0), torch.tensor(label, dtype=torch.long)


def load_data(datafile, kb):
    data = []
    with open(datafile, 'r', encoding='utf-8') as fin:
        for line in fin:
            data.append(json.loads(line.strip()))
    print(f"length of data: {len(data)}")

    samples = []
    for sample in data:
        query = sample.get("question")
        entity2attr = {}
        for known in sample.get("knowledge"):
            entity = known.get("name")
            attrname = known.get("attrname")
            if attrname == "Information":
                attrname = "简介"
            if entity not in entity2attr:
                entity2attr[entity] = set()
            entity2attr.get(entity).add(attrname)

        for entity, attrs in entity2attr.items():
            subgraph = kb.get(entity, {})
            if len(subgraph) == 0:
                continue
            text1 = query.replace(entity, "ne")
            for attr in attrs:
                samples.append([text1, attr, 1])
            for key in subgraph:
                if key not in attrs:
                    samples.append([text1, key, 0])
    print(f"length of sample: {len(samples)}")
    return samples


def load_kb(kbfile):
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


def load_dev_data(datafile, kb):
    samples = []
    with open(datafile, 'r', encoding='utf-8') as fin:
        for line in fin:
            samples.append(json.loads(line.strip()))
    print(f"length of data: {len(samples)}")

    data = []
    for sample in samples:
        query = sample.get("question")
        entity2attr = {}
        for known in sample.get("knowledge"):
            entity = known.get("name")
            attrname = known.get("attrname")
            if attrname == "Information":
                attrname = "简介"
            if entity not in entity2attr:
                entity2attr[entity] = set()
            entity2attr.get(entity).add(attrname)

        for entity, attrs in entity2attr.items():
            subgraph = kb.get(entity, {})
            text1 = query.replace(entity, "ne")

            _data = {"text1": text1, "text2": [], "labels": []}
            for attr in subgraph:
                text2 = attr
                _data.get("text2").append(text2)
                if attr in attrs:
                    _data.get("labels").append(1)
                else:
                    _data.get("labels").append(0)
            data.append(_data)

    print(f"length of sample: {len(data)}")
    return data


def load_data_fold(data, kb):
    samples = []
    for sample in data:
        query = sample.get("question")
        entity2attr = {}
        for known in sample.get("knowledge"):
            entity = known.get("name")
            attrname = known.get("attrname")
            if attrname == "Information":
                attrname = "简介"
            if entity not in entity2attr:
                entity2attr[entity] = set()
            entity2attr.get(entity).add(attrname)

        for entity, attrs in entity2attr.items():
            subgraph = kb.get(entity, {})
            if len(subgraph) == 0:
                continue
            text1 = query.replace(entity, "ne")
            for attr in attrs:
                samples.append([text1, attr, 1])
            for key in subgraph:
                if key not in attrs:
                    samples.append([text1, key, 0])
    print(f"length of sample: {len(samples)}")
    return samples


def load_dev_data_fold(samples, kb):
    data = []
    for sample in samples:
        query = sample.get("question")
        entity2attr = {}
        for known in sample.get("knowledge"):
            entity = known.get("name")
            attrname = known.get("attrname")
            if attrname == "Information":
                attrname = "简介"
            if entity not in entity2attr:
                entity2attr[entity] = set()
            entity2attr.get(entity).add(attrname)

        for entity, attrs in entity2attr.items():
            subgraph = kb.get(entity, {})
            text1 = query.replace(entity, "ne")

            _data = {"text1": text1, "text2": [], "labels": []}
            for attr in subgraph:
                text2 = attr
                _data.get("text2").append(text2)
                if attr in attrs:
                    _data.get("labels").append(1)
                else:
                    _data.get("labels").append(0)
            data.append(_data)

    print(f"length of sample: {len(data)}")
    return data


def train(args):
    save_model_path = args.save_model_path

    max_seq_len = args.max_seq_len
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    nb_epochs = args.epochs

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Loading dataset...")

    tokenizer = BertTokenizer.from_pretrained(args.pretrain_model_path)

    kb = load_kb(args.kb_file)
    train_data = load_data(args.train_file, kb)
    train_dataset = DatasetExtractor(train_data, max_seq_len, tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    dev_data = load_dev_data(args.dev_file, kb)

    print('Creating model...')
    model = ExtractorModel(device=device, model_path=args.pretrain_model_path).to(device)
    print('Model created!')

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    optimizer.zero_grad()

    best_score = -float("inf")
    not_up_epoch = 0

    model.zero_grad()
    for epoch in range(nb_epochs):
        model.train()
        loader = tqdm(train_dataloader, total=len(train_dataloader), unit="batches")
        running_loss = 0
        for i_batch, (input_ids, attention_mask, token_type_ids, labels) in enumerate(loader):
            model.zero_grad()
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)
            labels = labels.to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)
            loss = outputs[0]
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            loader.set_postfix(Loss=running_loss / ((i_batch + 1) * batch_size), Epoch=epoch)
            loader.set_description('{}/{}'.format(epoch, nb_epochs))
            loader.update()
        score = validate(device=device, model=model, dev_data=dev_data, tokenizer=tokenizer, args=args)
        if score > best_score:
            best_score = score
            not_up_epoch = 0
            print('Validation accuracy %f increased from previous epoch, save best_model' % score)
            torch.save(model.state_dict(), os.path.join(save_model_path, f"{args.pretrain_model_path.split('/')[-1]}.pt"))
        else:
            not_up_epoch += 1
            if not_up_epoch > args.patience:
                print(f"Corrcoef didn't up for %s batch, early stop!" % not_up_epoch)
                break


def train_fold(args):
    save_model_path = args.save_model_path

    max_seq_len = args.max_seq_len
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    nb_epochs = args.epochs

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Loading dataset...")

    tokenizer = BertTokenizer.from_pretrained(args.pretrain_model_path)

    kb = load_kb(args.kb_file)
    
    train_data = []
    with open(args.train_file, 'r', encoding='utf-8') as fin:
        for line in fin:
            train_data.append(json.loads(line.strip()))
    dev_data = []
    with open(args.dev_file, 'r', encoding='utf-8') as fin:
        for line in fin:
            dev_data.append(json.loads(line.strip()))
    all_data = train_data + dev_data
    total_num = len(all_data)
    
    for fold in range(5):
        print('fold:', fold)
        start = fold * (total_num // 5)
        end = (fold + 1) * (total_num // 5)
        train_data = all_data[0:start] + all_data[end:]
        dev_data = all_data[start:end]
    
        train_data = load_data_fold(train_data, kb)
        dev_data = load_dev_data_fold(dev_data, kb)
    
        train_dataset = DatasetExtractor(train_data, max_seq_len, tokenizer)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

        print('Creating model...')
        model = ExtractorModel(device=device, model_path=args.pretrain_model_path).to(device)
        print('Model created!')

        optimizer = AdamW(model.parameters(), lr=learning_rate)
        optimizer.zero_grad()

        best_score = -float("inf")
        not_up_epoch = 0

        model.zero_grad()
        for epoch in range(nb_epochs):
            model.train()
            loader = tqdm(train_dataloader, total=len(train_dataloader), unit="batches")
            running_loss = 0
            for i_batch, (input_ids, attention_mask, token_type_ids, labels) in enumerate(loader):
                model.zero_grad()
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                token_type_ids = token_type_ids.to(device)
                labels = labels.to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)
                loss = outputs[0]
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                loader.set_postfix(Loss=running_loss / ((i_batch + 1) * batch_size), Epoch=epoch)
                loader.set_description('{}/{}'.format(epoch, nb_epochs))
                loader.update()
            score = validate(device=device, model=model, dev_data=dev_data, tokenizer=tokenizer, args=args)
            if score > best_score:
                best_score = score
                not_up_epoch = 0
                print('Validation accuracy %f increased from previous epoch, save best_model' % score)
                torch.save(model.state_dict(), os.path.join(save_model_path, f"{args.pretrain_model_path.split('/')[-1]}_{fold}.pt"))
            else:
                not_up_epoch += 1
                if not_up_epoch > args.patience:
                    print(f"Corrcoef didn't up for %s batch, early stop!" % not_up_epoch)
                    break


def validate(device, model, dev_data, tokenizer, args):
    model.eval()
    acc = 0
    total = len(dev_data)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    with torch.no_grad():
        for sample in tqdm(dev_data):
            text1 = sample.get("text1")
            text2s = sample.get("text2")
            labels = sample.get("labels")
            logits = []
            for text2 in text2s:
                sample = tokenizer(text1, text2,
                                    max_length=args.max_seq_len, 
                                    truncation=True, 
                                    padding='max_length', 
                                    return_tensors='pt')
                outputs = model(input_ids=sample['input_ids'].to(device), 
                              attention_mask=sample['attention_mask'].to(device), 
                              token_type_ids=sample['token_type_ids'].to(device))
                
                logit = outputs[1][0][0][1].item()
                logits.append(logit)
            pred = np.array(logits).argmax()
            acc += labels[pred]

    print('acc:', acc / total)
    return acc / total


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='train')

    parser.add_argument('--train_file', type=str, default='KnowledgeSelection/data/extractor_train.json')
    parser.add_argument('--dev_file', type=str, default='KnowledgeSelection/data/extractor_valid.json')
    parser.add_argument('--kb_file', type=str, default='data/kg.json')
    parser.add_argument('--pretrain_model_path', type=str, default='hfl/chinese-roberta-wwm-ext')
    parser.add_argument('--save_model_path', type=str, default='KnowledgeSelection/model/intent')

    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--max_seq_len', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--validate_every', type=int, default=1)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    seed_everything(args.seed)  # 固定随机种子

    if args.mode == "train":
        train(args)
        # train_fold(args)
    elif args.mode == "dev":
        pass
    else:
        pass


if __name__ == "__main__":
    main()
