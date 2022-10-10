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
transformers.logging.set_verbosity_error()
from sklearn.metrics import f1_score


class ClassificationModel(nn.Module):
    def __init__(self, device, model_path):
        super(ClassificationModel, self).__init__()
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


class DatasetClassification(Dataset):
    def __init__(self, data, max_seq_len, tokenizer):
        self.data = data
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        context, cand, label = self.data[index]
        if len(context) + len(cand) + 3 > self.max_seq_len:
            context = context[-(self.max_seq_len - len(cand) - 3):]
        sample = self.tokenizer(context, cand,
                                max_length=self.max_seq_len, 
                                truncation=True, 
                                padding='max_length', 
                                return_tensors='pt')
        return sample['input_ids'].squeeze(0), sample['attention_mask'].squeeze(0), sample['token_type_ids'].squeeze(0), torch.tensor(label, dtype=torch.long)


def load_data(datafile):
    with open(datafile, 'r', encoding='utf-8') as fin:
        data = json.load(fin)

    samples = []
    for sample in tqdm(data):
        candidates = sample['candidates']
        labels = sample['labels']
        context = ''.join(sample['context'])
        for cand, label in zip(candidates, labels):
            samples.append((context, cand, label))
    print(f"length of sample: {len(samples)}")
    return samples


def train(args):
    save_model_path = args.save_model_path
    max_seq_len = args.max_seq_len
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    nb_epochs = args.epochs

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Loading dataset...")

    tokenizer = BertTokenizer.from_pretrained(args.pretrain_model_path)

    train_data = load_data(args.train_file)
    train_dataset = DatasetClassification(train_data, max_seq_len, tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    dev_data = load_data(args.dev_file)
    dev_dataset = DatasetClassification(dev_data, max_seq_len, tokenizer)
    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)

    print('Creating model...')
    model = ClassificationModel(device=device, model_path=args.pretrain_model_path).to(device)
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
        score = validate(device=device, model=model, dev_dataloader=dev_dataloader, args=args)
        if score > best_score:
            best_score = score
            not_up_epoch = 0
            print('Validation f1 %f increased from previous epoch, save best_model' % score)
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

    train_data = load_data(args.train_file)
    dev_data = load_data(args.dev_file)
    all_data = train_data + dev_data
    total_num = len(all_data)
    
    for fold in range(5):
        print('fold:', fold)
        start = fold * (total_num // 5)
        end = (fold + 1) * (total_num // 5)
        train_data = all_data[0:start] + all_data[end:]
        dev_data = all_data[start:end]
        
        train_dataset = DatasetClassification(train_data, max_seq_len, tokenizer)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        dev_dataset = DatasetClassification(dev_data, max_seq_len, tokenizer)
        dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)

        print('Creating model...')
        model = ClassificationModel(device=device, model_path=args.pretrain_model_path).to(device)
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
            score = validate(device=device, model=model, dev_dataloader=dev_dataloader, args=args)
            if score > best_score:
                best_score = score
                not_up_epoch = 0
                print('Validation f1 %f increased from previous epoch, save best_model' % score)
                torch.save(model.state_dict(), os.path.join(save_model_path, f"{args.pretrain_model_path.split('/')[-1]}_{fold}.pt"))
            else:
                not_up_epoch += 1
                if not_up_epoch > args.patience:
                    print(f"Corrcoef didn't up for %s batch, early stop!" % not_up_epoch)
                    break


def validate(device, model, dev_dataloader, args):
    model.eval()
    y_truth, y_pred = [], []
    
    with torch.no_grad():
        tk = tqdm(dev_dataloader, total=len(dev_dataloader), position=0, leave=True)
        for _, (input_ids, attention_mask, token_type_ids, y) in enumerate(tk):
            input_ids, attention_mask, token_type_ids, y = input_ids.to(device), attention_mask.to(device), token_type_ids.to(device), y.to(device).long()
            outputs = model(input_ids, attention_mask, token_type_ids)
            logits = outputs[1].logits
            y_truth.extend(y.cpu().numpy())
            y_pred.extend(logits.argmax(1).cpu().numpy())

    return f1_score(y_truth, y_pred)


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

    parser.add_argument('--train_file', type=str, default='KnowledgeSelection/data/classification_train.json')
    parser.add_argument('--dev_file', type=str, default='KnowledgeSelection/data/classification_valid.json')
    parser.add_argument('--pretrain_model_path', type=str, default='hfl/chinese-roberta-wwm-ext-large')
    parser.add_argument('--save_model_path', type=str, default='KnowledgeSelection/model/classification')

    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--max_seq_len', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--validate_every', type=int, default=1)
    parser.add_argument('--patience', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    seed_everything(args.seed)  # 固定随机种子

    if args.mode == "train":
        # train(args)
        train_fold(args)
    elif args.mode == "dev":
        pass
    else:
        pass


if __name__ == "__main__":
    main()
