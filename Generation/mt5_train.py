import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, AutoTokenizer
from transformers.models.mt5.modeling_mt5 import MT5ForConditionalGeneration
import random
from tqdm import tqdm
import logging
import jieba
jieba.setLogLevel(logging.CRITICAL)
import json
from itertools import chain
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'true'


class Config:
    TRAIN_BATCH_SIZE = 8
    VALID_BATCH_SIZE = 8
    TRAIN_EPOCHS = 50
    LEARNING_RATE = 1e-4
    SEED = 42
    MAX_IN_LEN = 512  # TODO
    MAX_OUT_LEN = 200  # TODO
    ACCUM_ITER = 1
    WEIGHT_DECAY = 1e-2
    MODEL = "google/mt5-large"
    BEAMS = 5
    history_max_tokens = 200  # TODO
    knowledge_max_tokens = 300  # TODO
    do_train = True
    do_test = False
    patience = 3


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


CFG = Config()
seed_everything(CFG.SEED)  # 固定随机种子
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CustomDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.tokenizer = tokenizer
        self.data = data
        self.speaker1 = self.tokenizer.convert_tokens_to_ids("<speaker1>")
        self.speaker2 = self.tokenizer.convert_tokens_to_ids("<speaker2>")
        self.knowledge_sep = self.tokenizer.convert_tokens_to_ids("<knowledge_sep>")
        self.pad_id = self.tokenizer.convert_tokens_to_ids("<pad>")
        self._create_examples()

    def _truncate_sequences(self, sequences, max_length):
        words_to_cut = sum(list(map(len, sequences))) - max_length
        if words_to_cut <= 0:
            return sequences

        while words_to_cut > len(sequences[0]):
            words_to_cut -= len(sequences[0])
            sequences = sequences[1:]
        
        sequences[0] = sequences[0][words_to_cut:]
        return sequences

    def _create_examples(self):
        self.examples = []
        for sample in self.data:
            history = sample["history"]
            response = sample["response"]
            knowledge = sample["knowledge"]
            historys = []
            for index in range(len(history)):
                history_ids = self.tokenizer.convert_tokens_to_ids(
                    self.tokenizer.tokenize(history[index]["text"]))
                historys.append(history_ids)
            historys = [
                [self.speaker1 if i % 2 == 0 else self.speaker2] + s
                for i, s in enumerate(historys)
            ]
            truncated_history = self._truncate_sequences(historys, CFG.history_max_tokens)

            with self.tokenizer.as_target_tokenizer():
                response = self.tokenizer(response, max_length=CFG.MAX_OUT_LEN, padding='max_length', truncation=True, return_tensors='pt')
            response = response["input_ids"].squeeze(0)  # [bs]
            response[response == self.tokenizer.pad_token_id] = -100

            if len(knowledge) == 0:
                used_knowledge = []
            else:
                str_knowledge = ""
                for item in knowledge:
                    temp = "-".join([str(item["attrname"]), str(item["attrvalue"]), str(item["name"])])
                    str_knowledge = ";".join([str_knowledge, temp])
                used_knowledge = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(str_knowledge))
                used_knowledge = used_knowledge[:CFG.knowledge_max_tokens]

            self.examples.append({
                "history": truncated_history,
                "knowledge": used_knowledge,
                "response": response
            })

    def build_input_from_segments(self, knowledge, history, response):
        sequence = [knowledge + [self.knowledge_sep]] + history
        input_ids = list(chain(*sequence))
        attention_mask = [1] * len(input_ids) + [0] * (CFG.MAX_IN_LEN - len(input_ids))
        input_ids += [self.pad_id] * (CFG.MAX_IN_LEN - len(input_ids))
        labels = response

        return torch.tensor(input_ids), torch.tensor(attention_mask), labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        example = self.examples[index]
        input_ids, attention_mask, labels = self.build_input_from_segments(
            example["knowledge"],
            example["history"],
            example["response"]
        )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class TestDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.tokenizer = tokenizer
        self.data = data
        self.speaker1 = self.tokenizer.convert_tokens_to_ids("<speaker1>")
        self.speaker2 = self.tokenizer.convert_tokens_to_ids("<speaker2>")
        self.knowledge_sep = self.tokenizer.convert_tokens_to_ids("<knowledge_sep>")
        self.pad_id = self.tokenizer.convert_tokens_to_ids("<pad>")
        self._create_examples()

    def _truncate_sequences(self, sequences, max_length):
        words_to_cut = sum(list(map(len, sequences))) - max_length
        if words_to_cut <= 0:
            return sequences

        while words_to_cut > len(sequences[0]):
            words_to_cut -= len(sequences[0])
            sequences = sequences[1:]
        
        sequences[0] = sequences[0][words_to_cut:]
        return sequences

    def _create_examples(self):
        self.examples = []
        for dialog_id, sample in self.data.items():
            history = sample["context"]
            knowledge = sample["attrs"]
            historys = []
            for index in range(len(history)):
                history_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(history[index]))
                historys.append(history_ids)
            historys = [
                [self.speaker1 if i % 2 == 0 else self.speaker2] + s
                for i, s in enumerate(historys)
            ]
            truncated_history = self._truncate_sequences(historys, CFG.history_max_tokens)

            if len(knowledge) == 0:
                used_knowledge = []
            else:
                str_knowledge = ""
                for item in knowledge:
                    temp = "-".join([str(item["attrname"]), str(item["attrvalue"]), str(item["name"])])
                    str_knowledge = ";".join([str_knowledge, temp])
                used_knowledge = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(str_knowledge))
                used_knowledge = used_knowledge[:CFG.knowledge_max_tokens]

            self.examples.append({
                "dialog_id": dialog_id,
                "history": truncated_history,
                "knowledge": used_knowledge
            })

    def build_input_from_segments(self, knowledge, history):
        sequence = [knowledge + [self.knowledge_sep]] + history
        input_ids = list(chain(*sequence))
        attention_mask = [1] * len(input_ids) + [0] * (CFG.MAX_IN_LEN - len(input_ids))
        input_ids += [self.pad_id] * (CFG.MAX_IN_LEN - len(input_ids))

        return torch.tensor(input_ids), torch.tensor(attention_mask)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        example = self.examples[index]
        dialog_id = example['dialog_id']
        input_ids, attention_mask = self.build_input_from_segments(
            example["knowledge"],
            example["history"]
        )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "dialog_id": dialog_id
        }


class AverageMeter:  # 为了tqdm实时显示loss和acc
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(epoch, tokenizer, model, device, loader, optimizer):
    print('epoch:', epoch)
    model.train()
    losses = AverageMeter()
    tk = tqdm(loader, total=len(loader), position=0, leave=True)
    for _, data in enumerate(tk):
        inputs = {k: v.to(device) for k, v in data.items()}
        outputs = model(**inputs)
        loss = outputs[0]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item() * CFG.ACCUM_ITER, inputs['input_ids'].size(0))
        tk.set_postfix(loss=losses.avg)


def validate(epoch, tokenizer, model, device, loader):
    model.eval()
    losses = AverageMeter()
    tk = tqdm(loader, total=len(loader), position=0, leave=True)
    with torch.no_grad():
        for _, data in enumerate(tk):
            inputs = {k: v.to(device) for k, v in data.items()}
            outputs = model(**inputs)
            loss = outputs[0]

            losses.update(loss.item() * CFG.ACCUM_ITER, inputs['input_ids'].size(0))
            tk.set_postfix(loss=losses.avg)
    return losses.avg


def inference(tokenizer, model, device, loader):
    model.eval()
    predictions = []
    tk = tqdm(loader, total=len(loader), position=0, leave=True)
    with torch.no_grad():
        for _, data in enumerate(tk):
            generated_ids = model.generate(
                input_ids=data['input_ids'].to(device, dtype=torch.long),
                attention_mask=data['attention_mask'].to(device, dtype=torch.long),
                max_length=CFG.MAX_OUT_LEN,
                num_beams=CFG.BEAMS,
                repetition_penalty=1.0,
                length_penalty=1.0,
                early_stopping=True
            )
            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
            predictions.extend(preds)
    return predictions


def main():
    model = MT5ForConditionalGeneration.from_pretrained(CFG.MODEL).to(device)
    tokenizer = AutoTokenizer.from_pretrained(CFG.MODEL)

    tokenizer.add_tokens(["<speaker1>", "<speaker2>", "<knowledge_sep>"])
    model.resize_token_embeddings(len(tokenizer))

    train_params = {
        'batch_size': CFG.TRAIN_BATCH_SIZE,
        'shuffle': True,
        'num_workers': 8
    }

    val_params = {
        'batch_size': CFG.VALID_BATCH_SIZE,
        'shuffle': False,
        'num_workers': 8
    }

    if CFG.do_train:
        with open('Generation/data/processed_train.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        train_data = []
        for dialogs in data:
            for dialog in dialogs:
                train_data.append(dialog)
        
        with open('Generation/data/processed_val.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        dev_data = []
        for dialogs in data:
            for dialog in dialogs:
                dev_data.append(dialog)
    
        train_set = CustomDataset(train_data, tokenizer)
        val_set = CustomDataset(dev_data, tokenizer)
        
        train_loader = DataLoader(train_set, **train_params)
        val_loader = DataLoader(val_set, **val_params)
    
    if CFG.do_test:  
        with open('KnowledgeSelection/data/result.json', 'r', encoding='utf-8') as f:
            test_data = json.load(f)

        test_set = TestDataset(test_data, tokenizer)

        test_loader = DataLoader(test_set, **val_params)

    if CFG.do_train:
        optimizer = AdamW(model.parameters(), lr=CFG.LEARNING_RATE, weight_decay=CFG.WEIGHT_DECAY)

        best_loss = float('inf')
        for epoch in range(CFG.TRAIN_EPOCHS):
            train(epoch, tokenizer, model, device, train_loader, optimizer)
            valid_loss = validate(epoch, tokenizer, model, device, val_loader)
            if valid_loss < best_loss:
                best_loss = valid_loss
                torch.save(model.state_dict(), 'Generation/model/{}.pt'.format(CFG.MODEL.split('/')[-1]))
                print('best_loss:', best_loss)
                no_update = 0
            elif valid_loss >= best_loss and (no_update < CFG.patience):
                no_update += 1
                print(f"no update: {no_update}")
            elif no_update == CFG.patience:
                print("Model has exceed patience. Exiting")
                break

    if CFG.do_test:
        model.load_state_dict(torch.load('Generation/model/{}.pt'.format(CFG.MODEL.split('/')[-1])))
        predictions = inference(tokenizer, model, device, test_loader)
        with open('generation_result.txt', 'w', encoding='utf-8') as f:
            for each in predictions:
                f.write(each + '\n')


if __name__ == '__main__':
    main()
