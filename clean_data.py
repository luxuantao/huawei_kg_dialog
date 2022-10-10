import json


def split_question(data):
    new_data = []
    for each in data:
        name = each['name']
        messages = each['messages']
        new_messages = []
        j = 0
        for i, dialog in enumerate(messages):
            if (i + j) % 2 == 0:
                new_messages.append(dialog)
            else:
                message = dialog['message']
                if not message.endswith('?') and not message.endswith('？'):
                    new_messages.append(dialog)
                else:
                    idx = message.rfind('，')
                    idx2 = message.rfind('。')
                    idx = max(idx, idx2)
                    if idx == -1:
                        new_messages.append(dialog)
                    else:
                        dialog['message'] = message[:idx] + '。'
                        new_messages.append(dialog)
                        new_messages.append({'message': message[idx+1:]})
                        j += 1
        new_data.append({'name': name, 'messages': new_messages})
    return new_data

with open('data/train.json', 'r', encoding='utf-8') as f:
    train_data = json.load(f)
    
with open('data/train.json', 'w', encoding='utf-8') as f:
    json.dump(train_data, f, ensure_ascii=False, indent=4)

new_train_data = split_question(train_data)

with open('data/processed_train.json', 'w', encoding='utf-8') as f:
    json.dump(new_train_data, f, ensure_ascii=False, indent=4)

with open('data/valid.json', 'r', encoding='utf-8') as f:
    val_data = json.load(f)
    
with open('data/valid.json', 'w', encoding='utf-8') as f:
    json.dump(val_data, f, ensure_ascii=False, indent=4)

new_val_data = split_question(val_data)

with open('data/processed_valid.json', 'w', encoding='utf-8') as f:
    json.dump(new_val_data, f, ensure_ascii=False, indent=4)
