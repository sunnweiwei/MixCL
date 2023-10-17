import sys

import faiss

sys.path += ['./']
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import BartForConditionalGeneration, BartTokenizer
from accelerate import Accelerator
from utils.evaluation import eval_f1, eval_all, f1_score, eval_acc, eval_em
from utils.io import write_file
import nltk
import json
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import copy
import spacy
import torch
import os
import logging
import warnings

warnings.filterwarnings("ignore")
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)


def tok_text(text):
    return ' '.join(text.split()).strip().lower()


ROLE = {'0_Wizard': 'User1: ', '1_Apprentice': 'User2: ', '0_Apprentice': 'User2: ',
        '1_Wizard': 'User1: ', 'apprentice': 'User2: '}


def load_wiki(file, head='Wikipedia denosing. '):
    def mask(text):
        text = text.split()
        ids = [j for j in range(int(len(text) * 0.5))]
        np.random.shuffle(ids)
        for j in ids:
            text[j] = '<mask>'
        return ' '.join(text)

    data = json.load(open(file))
    context = []
    response = []
    for title, passage in data.items():
        passage = [s + '</s>' for s in passage]
        context.append([f'{head}Topic: ' + title + '</s>'])
        response.append(passage[0])
        for i in range(0, len(passage) - 1):
            context.append([f'{title}Topic: ' + title + '</s>', mask(passage[i])])
            response.append(passage[i] + passage[i + 1])
    return context, response


def load_dialog(file, topic_subset=None, title='Predict response.  '):
    data = json.load(open(file))
    data_context = []
    data_response = []
    for example in data:
        if topic_subset is not None and example['chosen_topic'] not in topic_subset:
            continue
        context = [ROLE[turn['speaker']] + turn['text'] + '</s>' for turn in example['context']]
        topic = f'{title}Topic: ' + example['chosen_topic'] + '</s>'
        data_context.append([topic] + context)
        data_response.append(example['labels'][0])
    return data_context, data_response


def load_batch_dialog(file, title='Predict response.  '):
    data = json.load(open(file))
    data_context = []
    data_response = []
    for example in data:
        context = [ROLE[turn['speaker']] + turn['text'] + '</s>' for turn in example['context']]
        topic = f'{title}Topic: ' + example['chosen_topic'] + '</s>'
        data_context.append([topic] + context)
        data_response.append(example['labels'][0])
    return data_context, data_response


def load_knowledge(file, title='Predict knowledge. '):
    data = json.load(open(file))
    data_context = []
    data_response = []
    for example in data:
        if example['title'] == 'no_passages_used':
            continue
        context = [ROLE[turn['speaker']] + turn['text'] + '</s>' for turn in example['context']]
        topic = f'{title}Topic: ' + example['chosen_topic'] + '</s>'
        data_context.append([topic] + context)
        data_response.append(example['title'] + ' ' + example['checked_sentence'])
    return data_context, data_response


def load_batch_knowledge(file, title='Predict knowledge. '):
    data = json.load(open(file))
    data_context = []
    data_response = []
    for example in data:
        if example['title'] == 'no_passages_used':
            continue
        context = [ROLE[turn['speaker']] + turn['text'] + '</s>' for turn in example['context']]
        topic = f'{title}Topic: ' + example['chosen_topic'] + '</s>'
        item_context = []
        item_response = []
        item_context.append([topic] + context)
        item_response.append(example['title'] + ' ' + example['checked_sentence'])
        for k in example['knowledge']:
            for s in example['knowledge'][k]:
                if k == example['title'] or s == example['checked_sentence']:
                    continue
                item_context.append([topic] + context)
                item_response.append(k + ' ' + s)
        data_context.append(item_context)
        data_response.append(item_response)
    return data_context, data_response


def load_batch_knowledge_span(file):
    data = json.load(open(file))
    data = [item[1] for item in data]
    return data


def load_batch_knowledge_ner(file, knowledge_response):
    data = json.load(open(file))
    new_data = []
    for a, b in zip(data, knowledge_response):
        new_line = [[y, x] for x, y in zip(a, b)]
        new_data.append(new_line)
    return new_data


def rounder(number):
    rand = np.random.rand()
    if rand < number - int(number):
        return int(number) + 1
    else:
        return int(number)


def choice_index(number, sample_size):
    for i in range(len(sample_size)):
        if number < sum(sample_size[:i + 1]):
            return i, number - sum(sample_size[:i])


class Data(Dataset):
    def __init__(self, context, response, tokenizer, context_len=256, response_len=128):
        super(Dataset, self).__init__()
        self.context = context
        self.response = response
        self.tokenizer = tokenizer
        self.context_len = context_len
        self.response_len = response_len

    def __getitem__(self, index):
        context = self.context[index]
        response = self.response[index]
        if len(context[0]) > 0:
            topic = self.tokenizer.encode(context[0])
        else:
            topic = []
        context = topic + self.tokenizer.encode(' '.join(context[1:]))[-(self.context_len - len(topic)):]
        response = self.tokenizer.encode(response, truncation=True, max_length=self.response_len)
        context = torch.tensor(context)
        response = torch.tensor(response)
        return context, response

    def __len__(self):
        return len(self.context)

    @staticmethod
    def collate_fn(data):
        context, response = zip(*data)
        context = pad_sequence(context, batch_first=True, padding_value=1)
        return {
            'input_ids': context,
            'attention_mask': context.ne(1),
            'labels': pad_sequence(response, batch_first=True, padding_value=-100),
        }

class CLData(Dataset):
    def __init__(self, context, response, ratio, tokenizer, sign=None, context_len=256, response_len=128, **kwargs):
        super(Dataset, self).__init__()
        self.context = context
        self.response = response
        self.ratio = ratio
        self.data_size = [len(c) for c in self.context]
        self.sample_size = [int(self.data_size[0] / self.ratio[0] * r) for r in self.ratio]
        print(self.data_size, self.sample_size, [c1 / c2 for c1, c2 in zip(self.sample_size, self.data_size)])
        self.tokenizer = tokenizer
        self.context_len = context_len
        self.response_len = response_len
        self.sign = sign

    def __getitem__(self, index):
        corpus_id, index = choice_index(index, self.sample_size)
        sign = self.sign[corpus_id] if self.sign is not None else 1
        rand = np.random.rand()
        index = rounder((index + rand) / self.sample_size[corpus_id] * self.data_size[corpus_id])
        index = min(index, len(self.context[corpus_id]) - 1)
        # corpus_id = np.random.choice([c for c in range(len(self.ratio))], p=self.ratio)
        # index = rounder(index / self.data_size[0] * self.data_size[corpus_id])
        context = self.context[corpus_id][index]
        response = self.response[corpus_id][index]
        topic = self.tokenizer.encode(context[0])
        context = topic + self.tokenizer.encode(' '.join(context[1:]))[-(self.context_len - len(topic)):]
        response = self.tokenizer.encode(response, truncation=True, max_length=self.response_len)
        context = torch.tensor(context)
        response = torch.tensor(response)
        return context, response, sign

    def __len__(self):
        return sum(self.sample_size)

    @staticmethod
    def collate_fn(data):
        context, response, sign = zip(*data)
        context = pad_sequence(context, batch_first=True, padding_value=1)
        return {
            'input_ids': context,
            'attention_mask': context.ne(1),
            'labels': pad_sequence(response, batch_first=True, padding_value=-100),
            'sign': torch.tensor(sign)
        }


class BatchData(CLData):
    def __init__(self, *args, neg_num=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.neg_num = neg_num

    def set_neg_num(self, neg_num):
        self.neg_num = neg_num

    def __getitem__(self, index):
        corpus_id, index = choice_index(index, self.sample_size)
        sign = self.sign[corpus_id] if self.sign is not None else 1
        rand = np.random.rand()
        index = rounder((index + rand) / self.sample_size[corpus_id] * self.data_size[corpus_id])
        index = min(index, len(self.context[corpus_id]) - 1)
        context = self.context[corpus_id][index]
        response = self.response[corpus_id][index]
        if not isinstance(response, list):
            context = [context]
            response = [response]
        idd = [i for i in range(1, len(response))]
        np.random.shuffle(idd)
        if self.neg_num is None:
            idd = []
        else:
            idd = idd[:self.neg_num[corpus_id]]
        idd = [0] + idd
        context = context[0]
        topic = self.tokenizer.encode(context[0])
        context = topic + self.tokenizer.encode(' '.join(context[1:]))[-(self.context_len - len(topic)):]
        context = torch.tensor(context)
        batch_context = []
        batch_response = []
        batch_sign = []
        for i in idd:
            i_response = self.tokenizer.encode(response[i], truncation=True, max_length=self.response_len)
            i_response = torch.tensor(i_response)
            batch_context.append(context)
            batch_response.append(i_response)
            this_sign = sign if i != 0 else 1
            batch_sign.append(torch.tensor([this_sign] * len(i_response)))
        return batch_context, batch_response, batch_sign

    def __len__(self):
        return sum(self.sample_size)

    @staticmethod
    def collate_fn(data):
        context, response, sign = zip(*data)
        context = sum(context, [])
        response = sum(response, [])
        sign = sum(sign, [])
        sign = pad_sequence(sign, batch_first=True, padding_value=0)
        context = pad_sequence(context, batch_first=True, padding_value=1)
        return {
            'input_ids': context,
            'attention_mask': context.ne(1),
            'labels': pad_sequence(response, batch_first=True, padding_value=-100),
            'sign': sign
        }


class SpanData(BatchData):
    def span_mix(self, text, sub, neg_subs, type_matter=True):
        if len(sub) == 0 or len(neg_subs) == 0:
            return [text], [1]
        neg_subs = [s[1] for s in neg_subs if s[1] not in text]
        sub = [s for s in sub if not text.endswith(s[1])]

        neg_type = [s[0] for s in neg_subs]
        common_type = [s[0] for s in sub if s[0] in neg_type]
        if type_matter:
            if len(common_type) == 0:
                return [text], [1]
            chosen_type = np.random.choice(common_type)
            chosen_sub = np.random.choice([s[1] for s in sub if s[0] == chosen_type])
            chosen_neg_sub = np.random.choice([s[1] for s in neg_subs if s[0] == chosen_type])
        else:
            chosen_sub = np.random.choice([s[1] for s in sub])
            chosen_neg_sub = np.random.choice([s[1] for s in neg_subs])
        text = text.split(chosen_sub)
        text = [text[0], chosen_neg_sub, chosen_sub.join(text[1:])]
        sign = [1, -1, 1]
        text, sign = zip(*[[a, b] for a, b in zip(text, sign) if len(a) > 0])
        return list(text), list(sign)

    def __getitem__(self, index):
        corpus_id, index = choice_index(index, self.sample_size)
        rand = np.random.rand()
        index = rounder((index + rand) / self.sample_size[corpus_id] * self.data_size[corpus_id])
        index = min(index, len(self.context[corpus_id]) - 1)
        context = self.context[corpus_id][index]
        response = self.response[corpus_id][index]
        if not isinstance(response, list):
            context = [context]
            response = [response]
        context = context[0]
        topic = self.tokenizer.encode(context[0])
        context = topic + self.tokenizer.encode(' '.join(context[1:]))[-(self.context_len - len(topic)):]
        context = torch.tensor(context)

        if isinstance(response[0], list):
            text = response[0][0]
        else:
            text = response[0]
        this_response = self.tokenizer.encode(text, truncation=True, max_length=self.response_len)
        sign = [1] * len(this_response)

        batch_context = [context]
        batch_response = [torch.tensor(this_response)]
        batch_sign = [torch.tensor(sign)]

        if len(response) > 1:
            pos_text = response[0][0]
            pos_sub = response[0][1]
            neg_subs = sum([ss[1] for ss in response[1:]], [])

            text, sign = self.span_mix(pos_text, pos_sub, neg_subs, type_matter=True)

            sequence = [self.tokenizer.encode(s, add_special_tokens=False) for s in text]
            sign = [[s] * len(seq) for seq, s in zip(sequence, sign)]
            sequence = [0] + sum(sequence, []) + [2]
            sign = [1] + sum(sign, []) + [1]
            sequence = sequence[:self.response_len]
            sign = sign[:self.response_len]
            this_response = sequence
            sign = sign

            batch_context.append(context)
            batch_response.append(torch.tensor(this_response))
            batch_sign.append(torch.tensor(sign))
        return batch_context, batch_response, batch_sign


class EntDara(BatchData):
    def ent_mix(self, text1, ent1, ent2, ratio=1.0, type_matter=True):
        ent2 = [x for x in ent2 if x[1] not in text1]
        if type_matter:
            type2 = [s[0] for s in ent2]
            ent1 = [s for s in ent1 if s[0] in type2]
        num = max(int(len(ent1) * ratio), 1)
        np.random.shuffle(ent1)
        text1 = [text1]
        sign = [1]
        for ent in ent1[:num]:
            new_text = []
            new_sign = []
            for seg, s in zip(text1, sign):
                if s < 0 or ent[1] not in seg:
                    new_text.append(seg)
                    new_sign.append(s)
                    continue
                seg = seg.split(ent[1])
                np.random.shuffle(ent2)
                if type_matter:
                    replace = [x for x in ent2 if x[0] == ent[0] and x[1] != ent[1]][0][1]
                else:
                    replace = [x for x in ent2 if x[1] != ent[1]][0][1]
                seg = [seg[0], replace, ent[1].join(seg[1:])]
                new_text.extend(seg)
                new_sign.extend([1, -1, 1])
            text1 = new_text
            sign = new_sign
        text1, sign = zip(*[[a, b] for a, b in zip(text1, sign) if len(a) > 0])
        return list(text1), list(sign)

    def __getitem__(self, index):
        corpus_id, index = choice_index(index, self.sample_size)
        rand = np.random.rand()
        index = rounder((index + rand) / self.sample_size[corpus_id] * self.data_size[corpus_id])
        index = min(index, len(self.context[corpus_id]) - 1)
        context = self.context[corpus_id][index]
        response = self.response[corpus_id][index]
        if not isinstance(response, list):
            context = [context]
            response = [response]
        context = context[0]
        topic = self.tokenizer.encode(context[0])
        context = topic + self.tokenizer.encode(' '.join(context[1:]))[-(self.context_len - len(topic)):]
        context = torch.tensor(context)

        if isinstance(response[0], list):
            text = response[0][0]
        else:
            text = response[0]
        this_response = self.tokenizer.encode(text, truncation=True, max_length=self.response_len)
        sign = [1] * len(this_response)

        batch_context = [context]
        batch_response = [torch.tensor(this_response)]
        batch_sign = [torch.tensor(sign)]

        if len(response) > 1:
            pos_text = response[0][0]
            pos_sub = response[0][1]

            neg_subs = sum([ss[1] for ss in response[1:]], [])

            text, sign = self.ent_mix(pos_text, pos_sub, neg_subs, type_matter=True, ratio=0.5)

            sequence = [self.tokenizer.encode(s, add_special_tokens=False) for s in text]
            sign = [[s] * len(seq) for seq, s in zip(sequence, sign)]
            sequence = [0] + sum(sequence, []) + [2]
            sign = [1] + sum(sign, []) + [1]
            sequence = sequence[:self.response_len]
            sign = sign[:self.response_len]
            this_response = sequence

            batch_context.append(context)
            batch_response.append(torch.tensor(this_response))
            batch_sign.append(torch.tensor(sign))

        return batch_context, batch_response, batch_sign


class AllData(BatchData):
    def __init__(self, *args, data_type=None, neg_memory=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_type = data_type
        self.neg_memory = neg_memory

    def ent_mix(self, text1, ent1, ent2, ratio=1.0, type_matter=True):
        ent2 = [x for x in ent2 if x[1] not in text1]
        if type_matter:
            type2 = [s[0] for s in ent2]
            ent1 = [s for s in ent1 if s[0] in type2]
        num = max(int(len(ent1) * ratio), 1)
        np.random.shuffle(ent1)
        text1 = [text1]
        sign = [1]
        for ent in ent1[:num]:
            new_text = []
            new_sign = []
            for seg, s in zip(text1, sign):
                if s < 0 or ent[1] not in seg:
                    new_text.append(seg)
                    new_sign.append(s)
                    continue
                seg = seg.split(ent[1])
                np.random.shuffle(ent2)
                if type_matter:
                    replace = [x for x in ent2 if x[0] == ent[0] and x[1] != ent[1]][0][1]
                else:
                    replace = [x for x in ent2 if x[1] != ent[1]][0][1]
                seg = [seg[0], replace, ent[1].join(seg[1:])]
                new_text.extend(seg)
                new_sign.extend([1, -1, 1])
            text1 = new_text
            sign = new_sign
        text1, sign = zip(*[[a, b] for a, b in zip(text1, sign) if len(a) > 0])
        return list(text1), list(sign)

    def span_mix(self, text, sub, neg_subs, type_matter=True):
        if len(sub) == 0 or len(neg_subs) == 0:
            return [text], [1]
        neg_subs = [s[1] for s in neg_subs if s[1] not in text]
        sub = [s for s in sub if not text.endswith(s[1])]

        neg_type = [s[0] for s in neg_subs]
        common_type = [s[0] for s in sub if s[0] in neg_type]
        if type_matter:
            if len(common_type) == 0:
                return [text], [1]
            chosen_type = np.random.choice(common_type)
            chosen_sub = np.random.choice([s[1] for s in sub if s[0] == chosen_type])
            chosen_neg_sub = np.random.choice([s[1] for s in neg_subs if s[0] == chosen_type])
        else:
            chosen_sub = np.random.choice([s[1] for s in sub])
            chosen_neg_sub = np.random.choice([s[1] for s in neg_subs])
        text = text.split(chosen_sub)
        text = [text[0], chosen_neg_sub, chosen_sub.join(text[1:])]
        sign = [1, -1, 1]
        text, sign = zip(*[[a, b] for a, b in zip(text, sign) if len(a) > 0])
        return list(text), list(sign)

    def __getitem__(self, index):
        corpus_id, index = choice_index(index, self.sample_size)
        rand = np.random.rand()
        index = rounder((index + rand) / self.sample_size[corpus_id] * self.data_size[corpus_id])
        index = min(index, len(self.context[corpus_id]) - 1)
        context = self.context[corpus_id][index]
        response = self.response[corpus_id][index]
        data_type = self.data_type[corpus_id]
        if not isinstance(response, list):
            context = [context]
            response = [response]
        context = context[0]
        topic = self.tokenizer.encode(context[0])
        context = topic + self.tokenizer.encode(' '.join(context[1:]))[-(self.context_len - len(topic)):]
        context = torch.tensor(context)

        if isinstance(response[0], list):
            text = response[0][0]
        else:
            text = response[0]
        this_response = self.tokenizer.encode(text, truncation=True, max_length=self.response_len)
        sign = [1] * len(this_response)
        batch_context, batch_response, batch_sign = [], [], []

        if 'x' in data_type:
            batch_context.append(context)
            batch_response.append(torch.tensor(this_response))
            batch_sign.append(torch.tensor(sign))

        if 'cl' in data_type:
            pos_text = response[0][0]
            pos_sub = response[0][1]
            neg_subs = sum([ss[1] for ss in response[1:]], [])

            if 'ent' in data_type or 'ner' in data_type:
                text, sign = self.ent_mix(pos_text, pos_sub, neg_subs, type_matter=True, ratio=0.5)
            if 'span' in data_type:
                text, sign = self.span_mix(pos_text, pos_sub, neg_subs, type_matter=True)

            sequence = [self.tokenizer.encode(s, add_special_tokens=False) for s in text]
            sign = [[s] * len(seq) for seq, s in zip(sequence, sign)]
            sequence = [0] + sum(sequence, []) + [2]
            sign = [1] + sum(sign, []) + [1]
            sequence = sequence[:self.response_len]
            sign = sign[:self.response_len]
            this_response = sequence

            batch_context.append(context)
            batch_response.append(torch.tensor(this_response))
            batch_sign.append(torch.tensor(sign))

        return batch_context, batch_response, batch_sign


def produce_lc(model, lr=5e-5, lr_mult=0.99):
    layer_names = []
    for idx, (name, param) in enumerate(model.named_parameters()):
        layer_names.append(name)
    layer_names.reverse()
    parameters = []

    this_lr = lr
    for idx, name in enumerate(layer_names):
        if 'encoder.layers' in name:
            # print(f'{idx}: lr = {this_lr:.6f}, {name}')
            parameters += [{'params': [p for n, p in model.named_parameters() if n == name and p.requires_grad],
                            'lr': this_lr}]
            this_lr *= lr_mult
    this_lr = lr
    for idx, name in enumerate(layer_names):
        if 'decoder.layers' in name:
            # print(f'{idx}: lr = {this_lr:.6f}, {name}')
            parameters += [{'params': [p for n, p in model.named_parameters() if n == name and p.requires_grad],
                            'lr': this_lr}]
            this_lr *= lr_mult

    this_lr = lr
    for idx, name in enumerate(layer_names):
        if 'decoder.layers' not in name and 'encoder.layers' not in name:
            # print(f'{idx}: lr = {this_lr:.6f}, {name}')
            parameters += [{'params': [p for n, p in model.named_parameters() if n == name and p.requires_grad],
                            'lr': this_lr}]
    return parameters


def main():
    accelerator = Accelerator()
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    epochs = 10
    batch_size = 16
    model_name = 'facebook/bart-large'
    ckpt_name = 'bart-large-mixcl'
    print(model_name, ckpt_name)
    save_per_step = 5000
    tokenizer = BartTokenizer.from_pretrained(model_name)

    # topic = list(set([x['chosen_topic'] for x in json.load(open('dataset/wizard/train.json'))]))
    # topic_subset = topic[len(topic) // 2:]
    # json.dump(topic_subset, open('tmp/topic2.json', 'w'))

    dialog_context, dialog_response = load_dialog('dataset/wizard/train.json', title='d ')
    # reddit_context, reddit_response = load_reddit('dataset/reddit-v2.txt')
    wiki_context, wiki_response = load_wiki('dataset/wizard_wiki_full.json', title='k ')

    # neg_k_context, neg_k_response = load_neg_knowledge('dataset/wizard/train.json')
    # knowledge_context, knowledge_response = load_knowledge('dataset/wizard/train.json', title='w ')
    knowledge_context, knowledge_response = load_batch_knowledge('dataset/wizard/train.json')
    knowledge_span = load_batch_knowledge_span('dataset/wizard/train_knowledge_span_new.json')
    knowledge_ner = load_batch_knowledge_ner('dataset/wizard/train_knowledge_ner.json', knowledge_response)

    data_context = [dialog_context, knowledge_context, knowledge_context, wiki_context]
    data_response = [dialog_response, knowledge_span, knowledge_ner, wiki_response]
    data_type = ['x', 'span', 'ner', 'x']
    # ratio = [0.5, 0.4, 0.1]
    ratios = [[0.4, 0.15, 0.15, 0.3], [0.4, 0.15, 0.15, 0.3], [0.5, 0.15, 0.15, 0.2]] + [[0.5, 0.25, 0.25, 0.1]] * 10
    # signs = [1, 1, 1]
    # neg_num = [0, 0, 0]

    # data_context = [dialog_context]
    # data_response = [dialog_response]
    # ratio, signs, neg_num = [1.0], [1], [0]
    print(data_type, ratios)

    # neg_num_schedule = [[0, 8]] * 10

    dataset = AllData(data_context, data_response, ratio=ratios[0], tokenizer=tokenizer,
                      context_len=128, response_len=64, data_type=data_type)
    print('data:', len(dataset))
    data_loader = torch.utils.data.DataLoader(
        dataset, collate_fn=dataset.collate_fn, batch_size=batch_size, shuffle=True, num_workers=8)

    model = BartForConditionalGeneration.from_pretrained(model_name)

    # optimizer = AdamW(produce_lc(model, lr=5e-5, lr_mult=0.995))
    optimizer = AdamW(model.parameters(), lr=2e-5)

    model, optimizer, data_loader = accelerator.prepare(model, optimizer, data_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=1000, num_training_steps=epochs * len(data_loader))

    scheduler = accelerator.prepare(scheduler)

    print(tokenizer.decode(dataset[10][0][0]))
    print(tokenizer.decode(dataset[10][1][0]))

    steps = 0
    os.makedirs(f'ckpt/{ckpt_name}', exist_ok=True)
    for epoch in range(epochs):

        dataset = AllData(data_context, data_response, ratio=ratios[epoch], tokenizer=tokenizer,
                          context_len=128, response_len=64, data_type=data_type)
        print('data:', len(dataset))
        data_loader = torch.utils.data.DataLoader(
            dataset, collate_fn=dataset.collate_fn, batch_size=batch_size, shuffle=True, num_workers=8)
        data_loader = accelerator.prepare(data_loader)

        accelerator.wait_for_everyone()
        accelerator.print(f'train epoch={epoch}')
        tk0 = tqdm(data_loader, total=len(data_loader))
        losses = []
        for batch in tk0:
            sign = batch.pop('sign')
            # sign = (sign.unsqueeze(1) * torch.ones_like(batch['labels'])).view(-1)
            sign = sign.view(-1)

            labels = batch['labels'].view(-1)
            output = model(**batch)

            logits = output.logits
            logits = logits.view(-1, logits.size(-1))
            pos_loss = F.cross_entropy(logits, labels, reduction='none')
            neg_loss = F.nll_loss(torch.log(torch.clamp((1.0 - F.softmax(logits)), min=1e-5)), labels, reduction='none')
            pos_loss, neg_loss = (pos_loss * (sign > 0).float()).sum(), (neg_loss * (sign < 0).float()).sum()
            pos_tok_num = max(((sign > 0) & (labels != -100)).float().sum(), 1.0)
            neg_tok_num = max(((sign < 0) & (labels != -100)).float().sum(), 1.0)
            loss = pos_loss / pos_tok_num + neg_loss / neg_tok_num

            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            losses.append(loss.item())
            tk0.set_postfix(loss=sum(losses) / len(losses))

            steps += 1

            if steps % save_per_step == 0:
                if accelerator.is_local_main_process:
                    accelerator.print(f'Save at ckpt/{ckpt_name}/step.{steps}.pt')
                    accelerator.save(accelerator.unwrap_model(model).state_dict(),
                                     f'ckpt/{ckpt_name}/step.{steps}.pt')

        if accelerator.is_local_main_process:
            accelerator.print(f'Save at ckpt/{ckpt_name}/{epoch}.pt')
            accelerator.save(accelerator.unwrap_model(model).state_dict(), f'ckpt/{ckpt_name}/{epoch}.pt')


def lower(text):
    if isinstance(text, str):
        text = text.strip().lower()
        text = ' '.join(nltk.word_tokenize(text))
        return text.strip()
    return [lower(item) for item in text]


def test():
    from driver.train_bart import Data
    batch_size = 32
    model_name = 'facebook/bart-large'
    ckpt_name = 'bart-large-mixcl'
    data_name = 'wizard/seen'

    print(model_name, ckpt_name, data_name)

    tokenizer = BartTokenizer.from_pretrained(model_name)

    # topic_subset = json.load(open('tmp/topic1.json'))

    context, response = load_dialog(f'dataset/{data_name}.json')
    dataset = Data(context, response, tokenizer, context_len=128, response_len=64)
    print(data_name, len(dataset))
    data_loader = torch.utils.data.DataLoader(
        dataset, collate_fn=dataset.collate_fn, batch_size=batch_size, shuffle=False, num_workers=8)

    model = BartForConditionalGeneration.from_pretrained(model_name)
    # tokenizer.add_tokens(['<topic>', '<wizard>', '<apprentice>'])
    # model.resize_token_embeddings(len(tokenizer))
    model = model.cuda()
    # cache = None
    cache = eval_acc(None, json.load(open(f'dataset/{data_name}.json')), compute_cache=True)
    for epoch in range(100):
        if not os.path.exists(f'ckpt/{ckpt_name}/{epoch}.pt'):
            continue
        print(f'Test ckpt/{ckpt_name}/{epoch}.pt')
        model.load_state_dict(torch.load(f'ckpt/{ckpt_name}/{epoch}.pt'))
        tk0 = tqdm(data_loader, total=len(data_loader))
        # output_text_collect = read
        output_text_collect = []
        for batch in tk0:
            # batch = {k: v.cuda() for k, v in batch.items()}
            output = model.generate(
                input_ids=batch['input_ids'].cuda(),
                attention_mask=batch['attention_mask'].cuda(),
                max_length=128,
                # num_beams=3,
            )
            output_text = tokenizer.batch_decode(output, skip_special_tokens=True)
            output_text_collect.extend(output_text)

        true_text_collect = dataset.response
        print(eval_all(lower(output_text_collect), lower(true_text_collect)))
        print(eval_acc(output_text_collect, None, cache=cache))
        # print(eval_em(output_text_collect, json.load(open(f'dataset/{data_name}.json'))))
        write_file(output_text_collect, f'ckpt/{ckpt_name}/u{epoch}.txt')


def clean():
    def replace(text):
        return text.replace('-LRB-', '(').replace('-RRB-', ')')

    data = json.load(open('dataset/wizard/train_knowledge_span.json'))
    new_data = []
    for line in data:
        content = line[1]
        content = [[replace(s[0]), [[t[0], replace(t[1])] for t in s[1]]] for s in content]
        new_line = [line[0], content]
        new_data.append(new_line)
    json.dump(new_data, open('dataset/wizard/train_knowledge_span_new.json', 'w'))


if __name__ == '__main__':
    # Train
    main()
    # Test
    test()
    # clean()
