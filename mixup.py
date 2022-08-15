import stanza
import spacy
import string
from nltk import word_tokenize
import numpy as np
from tqdm import tqdm
import json
from transformers import pipeline
import sys
import torch


class MixUp:
    def __init__(self):
        self.nlp = None
        self.ner = spacy.load("en_core_web_sm")
        self.bert = None
        self.cache = {}

    def get_spans(self, node, layer=0):
        if layer > 996:
            return '', []
        if node.is_leaf():
            return str(node), []
        res = [self.get_spans(child, layer=layer + 1) for child in node.children]
        head = [child[0] for child in res]
        spans = [child[1] for child in res]
        if node.label not in ['ROOT', 'S']:
            return ' '.join(head), [[node.label, ' '.join(head)]] + sum(spans, [])
        return ' '.join(head), sum(spans, [])

    def spanning(self, text):
        if self.nlp is None:
            self.nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency', verbose=False, use_gpu=True)
        text = ' '.join(text.split()[:64])
        key = text
        if key in self.cache:
            return self.cache[key]
        spans = [self.get_spans(sent.constituency, layer=0) for sent in self.nlp(text).sentences]
        sub = sum([sent[1] for sent in spans], [])
        sub = [s for s in sub if s[1] not in string.punctuation]
        text = ' '.join([sent[0] for sent in spans])
        length = len(text.split())
        sub = [s for s in sub if max(length * 0.1, 1) < len(s[1].split()) < length * 0.5]
        sub = [[s[0], ' ' + s[1]] if ' ' + s[1] in text else s for s in sub]
        self.cache[key] = (text, sub)
        torch.cuda.empty_cache()
        return text, sub

    def nering(self, text):
        return [[ent.label_, ' ' + ent.text if ' ' + ent.text in text else ent.text] for ent in self.ner(text).ents]

    def span_mix(self, text1, text2, type_matter=True, cached=False):
        if cached:
            text1, sub1 = text1
            sub2 = text2
        else:
            text1, sub1 = self.spanning(text1)
            text2, sub2 = self.spanning(text2)
        if len(sub1) == 0 or len(sub2) == 0:
            return [text1], [1]
        type2 = [s[0] for s in sub2]
        common_type = [s[0] for s in sub1 if s[0] in type2]
        if type_matter:
            if len(common_type) == 0:
                return [text1], [1]
            chosen_type = np.random.choice(common_type)
            chosen_sub1 = np.random.choice([s[1] for s in sub1 if s[0] == chosen_type])
            chosen_sub2 = np.random.choice([s[1] for s in sub2 if s[0] == chosen_type])
        else:
            chosen_sub1 = np.random.choice([s[1] for s in sub1])
            chosen_sub2 = np.random.choice([s[1] for s in sub2])
        text1 = text1.split(chosen_sub1)
        text1 = [text1[0], chosen_sub2, chosen_sub1.join(text1[1:])]
        sign = [1, -1, 1]
        text1, sign = zip(*[[a, b] for a, b in zip(text1, sign) if len(a) > 0])
        return list(text1), list(sign)

    def ent_mix(self, text1, text2, ratio=1, type_matter=False, cached=False):
        if cached:
            ent1 = text1
            ent2 = text2
        else:
            text1 = ' '.join(word_tokenize(text1))
            text2 = ' '.join(word_tokenize(text2))
            ent1 = self.nering(text1)
            ent2 = self.nering(text2)
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


def build_cache(data):
    model = MixUp()
    knowledge_cache = []
    for example in tqdm(data):
        if example['title'] == 'no_passages_used':
            continue
        collect = []
        text = example['title'] + ' ' + example['checked_sentence']
        collect.append(model.nering(text))
        for k in example['knowledge']:
            for s in example['knowledge'][k]:
                if k == example['title'] or s == example['checked_sentence']:
                    continue
                text = k + ' ' + s
                collect.append(model.nering(text))
        knowledge_cache.append(collect)
    return knowledge_cache


def main():
    data = json.load(open('dataset/wizard/train.json'))
    model = MixUp()
    knowledge_cache = []
    data = [[i, x] for i, x in enumerate(data)]
    data.sort(key=lambda x: x[1]['chosen_topic'])

    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--id', type=str, required=True)
    args = parser.parse_args()
    ids = int(args.id)
    processes = 8
    length = len(data) // processes + 1
    data = data[ids * length:(ids + 1) * length]

    for index, example in tqdm(data):
        if example['title'] == 'no_passages_used':
            continue
        collect = []
        text = example['title'] + ' ' + example['checked_sentence']
        collect.append(model.nering(text))
        for k in example['knowledge']:
            for s in example['knowledge'][k]:
                if k == example['title'] or s == example['checked_sentence']:
                    continue
                text = k + ' ' + s
                collect.append(model.nering(text))
        knowledge_cache.append([index, collect])
    # knowledge_cache = mp(build_cache, data, processes=1)
    json.dump(knowledge_cache, open(f'new_tmp/ner{ids}.json', 'w'))


def ner_main():
    import sys

    sys.path += ['./']
    from utils.mp import mp
    data = json.load(open('dataset/wizard/train.json'))
    knowledge_cache = mp(build_cache, data, processes=40)
    json.dump(knowledge_cache, open(f'new_tmp/ner.json', 'w'))


def test():
    model = MixUp()
    text = 'Science fiction (often shortened to SF or sci-fi) is a \"genre\" of speculative fiction, typically dealing with imaginative concepts such as futuristic science and technology, space travel, time travel, faster than light travel, parallel universes, and extraterrestrial life.'
    text = text + text
    print(model.spanning(text))


if __name__ == '__main__':
    # test()
    # main()
    ner_main()
