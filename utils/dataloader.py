import json
import sys
from functools import partial

import torch
from torch.utils.data import DataLoader, IterableDataset
from torchtext.data import to_map_style_dataset
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import WikiText2, WikiText103
from torchtext.vocab import build_vocab_from_iterator
from transformers import GPT2Tokenizer
from collections import OrderedDict
from torchtext.vocab import vocab as pytorch_vocab


sys.path.insert(0, '/home/ruian/sw/ranking-engine/')
from src.models.feature_models.skill_model.skill_w2v_model import \
    SkillW2vTokenizer
    

from utils.constants import (CBOW_N_WORDS, MAX_SEQUENCE_LENGTH,
                             MIN_WORD_FREQUENCY, SKIPGRAM_N_WORDS)


def get_english_tokenizer(tokenizer_type):
    """
    Documentation:
    https://pytorch.org/text/stable/_modules/torchtext/data/utils.html#get_tokenizer
    """
    if tokenizer_type == 'gpt2':
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    elif tokenizer_type == 'english':
        tokenizer = get_tokenizer("basic_english", language="en")
    else:
        tokenizer = SkillW2vTokenizer().tokenize
    
    return tokenizer


class ProfileIterableDataset(IterableDataset):
    def __init__(self, filename):

        #Store the filename in object's memory
        self.tokenizer = SkillW2vTokenizer().tokenize
        self.text_pipeline = lambda x: SkillW2vTokenizer().convert_tokens_to_ids(self.tokenizer(x))
        self.filename = filename
        
        #And that's it, we no longer need to store the contents in the memory

    def line_mapper(self, line):
        
        #Splits the line into text and label and applies preprocessing to the text
        profile = json.loads(line)
        handson = profile[list(profile.keys())[0]]['handson']
        handson_skill = [each['handson_skill'] for each in handson]
        skill_sentence = " ".join(handson_skill)

        return skill_sentence

    def __iter__(self):
        #Create an iterator
        file_itr = open(self.filename)
        #Map each element using the line_mapper
        mapped_itr = map(self.line_mapper, file_itr)
        
        return mapped_itr

def get_data_iterator(ds_name, ds_type, data_dir, base):
    if ds_name == "WikiText2":
        data_iter = WikiText2(root=data_dir, split=(ds_type))
    elif ds_name == "WikiText103":
        data_iter = WikiText103(root=data_dir, split=(ds_type))
    elif ds_name == "profiles":
        if  ds_type == 'train':
            if base == "test":
                return ProfileIterableDataset('/home/ruian/mnt/ruian/profiles/jd-profile-small/profiles_gpt3_train_small.json')
            elif base == "prod":
                return ProfileIterableDataset('/home/ruian/mnt/ruian/profiles/jd-profile-small/profiles_gpt3_train.json')
        if ds_type == "valid":
            if base == "test":
                return ProfileIterableDataset('/home/ruian/mnt/ruian/profiles/jd-profile-small/profiles_gpt3_eval_small.json')
            elif base == "prod":
                return ProfileIterableDataset('/home/ruian/mnt/ruian/profiles/jd-profile-small/profiles_gpt3_eval.json')
    else:
        raise ValueError("Choose dataset from: WikiText2, WikiText103")
    data_iter = to_map_style_dataset(data_iter)
    return data_iter


def build_vocab(data_iter, tokenizer):
    """Builds vocabulary from iterator"""
    
    vocab = build_vocab_from_iterator(
        map(tokenizer, data_iter),
        specials=["<unk>"],
        min_freq=MIN_WORD_FREQUENCY,
    )
    vocab.set_default_index(vocab["<unk>"])
    return vocab


def collate_cbow(batch, text_pipeline):
    """
    Collate_fn for CBOW model to be used with Dataloader.
    `batch` is expected to be list of text paragrahs.
    
    Context is represented as N=CBOW_N_WORDS past words 
    and N=CBOW_N_WORDS future words.
    
    Long paragraphs will be truncated to contain
    no more that MAX_SEQUENCE_LENGTH tokens.
    
    Each element in `batch_input` is N=CBOW_N_WORDS*2 context words.
    Each element in `batch_output` is a middle word.
    """
    batch_input, batch_output = [], []
    for text in batch:
        text_tokens_ids = text_pipeline(text)

        if len(text_tokens_ids) < CBOW_N_WORDS * 2 + 1:
            continue

        if MAX_SEQUENCE_LENGTH:
            text_tokens_ids = text_tokens_ids[:MAX_SEQUENCE_LENGTH]

        for idx in range(len(text_tokens_ids) - CBOW_N_WORDS * 2):
            token_id_sequence = text_tokens_ids[idx : (idx + CBOW_N_WORDS * 2 + 1)]
            output = token_id_sequence.pop(CBOW_N_WORDS)
            input_ = token_id_sequence
            batch_input.append(input_)
            batch_output.append(output)

    batch_input = torch.tensor(batch_input, dtype=torch.long)
    batch_output = torch.tensor(batch_output, dtype=torch.long)
    return batch_input, batch_output


def collate_skipgram(batch, text_pipeline):
    """
    Collate_fn for Skip-Gram model to be used with Dataloader.
    `batch` is expected to be list of text paragrahs.
    
    Context is represented as N=SKIPGRAM_N_WORDS past words 
    and N=SKIPGRAM_N_WORDS future words.
    
    Long paragraphs will be truncated to contain
    no more that MAX_SEQUENCE_LENGTH tokens.
    
    Each element in `batch_input` is a middle word.
    Each element in `batch_output` is a context word.
    """
    batch_input, batch_output = [], []
    for text in batch:
        text_tokens_ids = text_pipeline(text)

        if len(text_tokens_ids) < SKIPGRAM_N_WORDS * 2 + 1:
            continue

        if MAX_SEQUENCE_LENGTH:
            text_tokens_ids = text_tokens_ids[:MAX_SEQUENCE_LENGTH]

        for idx in range(len(text_tokens_ids) - SKIPGRAM_N_WORDS * 2):
            token_id_sequence = text_tokens_ids[idx : (idx + SKIPGRAM_N_WORDS * 2 + 1)]
            input_ = token_id_sequence.pop(SKIPGRAM_N_WORDS)
            outputs = token_id_sequence

            for output in outputs:
                batch_input.append(input_)
                batch_output.append(output)

    batch_input = torch.tensor(batch_input, dtype=torch.long)
    batch_output = torch.tensor(batch_output, dtype=torch.long)
    return batch_input, batch_output


def get_dataloader_and_vocab(
    model_name, ds_name, ds_type, data_dir, batch_size, shuffle, vocab=None, tokenizer_type='english', base='test'
):

    data_iter = get_data_iterator(ds_name, ds_type, data_dir, base)
    tokenizer = get_english_tokenizer(tokenizer_type)

    if hasattr(tokenizer, 'name_or_path'):
        if tokenizer.name_or_path == 'gpt2':
        # gpt2 
            text_pipeline = lambda x: tokenizer(x).get('input_ids', [])
            vocab = tokenizer.get_vocab()
    elif tokenizer_type == "ranking_engine":
        text_pipeline = lambda x: SkillW2vTokenizer().convert_tokens_to_ids(tokenizer(x))
        skills = SkillW2vTokenizer()._w2v.key_to_index

        skills_cnt = {each:1 for each in skills.keys()}
        ordered_dict = OrderedDict(skills_cnt)
        vocab =  pytorch_vocab(ordered_dict)
        
        print("loading SkillW2vTokenizer tokenizer from ranking engine", type(vocab))
    else:
        text_pipeline = lambda x: vocab(tokenizer(x))

    if not vocab:
        vocab = build_vocab(data_iter, tokenizer)

    if model_name == "cbow":
        collate_fn = collate_cbow
    elif model_name == "skipgram":
        collate_fn = collate_skipgram
    else:
        raise ValueError("Choose model from: cbow, skipgram")

    dataloader = DataLoader(
        data_iter,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=partial(collate_fn, text_pipeline=text_pipeline),
    )
    return dataloader, vocab
    