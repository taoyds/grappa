''' Convenience functions for handling BERT '''
import torch
import sys
import pdb

from torch.autograd import Variable

import transformers
import numpy as np
import torch.nn.functional as F

from tqdm import tqdm

def load_model(version='bert-base-uncased'):
    ''' Load BERT model and corresponding tokenizer '''
    tokenizer = transformers.BertTokenizer.from_pretrained(version)
    model = transformers.BertModel.from_pretrained(version)
    model.eval()
    model.to('cuda')

    return model, tokenizer

def flatten(list_of_lists):
    for list in list_of_lists:
        for item in list:
            yield item

def convert_tokens_to_ids(model, tokenizer, tokens, pad=True):
    max_len = model.embeddings.position_embeddings.weight.size(0)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    ids = torch.tensor([token_ids])
    assert ids.size(1) < max_len
    if pad:
        padded_ids = torch.zeros(1, max_len).to(ids)
        padded_ids[0, :ids.size(1)] = ids
        mask = torch.zeros(1, max_len).to(ids)
        mask[0, :ids.size(1)] = 1
        return padded_ids, mask
    else:
        return ids

def subword_tokenize(tokenizer, tokens):
    subwords = list(map(tokenizer.tokenize, tokens))
    subword_lengths = list(map(len, subwords))
    subwords = ["[CLS]"] + list(flatten(subwords)) + ["[SEP]"]
    token_start_idxs = 1 + np.cumsum([0] + subword_lengths[:-1])
    return subwords, token_start_idxs

def subword_tokenize_to_ids(model, tokenizer, tokens):
    max_len = model.embeddings.position_embeddings.weight.size(0)
    subwords, token_start_idxs = subword_tokenize(tokenizer, tokens)
    subword_ids, mask = convert_tokens_to_ids(model, tokenizer, subwords)
    token_starts = torch.zeros(1, max_len).to(subword_ids)
    token_starts[0, token_start_idxs] = 1
    return subword_ids, mask, token_starts

def encode_batch(model, tokenizer, index2word, input_variable, input_length, batch_size, hidden_size, mode):
    # retrieve text
    with torch.no_grad():
        if isinstance(input_variable, str):
            texts = [input_variable]
        else:
            indices = [input_variable.cpu().numpy()[:,i] for i in range(batch_size)]
            texts = [" ".join([index2word[idx] for idx in index]) for index in indices]

        max_length = model.embeddings.position_embeddings.weight.size(0)
        batch_input_ids = torch.LongTensor(batch_size, max_length).to('cuda')
        batch_token_type_ids = torch.LongTensor(batch_size, max_length).to('cuda')
        batch_starts = []

        batch_max_subword_length = 0
        mask_lengths = []

        for i, text in enumerate(texts):
            tokens = text.split()
            ids, mask, token_starts = subword_tokenize_to_ids(model, tokenizer, tokens)
            mask_length = (mask != 0).max(0)[0].nonzero()[-1].item()
            assert(mask_length < max_length)
            if mask_length - 1 > batch_max_subword_length:
                batch_max_subword_length = mask_length - 1
            mask_lengths.append(mask_length)
            batch_input_ids[i] = ids
            batch_token_type_ids[i] = torch.zeros_like(mask)
            batch_starts.append(token_starts)

        assert(batch_input_ids.shape[0] == batch_size)
        assert(batch_input_ids.shape[1] == max_length)

        last_layer = model(batch_input_ids, batch_token_type_ids)[0]
        assert(last_layer.shape[0] == batch_size)
        assert(last_layer.shape[1] == max_length)
        assert(last_layer.shape[2] == hidden_size)

        if mode == "first-subword":
            encodings = Variable(torch.zeros(batch_size, input_length, hidden_size)).to('cuda')
        elif mode == "all-subword":
            encodings = Variable(torch.zeros(batch_size, batch_max_subword_length, hidden_size)).to('cuda')
        for i, _ in enumerate(texts):
            if mode == "first-subword":
                token_reprs = last_layer[i][batch_starts[i].nonzero()[:,1]]
            elif mode == "all-subword":
                token_reprs = last_layer[i][1:mask_lengths[i]]
                token_reprs = F.pad(input=token_reprs, pad=(0, 0, 0, batch_max_subword_length-(mask_lengths[i] - 1)), mode='constant', value=0)
            encodings[i] = token_reprs

    return encodings

def encode_batches(model, tokenizer, batches, index2word, batch_size, hidden_size, max_length, mode):
    # if mode == "all-subword":
    #     for i in tqdm(range(len(batches))):
    #         input_variable = batches[i][0]
    #         tokens = input_variable.split()
    #         subwords = list(flatten(list(map(tokenizer.tokenize, tokens))))
    #         if len(subwords) > max_length:
    #             max_length = len(subwords)
    #     print(f"max_length: {max_length}")
    batch_encodings = Variable(torch.zeros(len(batches), batch_size, max_length, hidden_size)).to('cuda')
    for i in tqdm(range(len(batches))):
        input_variable = batches[i][0]
        if isinstance(input_variable, str):
            input_length = len(input_variable.split())
        else:
            input_length = input_variable.size()[0]
        encoding = encode_batch(model, tokenizer, index2word, input_variable, input_length, batch_size, hidden_size, mode)
        import pdb; pdb.set_trace()
        batch_encodings[i] = F.pad(input=encoding, pad=(0,0,0,max_length-input_length), mode='constant', value=0)
    return batch_encodings
