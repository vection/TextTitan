from torchtext.data.utils import (
    get_tokenizer,
    ngrams_iterator,
)
import torch
from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator
import pickle


class Basic_tokenizer:
    def __init__(self, train_dataset=None, min_freq=1):
        self.min = min_freq
        if train_dataset:
            self.create_basic_tokenizer(train_dataset)

    def create_basic_tokenizer(self, train_dataset):
        self.tokenizer = get_tokenizer("basic_english")
        ngrams = 1

        def yield_tokens(data_iter, ngrams):
            for text in data_iter:
                yield iter(self.tokenizer(text))

        self.vocab = build_vocab_from_iterator(yield_tokens(train_dataset.text, ngrams), specials=["<unk>", "<pad>"],
                                               min_freq=self.min)
        self.vocab.set_default_index(self.vocab["<unk>"])

    def tokenize(self, text, max_length=None, return_tensors='pt', **kwargs):
        tokenized_text = []
        for x in text:
            res = self.vocab(list((self.tokenizer(x))))
            if max_length and len(res) < max_length:
                chars_to_add = int(max_length - len(res))
                for i in range(chars_to_add):
                    res.append(self.vocab['<pad>'])
            else:
                res = res[:max_length]
            if return_tensors == 'pt':
                res = torch.tensor(res)
            tokenized_text.append(res)
        if return_tensors == 'pt':
            tokenized_text = torch.stack(tokenized_text)

        return {'input_ids': tokenized_text}

    def save(self, save_path):
        output = open(save_path, 'wb')
        pickle.dump(self.vocab, output)
        output.close()
        print("Vocab saved")

    @staticmethod
    def load(path):
        output = open(path, 'rb')
        vocab = pickle.load(output)
        temp = Basic_tokenizer()
        temp.vocab = vocab
        return temp


