"""
This file is based on:
    https://github.com/karpathy/nanoGPT/blob/master/data/shakespeare/prepare.py
    https://github.com/karpathy/ng-video-lecture/blob/master/bigram.py
"""
import os
import requests
import tiktoken
import numpy as np

dirname = os.path.dirname(__file__)
input_file_path = os.path.join(dirname, 'input.txt')


class SimpleEncoder:

    def __init__(self, text):
        chars = sorted(list(set(text)))
        self.n_vocab = len(chars)
        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }

    def encode(self, input):
        return [self.stoi[c] for c in input]

    def encode_ordinary(self, input):
        return self.encode(input)

    def decode(self, input):
        return ''.join([self.itos[i] for i in input])


class TinyShakespeare:

    def __init__(self, decoder='gpt2'):

        self.trn_bin_path = os.path.join(dirname, decoder+'.trn.bin')
        self.val_bin_path = os.path.join(dirname, decoder+'.val.bin')

        if not os.path.exists(input_file_path):
            self._download()

        if decoder == 'simple':
            with open(input_file_path, 'r') as f:
                data = f.read()
            self.enc = SimpleEncoder(data)
        else:
            self.enc = tiktoken.get_encoding(decoder)

        if not os.path.exists(self.trn_bin_path):
            self._encode()
            self._save_encoding()
        else:
            self.trn_ids = np.fromfile(self.trn_bin_path, dtype=np.uint16)
            self.val_ids = np.fromfile(self.val_bin_path, dtype=np.uint16)

    def _download(self):
        data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        with open(input_file_path, 'w') as f:
            f.write(requests.get(data_url).text)

    def _encode(self):

        # read input file
        with open(input_file_path, 'r') as f:
            data = f.read()
        n = len(data)
        trn_data = data[:int(n*0.9)]
        val_data = data[int(n*0.9):]

        # encode
        self.trn_ids = self.enc.encode_ordinary(trn_data)
        self.val_ids = self.enc.encode_ordinary(val_data)
        print(f"train has {len(self.trn_ids):,} tokens")
        print(f"val has {len(self.val_ids):,} tokens")

    def _save_encoding(self):
        trn_ids = np.array(self.trn_ids, dtype=np.uint16)
        val_ids = np.array(self.val_ids, dtype=np.uint16)
        trn_ids.tofile(self.trn_bin_path)
        val_ids.tofile(self.val_bin_path)

    def decode(self, input):
        return self.enc.decode(input)

    @property
    def n_vocab(self):
        return self.enc.n_vocab

    @property
    def data_trn(self):
        return self.trn_ids

    @property
    def data_val(self):
        return self.val_ids