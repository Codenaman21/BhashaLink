import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import spacy
from collections import Counter
import os


spacy_en = spacy.load("en_core_web_sm")

def tokenize_en(text):
    return [tok.text.lower() for tok in spacy_en.tokenizer(text)]

def tokenize_gloss(text):
    return text.lower().strip().split()

class Vocab:
    def __init__(self, counter, specials=['<unk>', '<pad>', '<bos>', '<eos>'], min_freq=1):
        self.itos = list(specials)
        self.stoi = {tok: idx for idx, tok in enumerate(self.itos)}

        for token, freq in counter.items():
            if freq >= min_freq and token not in self.stoi:
                self.stoi[token] = len(self.itos)
                self.itos.append(token)

    def __len__(self):
        return len(self.itos)

    def __getitem__(self, token):
        return self.stoi.get(token, self.stoi['<unk>'])

    def lookup_token(self, idx):
        return self.itos[idx] if idx < len(self.itos) else '<unk>'

class ISLGlossDataset(Dataset):
    def __init__(self, csv_file, src_vocab=None, tgt_vocab=None, min_freq=1):
        df = pd.read_csv(csv_file)
        self.pairs = [
            (row['English-sentence'], row['ISL-GLOSS'])
            for _, row in df.iterrows()
        ]

        if src_vocab is None or tgt_vocab is None:
            self.src_vocab = self.build_vocab([src for src, _ in self.pairs], tokenize_en, min_freq)
            self.tgt_vocab = self.build_vocab([tgt for _, tgt in self.pairs], tokenize_gloss, min_freq)
        else:
            self.src_vocab = src_vocab
            self.tgt_vocab = tgt_vocab

    def build_vocab(self, texts, tokenizer, min_freq):
        counter = Counter()
        for text in texts:
            counter.update(tokenizer(text))
        return Vocab(counter, min_freq=min_freq)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src, tgt = self.pairs[idx]
        src_tokens = tokenize_en(src)
        tgt_tokens = tokenize_gloss(tgt)

        src_tensor = torch.tensor(
            [self.src_vocab['<bos>']] +
            [self.src_vocab[token] for token in src_tokens] +
            [self.src_vocab['<eos>']],
            dtype=torch.long
        )

        tgt_tensor = torch.tensor(
            [self.tgt_vocab['<bos>']] +
            [self.tgt_vocab[token] for token in tgt_tokens] +
            [self.tgt_vocab['<eos>']],
            dtype=torch.long
        )

        return src_tensor, tgt_tensor

def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_lens = [len(seq) for seq in src_batch]
    tgt_lens = [len(seq) for seq in tgt_batch]

    src_padded = torch.nn.utils.rnn.pad_sequence(src_batch, padding_value=0, batch_first=True)
    tgt_padded = torch.nn.utils.rnn.pad_sequence(tgt_batch, padding_value=0, batch_first=True)

    return src_padded, tgt_padded, src_lens, tgt_lens


def tokenize_sentence(text):
    return [tok.text.lower() for tok in spacy_en.tokenizer(text)]


_dataset = ISLGlossDataset("data/isl-gloss-dataset.csv")
src_vocab = _dataset.src_vocab
tgt_vocab = _dataset.tgt_vocab

if __name__ == '__main__':
    dataset = ISLGlossDataset("data/isl-gloss-dataset.csv")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    for batch in dataloader:
        src, tgt, src_lens, tgt_lens = batch
        print("SRC:", src)
        print("TGT:", tgt)
        break
