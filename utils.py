import re
import torch
from collections import Counter

SOS_token = 0
EOS_token = 1
PAD_token = 2

class Vocabulary:
    def __init__(self):
        self.word2index = {"<SOS>": SOS_token, "<EOS>": EOS_token, "<PAD>": PAD_token}
        self.index2word = {SOS_token: "<SOS>", EOS_token: "<EOS>", PAD_token: "<PAD>"}
        self.word2count = Counter()
        self.n_words = 3  # Count SOS, EOS, PAD

    def add_sentence(self, sentence):
        for word in self.tokenize(sentence):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1
        self.word2count[word] += 1

    @staticmethod
    def tokenize(sentence):
        return re.findall(r"\b\w+\b", sentence.lower())

    def sentence_to_indexes(self, sentence):
        return [self.word2index[word] for word in self.tokenize(sentence)]

    def indexes_to_sentence(self, indexes):
        return ' '.join([self.index2word[idx] for idx in indexes if idx not in [SOS_token, EOS_token, PAD_token]])

def read_qa_pairs(filename):
    pairs = []
    with open(filename, encoding='utf-8') as f:
        for line in f:
            q, a = line.strip().split('\t')
            pairs.append((q, a))
    return pairs

def prepare_data(filename):
    pairs = read_qa_pairs(filename)
    input_vocab = Vocabulary()
    output_vocab = Vocabulary()
    for q, a in pairs:
        input_vocab.add_sentence(q)
        output_vocab.add_sentence(a)
    return pairs, input_vocab, output_vocab

def tensor_from_sentence(vocab, sentence):
    indexes = vocab.sentence_to_indexes(sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long).view(-1, 1)

def tensors_from_pair(pair, input_vocab, output_vocab):
    input_tensor = tensor_from_sentence(input_vocab, pair[0])
    target_tensor = tensor_from_sentence(output_vocab, pair[1])
    return (input_tensor, target_tensor) 