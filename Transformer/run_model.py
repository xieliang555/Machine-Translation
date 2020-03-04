'''
	Transformer for neuron machine translation
	Ref: https://andrewpeng.dev/transformer-pytorch/
	Ref: https://spaces.ac.cn/archives/6933
	Ref: https://github.com/graykode/nlp-tutorial

'''

import torch
from torchtext.datasets import Multi30k
from torchtext.data import BucketIterator, Field
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import os
import time
from torchsummaryX import summary

import model


SRC = Field(tokenize = 'spacy',
			tokenizer_language = 'de',
			init_token = '<sos>',
			eos_token = '<eos>',
			lower = True)

TRG = Field(tokenize = 'spacy',
			tokenizer_language = 'en',
			init_token = '<sos>',
			eos_token = '<eos>',
			lower = True)

train_data, val_data, test_data = Multi30k.splits(exts = ('.de', '.en'), fields = (SRC, TRG))
train_iter, val_iter, test_iter = BucketIterator.splits((train_data, val_data, test_data), batch_size = 2)

print(len(train_iter))

SRC.build_vocab(train_data, min_freq = 2)
TRG.build_vocab(train_data, min_freq = 2)


D_MODEL = 32
N_HEAD = 1
NUM_ENC_LAYERS = 1
NUM_DEC_LAYERS = 1
DIM_FEEDWORD = 64
DROPOUT = 0.5
ACTIVATION = 'relu'
N_EPOCH = 10

net = model.Transformer(len(SRC.vocab), len(TRG.vocab), D_MODEL, N_HEAD, NUM_ENC_LAYERS, 
						NUM_ENC_LAYERS, DIM_FEEDWORD, DROPOUT, ACTIVATION)
optimizer = optim.Adam(net.parameters())
criterion = nn.CrossEntropyLoss(ignore_index = SRC.vocab.stoi['<pad>'])
writer = SummaryWriter(os.path.join('log/', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))


# print(net)
# summary(net, torch.zeros((10,1), dtype = torch.long), torch.zeros((10,1), dtype = torch.long))


for epoch in range(N_EPOCH):
	start_time = time.time()
	train_loss, train_bleu = model.train(net, train_iter, criterion, optimizer, TRG, epoch, writer)
	val_loss, val_bleu = model.evaluate(net, val_iter, criterion, TRG, epoch, writer)
	end_time = time.time()
	duration = int(end_time - start_time)
	print(f'epoch: {epoch} | duration: {duration // 60}m | end time: {time.strftime("%H:%M:%S", time.localtime(end_time))}')
	print(f'\ttrain loss: {train_loss:.3f} | train bleu: {train_bleu:.3f}')
	print(f'\tval loss: {val_loss:.3f} | val bleu: {val_bleu:.3f}\n')

test_loss, test_bleu = model.evaluate(net, test_iter, criterion, TRG, N_EPOCH+1, writer)
print(f'test loss: {test_loss:.3f} | test bleu: {test_bleu: .3f}')


