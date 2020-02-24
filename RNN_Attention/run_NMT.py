# The code follows the architecture of 
# "NEURAL MACHINE TRANSLATION BY JOINTLY LEARNING TO ALIGN AND TRANSLATE"
# which is Bidirectional GRU encoder + unidirectional GRU decoder + Bahdanau attention


import torch
from torchtext.data import Field, BucketIterator
from torchtext.datasets import Multi30k
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import time 
import os
from torchsummaryX import summary

import NMT
import utils


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# field is the object for tokenize, padding and numericalize. 
# The arguments define the tokenizer, language, and padding information
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


train_data, dev_data, test_data = Multi30k.splits(exts= ('.de', '.en'),
	fields = (SRC, TRG), root = '.data', train = 'train', 
	validation = 'val', test = 'test2016') 


SRC.build_vocab(train_data, min_freq = 2)
TRG.build_vocab(train_data, min_freq = 2)


train_ietrator, valid_iterator, test_iterator = BucketIterator.splits(
	(train_data, dev_data, test_data),
	batch_size = 2,
	device = device)


# for i, batch in enumerate(train_ietrator):
# 	x = batch.trg
# 	x = x.permute(1,0)
# 	print(x)
# 	if i==0:
# 		break


INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
ENC_EMB_DIM = 32
DEC_EMB_DIM = 32
ENC_HID_DIM = 64
DEC_HID_DIM = 64
ATTN_DIM = 8
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5
N_EPOCHS = 10
CLIP = 1


enc = NMT.Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
atten = NMT.Attention(ENC_HID_DIM, DEC_HID_DIM, ATTN_DIM)
dec = NMT.Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, atten)
model = NMT.Seq2Seq(enc, dec, device).to(device)
model.apply(utils.init_weights)
optimizer = optim.Adam(model.parameters())
PAD_IDX = TRG.vocab.stoi['<pad>']
criterion = nn.CrossEntropyLoss(ignore_index = PAD_IDX)
writer = SummaryWriter(os.path.join('log/', time.strftime("%Y-%m-%d %H:%M:%S" ,time.localtime(time.time()))))


# print(f'the model has {utils.count_parameters(model):,} trainable parameters\n')
# print(model)
# summary(model, torch.zeros((10,1), dtype = torch.long), torch.zeros((10,1), dtype = torch.long))

# utils.vis_samples(train_ietrator, writer, SRC, TRG)


for epoch in range(N_EPOCHS):

	start_time = time.time()

	train_loss, train_bleu = NMT.train(model, criterion, train_ietrator, optimizer, CLIP, epoch, TRG, writer)
	evaluate_loss, evaluate_bleu = NMT.evaluate(model, criterion, valid_iterator, epoch, TRG, writer)

	end_time = time.time()
	end_time_format = time.strftime("%H:%M:%S" ,time.localtime(end_time))
	epoch_mins, epoch_secs = utils.epoch_time(start_time, end_time)

	print(f'Epoch: {epoch+1:02} | Epoch duration: {epoch_mins}m {epoch_secs}s | Epoch end time: {end_time_format}')
	print(f'\tTrain loss: {train_loss:.3f} | Train bleu: {train_bleu:.3f}')
	print(f'\tEvaluate loss: {evaluate_loss:.3f} | Evaluate bleu: {evaluate_bleu:.3f}\n')


test_loss, test_bleu = NMT.evaluate(model, criterion, test_iterator, N_EPOCHS+1, TRG, writer)
print(f'Test loss: {test_loss:.3f} | Test bleu: {test_bleu:.3f}')





