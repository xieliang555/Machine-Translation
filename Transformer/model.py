import torch
import torch.nn as nn
import utils

import math


class Transformer(nn.Module):
	"""
		src_vocab_size: the source vocabulary size
		tgt_vocab_size: the target vocabulary size
		d_model: the embedding feature dimension
	"""
	def __init__(self, src_vocab_size, tgt_vocab_size, d_model = 512, 
				 nhead = 8, num_enc_layers = 6, num_dec_layers = 6, 
				 dim_feedforward = 2048, dropout = 0.1, activation = 'relu'):
		super(Transformer, self).__init__()
		self.d_model = d_model
		self.src_pad_mask = None
		self.tgt_pad_mask = None
		self.memory_pad_mask = None
		self.tgt_subsequent_mask = None

		self.embedding = nn.Embedding(src_vocab_size, d_model)
		self.pos_encoder = PostionalEncoding(d_model)
		self.dropout = nn.Dropout(dropout)
		self.transformer = nn.Transformer(d_model, nhead, num_enc_layers, 
										  num_dec_layers, dim_feedforward, 
										  dropout, activation)
		self.out = nn.Linear(d_model, tgt_vocab_size)


	def get_pad_mask(self, data):
		# the index of '<pad>' is 1
		mask = data.eq(1).transpose(0,1)
		mask = mask.masked_fill(mask == True, float('-inf')).masked_fill(mask == False, float(0.0))
		return mask


	def get_square_subsequent_mask(self, tgt):
		seq_len = tgt.size(0)
		mask = torch.triu(torch.ones(seq_len, seq_len), diagonal = 1)
		mask = mask.float().masked_fill(mask == 0, float(0.0)).masked_fill(mask == 1, float('-inf'))
		return mask


	def forward(self, src, tgt):
		if self.tgt_subsequent_mask is None or self.tgt_subsequent_mask.size(0) != len(tgt):
			self.tgt_subsequent_mask = self.get_square_subsequent_mask(tgt)
		if self.src_pad_mask is None or self.src_pad_mask.size(1) != len(src):
			self.src_pad_mask = self.get_pad_mask(src)
		if self.tgt_pad_mask is None or self.tgt_pad_mask.size(1) != len(tgt):
			self.tgt_pad_mask = self.get_pad_mask(tgt)
		if self.memory_pad_mask is None or self.memory_pad_mask.size(1) != len(src):
			self.memory_pad_mask = self.get_pad_mask(src)

		src = self.embedding(src) * math.sqrt(self.d_model)
		src = self.dropout(self.pos_encoder(src))
		tgt = self.embedding(tgt) * math.sqrt(self.d_model)
		tgt = self.dropout(self.pos_encoder(tgt))
		out = self.transformer(src, tgt, 
				tgt_mask = self.tgt_subsequent_mask,
				src_key_padding_mask = self.src_pad_mask,
				tgt_key_padding_mask = self.tgt_pad_mask,
				memory_key_padding_mask = self.memory_pad_mask)
		out = self.out(out)
		return out



class PostionalEncoding(nn.Module):
	"""docstring for PostionEncoder"""
	def __init__(self, d_model, max_len = 5000):
		super(PostionalEncoding, self).__init__()
		pe = torch.zeros(max_len, d_model)
		position = torch.arange(0, max_len, dtype = torch.float).unsqueeze(1)
		div_term = torch.exp( - torch.arange(0, d_model, 2).float() * math.log(10000) / d_model)
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)
		pe = pe.unsqueeze(0).transpose(0,1)
		self.register_buffer('pe', pe)

	def forward(self, x):
		x = x + self.pe[:x.size(0), :]
		return x



def train(model, train_iter, criterion, optimizer, TRG, epoch, writer):
	model.train()
	epoch_loss = 0.0
	epoch_bleu = 0.0
	running_loss = 0.0
	running_bleu = 0.0
	for batch_idx, batch in enumerate(train_iter):
		src = batch.src
		trg = batch.trg
		optimizer.zero_grad()
		output = model(src, trg)
		loss = criterion(output.view(-1, output.shape[-1]), trg.view(-1))
		loss.backward()
		optimizer.step()

		batch_loss = loss.item()
		batch_bleu = utils.count_bleu(output, trg, TRG)
		epoch_loss += batch_loss
		epoch_bleu += batch_bleu
		running_loss += batch_loss
		running_bleu += batch_bleu

		if batch_idx % 1000 == 999:
			writer.add_scalar('train loss',
							  running_loss / 1000,
							  epoch * len(train_iter) + batch_idx)
			writer.add_scalar('train bleu',
							  running_bleu / 1000,
							  epoch * len(train_iter) + batch_idx)

			running_loss = 0.0
			running_bleu = 0.0


	return epoch_loss / len(train_iter), epoch_bleu / len(train_iter)


def evaluate(model, val_iter, criterion, TRG, epoch, writer):
	model.eval()
	epoch_loss = 0.0
	epoch_bleu = 0.0
	for batch_idx, batch in enumerate(val_iter):
		src = batch.src
		tgt = batch.trg
		output = model(src, tgt)
		loss = criterion(output.view(-1, output.shape[-1]), tgt.view(-1))

		epoch_loss += loss.item()
		epoch_bleu += utils.count_bleu(output, tgt, TRG)

		if batch_idx % 25 ==24:
			output = output.permute(1,0,2).max(2)[1]
			output = [' '.join(utils.itos(idx_list, TRG)) for idx_list in output]
			tgt = tgt.permute(1,0)
			tgt = [' '.join(utils.itos(idx_list, TRG)) for idx_list in tgt]
			writer.add_text('output', 
							str(output),
							epoch * len(val_iter) + batch_idx)
			writer.add_text('tgt',
							str(tgt),
							epoch * len(val_iter) + batch_idx)
	
	return epoch_loss / len(val_iter), epoch_bleu / len(val_iter)



		
		
		



